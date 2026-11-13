"""Retrieval ablation CLI: BM25 vs Vector vs Hybrid(RRF) vs +Rerank.

Examples
--------
    python -m src.main ingest --file data/railway_standard_guide.md --reset
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --top-k 3 --json
    python scripts/eval_retrieval.py --min-hit-rate 0.75   # CI gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.rag.evaluation import compare_modes, load_eval_cases  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.rag.store import ChromaStore  # noqa: E402
from src.utils.logging_utils import configure_stdio  # noqa: E402


def main() -> int:
    """Run the ablation and optionally enforce a hit-rate gate.

    Returns:
        int: Exit code (``0`` pass, ``1`` gate failed, ``2`` empty index).
    """
    configure_stdio()
    parser = argparse.ArgumentParser(description="Rail-CRAG retrieval ablation")
    parser.add_argument("--top-k", type=int, default=5, help="cut-off rank")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.0,
        help="fail (exit 1) when hybrid+rerank hit rate is below this value",
    )
    parser.add_argument("--cases", type=Path, default=None, help="custom evaluation cases")
    args = parser.parse_args()

    settings = load_settings(require_keys=False)
    store = ChromaStore(settings)
    if store.count() == 0:
        print(
            "The index is empty. Run:\n"
            "  python -m src.main ingest --file data/railway_standard_guide.md --reset",
            file=sys.stderr,
        )
        return 2

    retriever = HybridRetriever(
        store,
        rrf_k=settings.rrf_k,
        candidate_pool=settings.candidate_pool,
        default_mode=settings.retrieval_mode,
        default_rerank=settings.rerank,
    )
    cases = load_eval_cases(args.cases)
    reports = compare_modes(retriever, cases, top_k=args.top_k)

    if args.json:
        print(
            json.dumps(
                {
                    "corpus": retriever.stats(),
                    "cases": len(cases),
                    "top_k": args.top_k,
                    "reports": [report.as_dict() for report in reports],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"cases={len(cases)} top_k={args.top_k} corpus={retriever.stats()}")
        header = f"{'configuration':<20}{'HitRate@k':>12}{'MRR':>10}{'nDCG@k':>10}{'latency(ms)':>14}"
        print(header)
        print("-" * len(header))
        for report in reports:
            print(
                f"{report.name:<20}{report.hit_rate:>12.1%}{report.mrr:>10.3f}"
                f"{report.ndcg:>10.3f}{report.latency_ms:>14.1f}"
            )
        baseline, reranked = reports[2], reports[3]
        print()
        print(
            f"rerank delta: HitRate {baseline.hit_rate:.1%} -> {reranked.hit_rate:.1%}, "
            f"MRR {baseline.mrr:.3f} -> {reranked.mrr:.3f}, "
            f"nDCG {baseline.ndcg:.3f} -> {reranked.ndcg:.3f}"
        )
        if baseline.failures:
            print(f"hybrid misses: {baseline.failures}")

    best = reports[3]
    if best.hit_rate < args.min_hit_rate:
        print(
            f"gate failed: hybrid+rerank hit rate {best.hit_rate:.3f} "
            f"< required {args.min_hit_rate:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
