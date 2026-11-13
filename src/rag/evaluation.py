"""Retrieval evaluation: turn "did the RAG change help?" into numbers.

The project follows the CRAG paper, so the retrieval layer needs the same kind
of evidence as the generation layer. Metrics are the standard IR trio computed
per query with binary relevance (a hit is a result containing one of the
case's expected keywords):

* ``hit_rate@k``: share of queries with at least one hit in the top k;
* ``MRR``: mean reciprocal rank of the first hit;
* ``nDCG@k``: ranking quality, ``DCG = 1/log2(rank+1)`` for the first hit;
* ``latency_ms``: mean retrieval latency, including reranking.

Cases live in ``tests/data/retrieval_eval.json`` so the same suite runs in CI
and in the ablation CLI.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

DEFAULT_CASES_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "retrieval_eval.json"


@dataclass
class EvalReport:
    """Aggregated metrics for one retrieval configuration.

    Args:
        name: Configuration label.
        queries: Number of evaluated queries.
        top_k: Cut-off rank.
        hit_rate: ``hit_rate@k``.
        mrr: Mean reciprocal rank.
        ndcg: Mean nDCG@k.
        latency_ms: Mean latency.
        failures: Query ids with no hit.
    """

    name: str
    queries: int
    top_k: int
    hit_rate: float
    mrr: float
    ndcg: float
    latency_ms: float
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        Returns:
            dict[str, Any]: Rounded metrics.
        """
        return {
            "name": self.name,
            "queries": self.queries,
            "top_k": self.top_k,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "latency_ms": round(self.latency_ms, 1),
            "failures": list(self.failures),
        }


def load_eval_cases(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load and validate retrieval evaluation cases.

    Args:
        path: Optional cases file.

    Returns:
        list[dict[str, Any]]: Evaluation cases.

    Raises:
        ValueError: If the file is not a list or a case is incomplete.
    """
    cases_path = path or DEFAULT_CASES_PATH
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{cases_path} must contain a JSON array")
    for case in cases:
        if not case.get("id") or not case.get("query") or not case.get("expected_keywords"):
            raise ValueError(f"{cases_path} has an incomplete case: {case!r}")
    return cases


def _is_hit(item: dict[str, Any], keywords: Sequence[str]) -> bool:
    """Whether a retrieval result contains any expected keyword.

    Args:
        item: Retrieval result.
        keywords: Expected keywords.

    Returns:
        bool: True on the first match.
    """
    haystack = f"{item.get('path') or ''}\n{item.get('content') or ''}".lower()
    return any(str(keyword).lower() in haystack for keyword in keywords)


def evaluate(
    retriever: Any,
    cases: Iterable[dict[str, Any]],
    *,
    name: str = "hybrid",
    top_k: int = 5,
    mode: str = "hybrid",
    rerank: Optional[bool] = None,
) -> EvalReport:
    """Evaluate one retrieval configuration.

    Args:
        retriever: Object exposing ``retrieve(query, top_k=..., mode=..., rerank=...)``.
        cases: Evaluation cases.
        name: Configuration label.
        top_k: Cut-off rank.
        mode: Retrieval mode.
        rerank: Rerank override.

    Returns:
        EvalReport: Aggregated metrics.
    """
    case_list = list(cases)
    hits = 0
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    elapsed_total = 0.0
    failures: list[str] = []

    for case in case_list:
        started = time.perf_counter()
        results = retriever.retrieve(
            case["query"],
            top_k=top_k,
            mode=mode,
            rerank=rerank,
        )
        elapsed_total += (time.perf_counter() - started) * 1000

        first_hit_rank = next(
            (
                rank
                for rank, item in enumerate(results, start=1)
                if _is_hit(item, case["expected_keywords"])
            ),
            None,
        )
        if first_hit_rank is None:
            failures.append(str(case["id"]))
            continue

        hits += 1
        reciprocal_rank_sum += 1.0 / first_hit_rank
        ndcg_sum += 1.0 / math.log2(first_hit_rank + 1)

    total = max(1, len(case_list))
    return EvalReport(
        name=name,
        queries=len(case_list),
        top_k=top_k,
        hit_rate=hits / total,
        mrr=reciprocal_rank_sum / total,
        ndcg=ndcg_sum / total,
        latency_ms=elapsed_total / total,
        failures=failures,
    )


def compare_modes(
    retriever: Any,
    cases: Optional[Sequence[dict[str, Any]]] = None,
    *,
    top_k: int = 5,
) -> list[EvalReport]:
    """Run the four standard ablation configurations.

    Args:
        retriever: Hybrid retriever.
        cases: Optional cases (defaults to the bundled suite).
        top_k: Cut-off rank.

    Returns:
        list[EvalReport]: BM25, vector, hybrid and hybrid+rerank reports.
    """
    case_list = list(cases) if cases is not None else load_eval_cases()
    return [
        evaluate(retriever, case_list, name="BM25", top_k=top_k, mode="bm25", rerank=False),
        evaluate(retriever, case_list, name="Vector", top_k=top_k, mode="vector", rerank=False),
        evaluate(retriever, case_list, name="Hybrid (RRF)", top_k=top_k, mode="hybrid", rerank=False),
        evaluate(retriever, case_list, name="Hybrid + Rerank", top_k=top_k, mode="hybrid", rerank=True),
    ]
