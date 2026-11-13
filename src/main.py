"""Command line entry point for Rail-CRAG.

Examples
--------
    python -m src.main ingest --file data/railway_standard_guide.md
    python -m src.main ingest --glob "data/**/*.md" --reset
    python -m src.main chat --query "高铁最小曲线半径是多少？"
    python -m src.main stats
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings, SettingsError, load_settings
from .graph.builder import build_crag_graph
from .ingestion.pipeline import Ingestor, find_source_files
from .rag.retriever import HybridRetriever
from .rag.store import ChromaStore
from .utils.logging_utils import configure_stdio, setup_logging


def _build_store(settings: Settings, reset: bool = False) -> ChromaStore:
    """Open the knowledge store, optionally recreating the collection.

    Args:
        settings: Runtime settings.
        reset: Whether to drop the existing collection first.

    Returns:
        ChromaStore: Ready-to-use store.
    """
    store = ChromaStore(settings)
    if reset:
        store.reset()
    return store


def _run_ingest(args: argparse.Namespace) -> int:
    """Ingest one file or a glob pattern.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code.
    """
    logger = setup_logging(name="rail-crag.ingest")
    settings = load_settings(require_keys=False)
    store = _build_store(settings, reset=args.reset)
    ingestor = Ingestor(store, logger=logger)

    sources = [args.file] if args.file else find_source_files(args.glob or "")
    if not sources:
        print("No files matched.", file=sys.stderr)
        return 1

    outcomes = ingestor.ingest_many(sources)
    total = 0
    for outcome in outcomes:
        if outcome.error:
            print(f"SKIP {outcome.source}: {outcome.error}", file=sys.stderr)
        else:
            print(f"OK   {outcome.source}: {outcome.chunks} chunks")
            total += outcome.chunks
    print(f"Ingested {total} chunks into '{store.collection_name}' ({store.persist_dir})")
    return 0 if total else 1


def _run_chat(args: argparse.Namespace) -> int:
    """Answer a single question through the CRAG graph.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code.
    """
    logger = setup_logging(name="rail-crag.chat")
    graph = build_crag_graph()
    result = graph.invoke({"question": args.query})

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.query,
                    "answer": result.get("final_answer", ""),
                    "confidence": result.get("confidence", ""),
                    "retrieval": result.get("retrieval_trace", {}),
                    "sources": [
                        {
                            "path": doc.get("path"),
                            "source": doc.get("source"),
                            "score": doc.get("score"),
                        }
                        for doc in result.get("retrieved_documents", [])
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    logger.info("confidence=%s", result.get("confidence", "unknown"))
    for doc in result.get("retrieved_documents", []):
        logger.info(
            "  [%.3f] %s (%s)", float(doc.get("score") or 0.0), doc.get("path"), doc.get("source")
        )
    print(result.get("final_answer", ""))
    return 0


def _run_stats(args: argparse.Namespace) -> int:
    """Print index and retrieval configuration.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code.
    """
    settings = load_settings(require_keys=False)
    store = ChromaStore(settings)
    retriever = HybridRetriever(
        store,
        rrf_k=settings.rrf_k,
        candidate_pool=settings.candidate_pool,
        default_mode=settings.retrieval_mode,
        default_rerank=settings.rerank,
    )
    print(json.dumps(retriever.stats(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        argparse.ArgumentParser: Parser with ``ingest``, ``chat`` and ``stats``.
    """
    parser = argparse.ArgumentParser(prog="rail-crag", description="Rail-CRAG command line")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ingest = subparsers.add_parser("ingest", help="parse and index documents")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="single .md/.markdown/.pdf file")
    source.add_argument("--glob", help="glob pattern, e.g. \"data/**/*.md\"")
    ingest.add_argument(
        "--reset", action="store_true", help="drop the existing collection before ingesting"
    )

    chat = subparsers.add_parser("chat", help="run one question through CRAG")
    chat.add_argument("--query", required=True, help="question to answer")
    chat.add_argument("--json", action="store_true", help="print the full result as JSON")

    subparsers.add_parser("stats", help="print index and retrieval configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected subcommand.

    Args:
        argv: Optional argument list (tests).

    Returns:
        int: Process exit code.
    """
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "ingest":
            return _run_ingest(args)
        if args.mode == "chat":
            return _run_chat(args)
        return _run_stats(args)
    except SettingsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
