"""Project entry point (CLI)."""
from __future__ import annotations

import argparse
import os

from .components.vector_store import VectorStore
from .config import load_settings
from .utils.logging_utils import setup_logging
from .graph.builder import build_crag_graph
from .ingestion.mineru_parser import MarkdownHierarchySplitter


def _ingest_markdown(path: str) -> int:
    """Ingest a single Markdown file into ChromaDB.

    Args:
        path: Path to markdown file.

    Returns:
        int: Number of chunks ingested.
    """
    logger = setup_logging(name="rail-crag.ingest")
    settings = load_settings(require_keys=False)
    splitter = MarkdownHierarchySplitter()
    store = VectorStore(settings, logger=logger)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            markdown = handle.read()
    except Exception as exc:
        logger.exception("Failed to read markdown: %s", exc)
        return 0

    chunks = splitter.parse(markdown)
    return store.add_chunks(chunks, source_name=os.path.basename(path))


def _chat_query(query: str) -> None:
    """Run a single query through CRAG graph.

    Args:
        query: User question.
    """
    logger = setup_logging(name="rail-crag")
    app = build_crag_graph()
    user_input = {"question": query}

    final_answer = ""
    for output in app.stream(user_input):
        for key, value in output.items():
            logger.info("Finished step: %s", key)
            if isinstance(value, dict) and "final_answer" in value:
                final_answer = value.get("final_answer", "")

    logger.info("Final output generated.")
    print(final_answer)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Rail-CRAG CLI")
    parser.add_argument("mode", choices=["ingest", "chat"], help="Operation mode")
    parser.add_argument("--file", help="Path to markdown file for ingestion")
    parser.add_argument("--query", help="Question to ask")
    args = parser.parse_args()

    if args.mode == "ingest":
        if not args.file:
            raise ValueError("--file is required for ingest mode")
        _ingest_markdown(args.file)
    elif args.mode == "chat":
        if not args.query:
            raise ValueError("--query is required for chat mode")
        _chat_query(args.query)


if __name__ == "__main__":
    main()
