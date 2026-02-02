"""Project entry point (CLI)."""
from __future__ import annotations

import argparse
import os

from .components.vector_store import VectorStore
from .config import load_settings
from .utils.logging_utils import setup_logging
from .graph.builder import build_crag_graph
from .ingestion.mineru_parser import MarkdownHierarchySplitter
from .ingestion.pdf_loader import MinerULoader


def _ingest_file(path: str) -> int:
    """Ingest a single file (MD or PDF) into ChromaDB.

    Args:
        path: Path to the source file.

    Returns:
        int: Number of chunks ingested.
    """
    logger = setup_logging(name="rail-crag.ingest")
    settings = load_settings(require_keys=False)
    
    # 1. Determine Loader
    content = ""
    if path.lower().endswith(".pdf"):
        logger.info("Detected PDF file. Invoking MinerU loader...")
        try:
            loader = MinerULoader()
            content = loader.parse_pdf(path)
        except Exception as exc:
            logger.exception("Failed to parse PDF with MinerU: %s", exc)
            return 0
    else:
        # Default to Markdown text read
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
        except Exception as exc:
            logger.exception("Failed to read file: %s", exc)
            return 0

    # 2. Split & Index
    splitter = MarkdownHierarchySplitter()
    store = VectorStore(settings, logger=logger)

    chunks = splitter.parse(content)
    if not chunks:
        logger.warning("No chunks generated from file: %s", path)
        return 0
        
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
        count = _ingest_file(args.file)
        print(f"Successfully ingested {count} chunks from {args.file}")
    elif args.mode == "chat":
        if not args.query:
            raise ValueError("--query is required for chat mode")
        _chat_query(args.query)


if __name__ == "__main__":
    main()
