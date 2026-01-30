"""Ingest MinerU markdown outputs into ChromaDB."""
from __future__ import annotations

import glob
import logging
import os
import uuid
from dataclasses import dataclass
from typing import List

from ..config import load_settings
from ..utils.chroma_store import ChromaConfig, get_collection, upsert_texts
from ..utils.logging_utils import setup_logging
from .mineru_parser import MarkdownHierarchySplitter, Chunk


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for ingestion.

    Args:
        input_glob: Glob pattern for MinerU markdown files.
        collection_name: Chroma collection name.
        batch_size: Upsert batch size.
    """

    input_glob: str
    collection_name: str = "rail_crag"
    batch_size: int = 128


def _read_markdown(path: str, logger: logging.Logger) -> str:
    """Read a markdown file with error handling.

    Args:
        path: File path.
        logger: Logger instance.

    Returns:
        str: File contents.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception as exc:
        logger.exception("Failed to read %s: %s", path, exc)
        raise


def _chunks_to_docs(chunks: List[Chunk], source: str) -> tuple[list[str], list[dict], list[str]]:
    """Convert chunks to Chroma documents.

    Args:
        chunks: Parsed chunks.
        source: Source filename.

    Returns:
        tuple: (documents, metadatas, ids)
    """
    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for chunk in chunks:
        documents.append(chunk.content)
        metadata = dict(chunk.metadata)
        metadata["source"] = source
        metadatas.append(metadata)
        ids.append(str(uuid.uuid4()))

    return documents, metadatas, ids


def ingest_markdown(config: IngestConfig) -> None:
    """Ingest MinerU markdown files into ChromaDB.

    Args:
        config: Ingestion configuration.
    """
    logger = setup_logging(name="rail-crag.ingest")
    settings = load_settings()

    splitter = MarkdownHierarchySplitter()
    collection = get_collection(
        ChromaConfig(persist_dir=settings.chroma_persist_dir, collection_name=config.collection_name),
        logger=logger,
    )

    files = glob.glob(config.input_glob, recursive=True)
    if not files:
        logger.warning("No markdown files matched: %s", config.input_glob)
        return

    for path in files:
        logger.info("Ingesting %s", path)
        markdown = _read_markdown(path, logger)
        chunks = splitter.parse(markdown)
        documents, metadatas, ids = _chunks_to_docs(chunks, os.path.basename(path))
        upsert_texts(collection, documents, metadatas, ids, logger=logger)


def main() -> None:
    """CLI entry for ingestion.

    Environment variables:
        INPUT_GLOB: glob for markdown files.
        COLLECTION_NAME: Chroma collection name.
    """
    input_glob = os.getenv("INPUT_GLOB", "./data/**/*.md")
    collection_name = os.getenv("COLLECTION_NAME", "rail_crag")
    ingest_markdown(IngestConfig(input_glob=input_glob, collection_name=collection_name))


if __name__ == "__main__":
    main()
