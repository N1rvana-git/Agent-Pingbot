"""Ingestion pipeline shared by the CLI and the HTTP API.

There used to be two independent ingestion paths (``src/main.py`` and
``src/ingestion/ingest_mineru.py``) that drifted apart. Everything now goes
through :class:`Ingestor`, which decides how to read a file, how to split it and
how to write it to the store.
"""
from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ..rag.store import ChromaStore
from .mineru_parser import MarkdownHierarchySplitter

SUPPORTED_SUFFIXES = (".md", ".markdown")
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


@dataclass(frozen=True)
class IngestOutcome:
    """Result of ingesting one source file.

    Args:
        source: File path as given by the caller.
        chunks: Number of chunks written.
        error: Optional reason the file was skipped.
    """

    source: str
    chunks: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Whether the file produced at least one chunk."""
        return self.chunks > 0 and self.error is None


def read_text_file(path: str) -> str:
    """Read a text file, tolerating UTF-8 BOM and legacy Chinese encodings.

    Args:
        path: File path.

    Returns:
        str: File contents.

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If no supported encoding decodes the file.
    """
    raw = Path(path).read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, len(raw), f"cannot decode {path}")


class Ingestor:
    """Read, split and index source documents.

    Args:
        store: Target knowledge store.
        splitter: Optional splitter override.
        loader: Optional PDF loader override (tests).
        logger: Optional logger.
    """

    def __init__(
        self,
        store: ChromaStore,
        *,
        splitter: Optional[MarkdownHierarchySplitter] = None,
        loader: Optional[object] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._store = store
        self._splitter = splitter or MarkdownHierarchySplitter()
        self._loader = loader
        self._logger = logger or logging.getLogger(__name__)

    def ingest_file(self, path: str) -> IngestOutcome:
        """Ingest one Markdown or PDF file.

        Args:
            path: Source file path.

        Returns:
            IngestOutcome: Written chunk count or the skip reason.
        """
        source = Path(path)
        if not source.exists():
            return IngestOutcome(str(source), 0, "file not found")

        try:
            content = self._read(source)
        except Exception as exc:  # noqa: BLE001 - report per-file failure, keep going
            self._logger.exception("Failed to read %s", source)
            return IngestOutcome(str(source), 0, str(exc))

        chunks = self._splitter.parse(content)
        if not chunks:
            self._logger.warning("No chunks produced from %s", source)
            return IngestOutcome(str(source), 0, "no chunks produced")

        written = self._store.add_chunks(chunks, source_name=source.name)
        return IngestOutcome(str(source), written)

    def ingest_many(self, paths: Iterable[str]) -> List[IngestOutcome]:
        """Ingest several files, continuing after failures.

        Args:
            paths: Source file paths.

        Returns:
            List[IngestOutcome]: One outcome per file.
        """
        return [self.ingest_file(path) for path in paths]

    def ingest_text(self, markdown: str, source_name: str) -> IngestOutcome:
        """Ingest Markdown supplied directly (HTTP upload).

        Args:
            markdown: Markdown content.
            source_name: Logical source name stored in metadata.

        Returns:
            IngestOutcome: Written chunk count.
        """
        chunks = self._splitter.parse(markdown)
        if not chunks:
            return IngestOutcome(source_name, 0, "no chunks produced")
        written = self._store.add_chunks(chunks, source_name=source_name)
        return IngestOutcome(source_name, written)

    def _read(self, source: Path) -> str:
        """Read a source file, invoking MinerU for PDFs.

        Args:
            source: Source path.

        Returns:
            str: Markdown content.

        Raises:
            RuntimeError: If PDF support is not installed.
        """
        if source.suffix.lower() == ".pdf":
            return self._read_pdf(source)
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"unsupported file type {source.suffix!r}; use one of {SUPPORTED_SUFFIXES} or .pdf"
            )
        return read_text_file(str(source))

    def _read_pdf(self, source: Path) -> str:
        """Convert a PDF to Markdown through MinerU.

        Args:
            source: PDF path.

        Returns:
            str: Markdown content.

        Raises:
            RuntimeError: If ``magic-pdf`` is not installed.
        """
        loader = self._loader
        if loader is None:
            try:
                from .pdf_loader import MinerULoader
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PDF ingestion requires the MinerU toolchain; "
                    "install requirements-pdf.txt and run `magic-pdf` once"
                ) from exc
            loader = MinerULoader()
        return loader.parse_pdf(str(source))


def find_source_files(pattern: str) -> List[str]:
    """Expand a glob into a sorted list of ingestible files.

    Args:
        pattern: Glob pattern (``**`` is recursive).

    Returns:
        List[str]: Matching file paths.
    """
    matches = [
        path
        for path in glob.glob(pattern, recursive=True)
        if Path(path).suffix.lower() in SUPPORTED_SUFFIXES + (".pdf",)
    ]
    return sorted(matches)
