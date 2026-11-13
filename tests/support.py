"""Shared test helpers."""
from __future__ import annotations

import atexit
import dataclasses
import shutil
import tempfile
import unittest
from typing import Any, Dict, Optional, Sequence

from src.config import Settings, load_settings
from src.rag.store import ChunkRecord, VectorHit

_TEMP_ROOT = tempfile.mkdtemp(prefix="rail-crag-tests-")
atexit.register(shutil.rmtree, _TEMP_ROOT, True)


def make_settings(**overrides: Any) -> Settings:
    """Build isolated, offline settings for a test.

    Args:
        **overrides: Fields to replace on the base settings.

    Returns:
        Settings: Settings with temp Chroma directory and no API keys.
    """
    base = load_settings(require_keys=False)
    values: Dict[str, Any] = {
        "openai_api_key": "",
        "tavily_api_key": "",
        "embedding_backend": "hashing",
        "embedding_base_url": "",
        "chroma_persist_dir": tempfile.mkdtemp(prefix="case-", dir=_TEMP_ROOT),
        "collection_name": "test_collection",
        "retrieval_mode": "hybrid",
        "rerank": True,
        "retriever_k": 5,
    }
    values.update(overrides)
    return dataclasses.replace(base, **values)


class TempDirMixin(unittest.TestCase):
    """TestCase mixin that removes temporary directories created by ``temp_dir``."""

    def temp_dir(self) -> str:
        """Create and register a temporary directory.

        Returns:
            str: Directory path.
        """
        path = tempfile.mkdtemp(prefix="rail-crag-case-")
        self.addCleanup(shutil.rmtree, path, True)
        return path


class FakeStore:
    """In-memory store double for retriever tests.

    Args:
        chunks: Corpus chunks.
        vector_order: Chunk ids returned by the dense leg, best first.
        vector_error: When set, the dense leg raises this message.
    """

    def __init__(
        self,
        chunks: Sequence[ChunkRecord],
        *,
        vector_order: Optional[Sequence[str]] = None,
        vector_error: Optional[str] = None,
    ) -> None:
        self._chunks = list(chunks)
        self._vector_order = list(vector_order or [])
        self._vector_error = vector_error

    def count(self) -> int:
        """Return the corpus size."""
        return len(self._chunks)

    def all_chunks(self) -> list[ChunkRecord]:
        """Return the corpus."""
        return list(self._chunks)

    def vector_search(self, query: str, k: int) -> list[VectorHit]:
        """Return the configured dense ranking."""
        if self._vector_error:
            raise RuntimeError(self._vector_error)
        return [
            VectorHit(chunk_id=chunk_id, distance=0.5, score=0.75)
            for chunk_id in self._vector_order[:k]
        ]

    def stats(self) -> dict[str, Any]:
        """Return fake index metadata."""
        return {"collection": "fake", "chunks": len(self._chunks)}
