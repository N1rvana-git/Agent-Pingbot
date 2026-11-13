"""Chroma persistence layer for the knowledge base.

This module is the single place that talks to ChromaDB. It owns three
responsibilities and nothing else:

* write parsed chunks with deterministic ids (re-ingesting a file is idempotent);
* expose the whole corpus so BM25 can index it;
* run vector queries for the dense retrieval leg.

Keeping it separate from parsing and from the retriever is what makes the
hybrid retrieval testable with a fake store and keeps Chroma from leaking into
graph code.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import chromadb
from chromadb.config import Settings as ChromaClientSettings

from ..config import Settings
from .embeddings import build_embedding_function, describe_embedding

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 256
_READ_PAGE_SIZE = 500


class IndexMismatchError(RuntimeError):
    """Raised when the persisted index cannot serve the current embedding backend."""


class TextChunkLike(Protocol):
    """Minimal shape required to index a chunk (structurally typed)."""

    content: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ChunkRecord:
    """A stored corpus chunk.

    Args:
        chunk_id: Stable id derived from source, path and content.
        content: Chunk text.
        metadata: Flat scalar metadata (source, path, part, ...).
    """

    chunk_id: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorHit:
    """One dense-retrieval hit.

    Args:
        chunk_id: Stable chunk id.
        distance: Raw Chroma distance (squared L2 for normalised vectors).
        score: Cosine-like similarity in ``[-1, 1]``.
    """

    chunk_id: str
    distance: float
    score: float


def _clean_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only scalar values that Chroma accepts.

    Args:
        metadata: Arbitrary metadata.

    Returns:
        dict[str, Any]: Filtered metadata.
    """
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    return cleaned


def _chunk_id(source: str, chunk: TextChunkLike, index: int) -> str:
    """Build a deterministic chunk id.

    Args:
        source: Source file name.
        chunk: Parsed chunk.
        index: Position within the file.

    Returns:
        str: Stable hexadecimal id.
    """
    path = str(chunk.metadata.get("path") or "")
    digest = hashlib.sha1(
        f"{source}|{path}|{index}|{chunk.content}".encode("utf-8")
    ).hexdigest()
    return digest[:20]


class ChromaStore:
    """Chroma-backed knowledge store.

    Args:
        settings: Runtime settings.
        embedding_function: Optional embedding function override (tests, CLI).
        persist_dir: Optional index directory override.
        logger: Optional logger.
    """

    def __init__(
        self,
        settings: Settings,
        embedding_function: Optional[Any] = None,
        persist_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._embedding = embedding_function or build_embedding_function(settings)
        self._collection_name = settings.collection_name
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaClientSettings(anonymized_telemetry=False),
        )
        self._collection = self._open_collection()

    # -- collection management -------------------------------------------
    def _open_collection(self) -> Any:
        """Open or create the collection with a compatible embedding function.

        Returns:
            Any: Chroma collection.

        Raises:
            IndexMismatchError: If the index was built with another backend.
        """
        metadata = {
            "embedding_backend": str(getattr(self._embedding, "name")()),
            "embedding_dim": int(getattr(self._embedding, "dim", 0) or 0),
        }
        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding,
            metadata=metadata,
        )
        stored = collection.metadata or {}
        stored_backend = str(stored.get("embedding_backend") or "")
        stored_dim = int(stored.get("embedding_dim") or 0)
        current_backend = metadata["embedding_backend"]
        current_dim = metadata["embedding_dim"]
        if stored_backend and stored_backend != current_backend:
            raise IndexMismatchError(
                f"collection {self._collection_name!r} was built with {stored_backend!r} "
                f"but the current backend is {current_backend!r}; re-ingest with "
                "`python -m src.main ingest --file <path> --reset`"
            )
        if stored_dim and current_dim and stored_dim != current_dim:
            raise IndexMismatchError(
                f"collection {self._collection_name!r} stores {stored_dim}-dim vectors "
                f"but the current backend produces {current_dim}-dim; re-ingest with --reset"
            )
        return collection

    def reset(self) -> None:
        """Delete and recreate the collection (used by ``ingest --reset``)."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - missing collection is fine
            pass
        self._collection = self._open_collection()

    @property
    def collection_name(self) -> str:
        """Name of the underlying Chroma collection."""
        return self._collection_name

    @property
    def persist_dir(self) -> str:
        """Directory holding the persisted index."""
        return self._persist_dir

    def count(self) -> int:
        """Return the number of stored chunks."""
        return int(self._collection.count())

    # -- writes -----------------------------------------------------------
    def add_chunks(self, chunks: Iterable[TextChunkLike], source_name: str) -> int:
        """Upsert parsed chunks under deterministic ids.

        Args:
            chunks: Parsed chunks (content + metadata).
            source_name: Source file name stored in metadata.

        Returns:
            int: Number of chunks written.
        """
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        seen: set[str] = set()

        for index, chunk in enumerate(chunks):
            content = (chunk.content or "").strip()
            if not content:
                continue
            chunk_id = _chunk_id(source_name, chunk, index)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            metadata = _clean_metadata({**dict(chunk.metadata), "source": source_name})
            metadata.setdefault("source", source_name)
            documents.append(content)
            metadatas.append(metadata)
            ids.append(chunk_id)

        if not documents:
            return 0

        # Replace this source atomically at the chunk level: re-ingesting an
        # updated standard must not leave stale chunks behind.
        try:
            self._collection.delete(where={"source": source_name})
        except Exception as exc:  # noqa: BLE001 - empty/new collection
            self._logger.debug("No previous chunks to replace for %s: %s", source_name, exc)

        for start in range(0, len(documents), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        self._logger.info("Upserted %d chunks from %s", len(documents), source_name)
        return len(documents)

    # -- reads ------------------------------------------------------------
    def all_chunks(self) -> list[ChunkRecord]:
        """Read the whole corpus, ordered by chunk id.

        Returns:
            list[ChunkRecord]: Stored chunks.
        """
        records: list[ChunkRecord] = []
        offset = 0
        while True:
            page = self._collection.get(
                limit=_READ_PAGE_SIZE,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids: Sequence[str] = page.get("ids") or []
            if not ids:
                break
            documents: Sequence[Optional[str]] = page.get("documents") or [None] * len(ids)
            metadatas: Sequence[Optional[Mapping[str, Any]]] = page.get("metadatas") or [None] * len(ids)
            for chunk_id, document, metadata in zip(ids, documents, metadatas):
                records.append(
                    ChunkRecord(
                        chunk_id=str(chunk_id),
                        content=str(document or ""),
                        metadata=dict(metadata or {}),
                    )
                )
            offset += len(ids)
            if len(ids) < _READ_PAGE_SIZE:
                break
        records.sort(key=lambda record: record.chunk_id)
        return records

    def vector_search(self, query: str, k: int) -> list[VectorHit]:
        """Run a dense query and convert distances to similarities.

        Chroma's default space is squared L2; with unit-length vectors
        ``distance == 2 - 2 * cosine``, so ``1 - distance / 2`` recovers the
        cosine similarity without depending on the collection configuration.

        Args:
            query: Query text.
            k: Number of neighbours.

        Returns:
            list[VectorHit]: Hits ordered by similarity.

        Raises:
            IndexMismatchError: If the persisted dimension differs from the backend.
        """
        if not query.strip() or k < 1 or self.count() == 0:
            return []
        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=max(1, k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001 - Chroma raises several types
            message = str(exc).lower()
            if "dimension" in message:
                raise IndexMismatchError(
                    "index dimension does not match the embedding backend; "
                    "re-ingest with `python -m src.main ingest --file <path> --reset`"
                ) from exc
            raise

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[VectorHit] = []
        for chunk_id, distance in zip(ids, distances):
            distance_value = float(distance)
            score = max(-1.0, min(1.0, 1.0 - distance_value / 2.0))
            hits.append(VectorHit(chunk_id=str(chunk_id), distance=distance_value, score=score))
        return hits

    def stats(self) -> dict[str, Any]:
        """Return index metadata for ``/healthz`` and the evaluation report.

        Returns:
            dict[str, Any]: Collection name, chunk count and embedding backend.
        """
        return {
            "collection": self._collection_name,
            "chunks": self.count(),
            "persist_dir": self._persist_dir,
            "embedding": describe_embedding(self._embedding),
        }


@lru_cache(maxsize=4)
def get_store(settings: Settings) -> ChromaStore:
    """Return a cached store for the given settings.

    A process-wide cache avoids reopening the Chroma client on every turn while
    keeping different settings (notably test temp directories) isolated.

    Args:
        settings: Runtime settings.

    Returns:
        ChromaStore: Shared store instance.
    """
    return ChromaStore(settings)
