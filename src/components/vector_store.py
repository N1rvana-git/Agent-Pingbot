"""Vector store wrapper for ChromaDB."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from ..config import Settings
from ..utils.chroma_store import ChromaConfig, get_collection, get_openai_embedding_function, query_texts
from ..ingestion.mineru_parser import Chunk


@dataclass(frozen=True)
class RetrievedDoc:
    """Retrieved document structure.

    Args:
        doc_id: Document id.
        content: Document content.
        metadata: Document metadata.
    """

    doc_id: str
    content: str
    metadata: dict


class VectorStore:
    """ChromaDB vector store wrapper."""

    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        """Initialize vector store.

        Args:
            settings: Project settings.
            logger: Optional logger.
        """
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        embedding_fn = get_openai_embedding_function(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
            logger=self._logger,
        )
        self._collection = get_collection(
            ChromaConfig(persist_dir=settings.chroma_persist_dir),
            logger=self._logger,
            embedding_function=embedding_fn,
        )

    def search(self, query: str, k: int) -> List[RetrievedDoc]:
        """Search top-k documents.

        Args:
            query: Query text.
            k: Top-k results.

        Returns:
            List[RetrievedDoc]: Retrieved documents.
        """
        try:
            result = query_texts(self._collection, query, k, logger=self._logger)
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            ids = result.get("ids", [[]])[0]
            return [
                RetrievedDoc(doc_id=doc_id, content=doc, metadata=meta)
                for doc_id, doc, meta in zip(ids, documents, metadatas)
            ]
        except Exception as exc:
            self._logger.exception("VectorStore search failed: %s", exc)
            return []

    def add_chunks(self, chunks: List[Chunk], source_name: str) -> int:
        """Upsert chunks into the collection.

        Args:
            chunks: Parsed chunks.
            source_name: Source filename.

        Returns:
            int: Number of upserted chunks.
        """
        if not chunks:
            return 0
        try:
            documents: List[str] = []
            metadatas: List[dict] = []
            ids: List[str] = []
            for chunk in chunks:
                documents.append(chunk.content)
                metadata = dict(chunk.metadata)
                metadata["source"] = source_name
                metadatas.append(metadata)
                ids.append(str(uuid.uuid4()))
            self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            self._logger.info("Upserted %d chunks to Chroma", len(chunks))
            return len(chunks)
        except Exception as exc:
            self._logger.exception("VectorStore upsert failed: %s", exc)
            return 0
