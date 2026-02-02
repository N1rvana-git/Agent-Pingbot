"""ChromaDB storage utilities."""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection


@dataclass(frozen=True)
class ChromaConfig:
    """Configuration for ChromaDB.

    Args:
        persist_dir: Path to persist ChromaDB data.
        collection_name: Name of the collection.
    """

    persist_dir: str
    collection_name: str = "rail_crag"


class SimpleHashEmbeddingFunction:
    """Simple deterministic embedding function (no external dependencies).

    This is a placeholder embedding to allow local Chroma usage without
    additional models. Replace with a real embedding model in production.

    Args:
        dimensions: Vector dimensions.
    """

    def __init__(self, dimensions: int = 128) -> None:
        """Initialize the embedding function.

        Args:
            dimensions: Vector dimensions.
        """
        self._dimensions = dimensions

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed input texts.

        Args:
            input: Input text list.

        Returns:
            List[List[float]]: Embedding vectors.
        """
        return [self._embed_text(text) for text in input]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """Embed documents for indexing.

        Args:
            input: Document texts.

        Returns:
            List[List[float]]: Embedding vectors.
        """
        return self.__call__(input)

    def embed_query(self, input: object) -> List[float]:
        """Embed a single query string.

        Args:
            input: Query text or list of query texts.

        Returns:
            List[float]: Embedding vector.
        """
        if isinstance(input, list):
            query_text = " ".join(str(item) for item in input)
        else:
            query_text = str(input)
        return [self._embed_text(query_text)]

    def name(self) -> str:
        """Return embedding function name.

        Returns:
            str: Name identifier for the embedding function.
        """
        return "simple_hash"

    def get_config(self) -> dict:
        """Return embedding function configuration.

        Returns:
            dict: Configuration metadata.
        """
        return {"dimensions": self._dimensions}

    def _embed_text(self, text: str) -> List[float]:
        """Embed a single text deterministically.

        Args:
            text: Input text.

        Returns:
            List[float]: Embedding vector.
        """
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = digest[0] % self._dimensions
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def get_collection(
    config: ChromaConfig,
    logger: Optional[logging.Logger] = None,
    embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> Collection:
    """Get or create a Chroma collection.

    Args:
        config: Chroma configuration.
        logger: Optional logger.

    Returns:
        Collection: Chroma collection instance.
    """
    log = logger or logging.getLogger(__name__)
    try:
        client = chromadb.PersistentClient(path=config.persist_dir)
        embedder = embedding_function or SimpleHashEmbeddingFunction()
        collection = client.get_or_create_collection(name=config.collection_name, embedding_function=embedder)
        return collection
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Failed to get Chroma collection: %s", exc)
        raise


def upsert_texts(
    collection: Collection,
    texts: List[str],
    metadatas: List[dict],
    ids: List[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upsert texts into Chroma collection.

    Args:
        collection: Chroma collection.
        texts: List of documents.
        metadatas: List of metadata dicts.
        ids: List of ids.
        logger: Optional logger.

    Raises:
        ValueError: If input list lengths mismatch.
    """
    log = logger or logging.getLogger(__name__)
    if not (len(texts) == len(metadatas) == len(ids)):
        raise ValueError("texts, metadatas, and ids must be same length")

    try:
        collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
        log.info("Upserted %d documents into Chroma", len(texts))
    except Exception as exc:
        log.exception("Failed to upsert texts: %s", exc)
        raise


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield items in batches.

    Args:
        items: Items to batch.
        batch_size: Batch size.

    Yields:
        List[str]: Batched items.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def query_texts(
    collection: Collection,
    query: str,
    top_k: int,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Query Chroma collection.

    Args:
        collection: Chroma collection.
        query: Query text.
        top_k: Number of results.
        logger: Optional logger.

    Returns:
        dict: Query results from Chroma.
    """
    log = logger or logging.getLogger(__name__)
    try:
        return collection.query(query_texts=[query], n_results=top_k)
    except Exception as exc:
        log.exception("Failed to query Chroma: %s", exc)
        raise


def get_openai_embedding_function(
    api_key: str,
    model_name: str,
    logger: Optional[logging.Logger] = None,
) -> Callable[[List[str]], List[List[float]]]:
    """Create an OpenAI embedding function for Chroma.

    Args:
        api_key: OpenAI API key.
        model_name: Embedding model name.
        logger: Optional logger.

    Returns:
        Callable[[List[str]], List[List[float]]]: Embedding function.
    """
    log = logger or logging.getLogger(__name__)
    if not api_key:
        log.warning("OPENAI_API_KEY not set; using SimpleHashEmbeddingFunction")
        return SimpleHashEmbeddingFunction()
    return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)
