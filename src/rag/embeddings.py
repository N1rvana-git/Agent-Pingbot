"""Embedding backends for the Chroma collection.

Chroma calls the embedding function in two shapes: ``__call__`` / ``embed_query``
with a list of strings, both expected to return a list of vectors. The hashing
backend is fully offline and deterministic; it exists so a fresh clone can run
the full pipeline, the evaluation suite and CI without an API key.

The OpenAI-compatible backend deliberately uses ``urllib`` instead of pulling
another SDK: any gateway exposing ``POST /embeddings`` works, including
self-hosted bge / TEI / vLLM deployments.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Sequence

from ..config import Settings
from .text import HASH_VECTOR_DIMS, hash_vector, l2_normalize, tokenize

logger = logging.getLogger(__name__)

_WARNED_HASHING = False


class HashingEmbeddingFunction:
    """Deterministic character-ngram hashing embeddings (Chroma-compatible).

    Args:
        dims: Vector dimension.
        quiet: Suppress the one-time degradation warning (used by tests).
    """

    def __init__(self, dims: int = HASH_VECTOR_DIMS, quiet: bool = False) -> None:
        global _WARNED_HASHING
        if dims <= 0:
            raise ValueError("dims must be positive")
        self._dims = dims
        if not quiet and not _WARNED_HASHING:
            _WARNED_HASHING = True
            logger.warning(
                "Embedding backend: hashing-%d (no API key / EMBEDDING_BACKEND=hashing). "
                "Semantic matches are lexical only; configure EMBEDDING_BASE_URL and "
                "EMBEDDING_MODEL for real embeddings.",
                dims,
            )

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            input: Document texts.

        Returns:
            list[list[float]]: One vector per document.
        """
        return [hash_vector(tokenize(text), self._dims) for text in input]

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        """Alias of ``__call__`` kept for Chroma compatibility."""
        return self.__call__(input)

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        """Embed a batch of queries (Chroma passes a one-element list)."""
        return self.__call__(input)

    def name(self) -> str:
        """Return the Chroma function name."""
        return f"hashing-{self._dims}"

    def get_config(self) -> dict[str, Any]:
        """Return the persisted configuration."""
        return {"dims": self._dims}

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate a persisted configuration (Chroma protocol hook)."""
        dims = int(config.get("dims", 0))
        if dims <= 0:
            raise ValueError("hashing embedding config requires a positive 'dims'")

    def is_legacy(self) -> bool:
        """Whether this is a pre-config embedding function (always False)."""
        return False

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self._dims

    @property
    def is_semantic(self) -> bool:
        """Whether the vectors carry semantic (model) information."""
        return False


class OpenAICompatibleEmbeddingFunction:
    """Embedding function for any OpenAI-compatible ``/embeddings`` endpoint.

    Args:
        base_url: Gateway root or full ``/embeddings`` URL.
        model: Embedding model name.
        api_key: Optional bearer token.
        dims: Declared dimension (0 = learn it from the first response).
        timeout: Request timeout in seconds.
        quiet: Suppress logging (used by tests).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        dims: int = 0,
        timeout: float = 30.0,
        quiet: bool = False,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._dims = int(dims)
        self._timeout = timeout
        if not quiet:
            logger.info("Embedding backend: %s (%s)", self.name(), self._base_url)

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            input: Document texts.

        Returns:
            list[list[float]]: One normalised vector per text.

        Raises:
            RuntimeError: If the gateway returns an unexpected payload.
        """
        endpoint = (
            self._base_url
            if self._base_url.endswith("/embeddings")
            else f"{self._base_url}/embeddings"
        )
        payload = json.dumps(
            {"model": self._model, "input": list(input)}, ensure_ascii=False
        ).encode("utf-8")
        request = urllib.request.Request(endpoint, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        if self._api_key:
            request.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - surface a readable message
            raise RuntimeError(f"embedding request failed: {exc}") from exc

        items = sorted(body.get("data") or [], key=lambda item: item.get("index", 0))
        if len(items) != len(input):
            raise RuntimeError(
                f"embedding endpoint returned {len(items)} vectors for {len(input)} inputs"
            )
        vectors = [l2_normalize([float(value) for value in item["embedding"]]) for item in items]
        if vectors and not self._dims:
            self._dims = len(vectors[0])
        return vectors

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        """Alias of ``__call__`` kept for Chroma compatibility."""
        return self.__call__(input)

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        """Embed a batch of queries."""
        return self.__call__(input)

    def name(self) -> str:
        """Return the Chroma function name."""
        return f"openai-compatible:{self._model}"

    def get_config(self) -> dict[str, Any]:
        """Return the persisted configuration (never the API key)."""
        return {"base_url": self._base_url, "model": self._model}

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate a persisted configuration (Chroma protocol hook)."""
        if not config.get("base_url") or not config.get("model"):
            raise ValueError("embedding config requires 'base_url' and 'model'")

    def is_legacy(self) -> bool:
        """Whether this is a pre-config embedding function (always False)."""
        return False

    @property
    def dim(self) -> int:
        """Declared or inferred vector dimension."""
        return self._dims

    @property
    def is_semantic(self) -> bool:
        """Whether the vectors carry semantic (model) information."""
        return True


def build_embedding_function(
    settings: Settings,
    *,
    backend: str | None = None,
    quiet: bool = False,
) -> Any:
    """Build the embedding function described by the settings.

    ``auto`` prefers a configured embedding gateway and falls back to hashing
    with a visible warning; ``openai`` without an endpoint is a configuration
    error rather than a silent downgrade.

    Args:
        settings: Runtime settings.
        backend: Optional override (``auto``/``hashing``/``openai``).
        quiet: Suppress logging.

    Returns:
        Any: Chroma-compatible embedding function.

    Raises:
        ValueError: If ``openai`` is requested without an endpoint and model.
    """
    selected = (backend or settings.embedding_backend).lower()

    if selected == "hashing":
        return HashingEmbeddingFunction(quiet=quiet)

    if selected == "openai":
        base_url = settings.embedding_base_url
        if base_url:
            return OpenAICompatibleEmbeddingFunction(
                base_url=base_url,
                model=settings.embedding_model,
                api_key=settings.embedding_api_key,
                dims=settings.embedding_dim,
                quiet=quiet,
            )
        raise ValueError(
            "EMBEDDING_BACKEND=openai requires EMBEDDING_BASE_URL (and EMBEDDING_MODEL)"
        )

    if settings.embedding_base_url and settings.embedding_model:
        return OpenAICompatibleEmbeddingFunction(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            dims=settings.embedding_dim,
            quiet=quiet,
        )

    if not quiet:
        logger.info("No embedding endpoint configured; using deterministic hashing vectors")
    return HashingEmbeddingFunction(quiet=quiet)


def describe_embedding(function: Any) -> dict[str, Any]:
    """Return JSON-serialisable metadata for health checks.

    Args:
        function: Embedding function.

    Returns:
        dict[str, Any]: Backend name, dimension and semantic flag.
    """
    return {
        "backend": str(getattr(function, "name", lambda: "unknown")()),
        "dim": int(getattr(function, "dim", 0) or 0),
        "semantic": bool(getattr(function, "is_semantic", False)),
    }
