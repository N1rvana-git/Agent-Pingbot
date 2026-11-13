"""Runtime settings for Rail-CRAG.

Every value comes from an environment variable so the same image can run
offline (deterministic hashing embeddings, no API key) or against an LLM /
embedding gateway. Invalid values fail at startup instead of surfacing as a
confusing runtime error during the first query.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional: .env support for local runs
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()

RETRIEVAL_MODES = frozenset({"hybrid", "bm25", "vector"})
EMBEDDING_BACKENDS = frozenset({"auto", "hashing", "openai"})


class SettingsError(ValueError):
    """Raised when an environment variable has an unusable value."""


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    Args:
        openai_api_key: Key for the chat/embedding gateway (empty = offline mode).
        tavily_api_key: Key for web search (empty = correction step disabled).
        chroma_persist_dir: Directory holding the Chroma index.
        collection_name: Chroma collection used by the knowledge base.
        retriever_k: Number of chunks returned to the graph.
        search_k: Number of web search results.
        retrieval_mode: ``hybrid`` (BM25 + vector + RRF), ``bm25`` or ``vector``.
        rrf_k: Reciprocal-rank-fusion constant.
        rerank: Whether the fused candidate pool is reranked.
        candidate_pool: Fused candidates kept before reranking.
        upper_threshold: CRAG score above which documents are "correct".
        lower_threshold: CRAG score below which documents are "incorrect".
        embedding_backend: ``auto``, ``hashing`` or ``openai``.
        embedding_model: Embedding model name.
        embedding_base_url: OpenAI-compatible embedding endpoint (optional).
        embedding_api_key: Embedding key (falls back to the chat key).
        embedding_dim: Declared embedding dimension (0 = infer from the model).
        eval_model: Model used by the retrieval evaluator.
        gen_model: Model used by the answer generator.
        rewrite_model: Model used to rewrite queries for web search.
    """

    openai_api_key: str
    tavily_api_key: str
    chroma_persist_dir: str
    collection_name: str
    retriever_k: int
    search_k: int
    retrieval_mode: str
    rrf_k: int
    rerank: bool
    candidate_pool: int
    upper_threshold: float
    lower_threshold: float
    embedding_backend: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_dim: int
    eval_model: str
    gen_model: str
    rewrite_model: str

    @property
    def has_chat_model(self) -> bool:
        """Whether an LLM is configured for chat/evaluation."""
        return bool(self.openai_api_key)

    @property
    def has_web_search(self) -> bool:
        """Whether Tavily is configured."""
        return bool(self.tavily_api_key)


def load_settings(require_keys: bool = False) -> Settings:
    """Load settings from environment variables.

    Args:
        require_keys: Enforce that both API keys are present.

    Returns:
        Settings: Validated settings.

    Raises:
        SettingsError: If a variable is malformed or a required key is missing.
    """
    openai_api_key = _env("OPENAI_API_KEY")
    tavily_api_key = _env("TAVILY_API_KEY")

    if require_keys and not openai_api_key:
        raise SettingsError("OPENAI_API_KEY is required")
    if require_keys and not tavily_api_key:
        raise SettingsError("TAVILY_API_KEY is required")

    retrieval_mode = _env("RETRIEVAL_MODE", "hybrid").lower()
    if retrieval_mode not in RETRIEVAL_MODES:
        raise SettingsError(
            f"RETRIEVAL_MODE must be one of {sorted(RETRIEVAL_MODES)}, got {retrieval_mode!r}"
        )

    embedding_backend = _env("EMBEDDING_BACKEND", "auto").lower()
    if embedding_backend not in EMBEDDING_BACKENDS:
        raise SettingsError(
            f"EMBEDDING_BACKEND must be one of {sorted(EMBEDDING_BACKENDS)}, got {embedding_backend!r}"
        )

    retriever_k = _env_int("RETRIEVER_K", 5)
    search_k = _env_int("SEARCH_K", 5)
    rrf_k = _env_int("RRF_K", 60)
    candidate_pool = _env_int("RETRIEVAL_POOL", 20)
    if retriever_k < 1:
        raise SettingsError("RETRIEVER_K must be >= 1")
    if search_k < 1:
        raise SettingsError("SEARCH_K must be >= 1")
    if rrf_k < 1:
        raise SettingsError("RRF_K must be >= 1")
    if candidate_pool < 1:
        raise SettingsError("RETRIEVAL_POOL must be >= 1")

    upper_threshold = _env_float("CRAG_UPPER_THRESHOLD", 0.5)
    lower_threshold = _env_float("CRAG_LOWER_THRESHOLD", -0.5)
    if upper_threshold <= lower_threshold:
        raise SettingsError(
            "CRAG_UPPER_THRESHOLD must be greater than CRAG_LOWER_THRESHOLD "
            f"({upper_threshold} <= {lower_threshold})"
        )

    return Settings(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        chroma_persist_dir=_env("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=_env("COLLECTION_NAME", "rail_crag"),
        retriever_k=retriever_k,
        search_k=search_k,
        retrieval_mode=retrieval_mode,
        rrf_k=rrf_k,
        rerank=_env_bool("RAG_RERANK", True),
        candidate_pool=candidate_pool,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        embedding_backend=embedding_backend,
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-large"),
        embedding_base_url=_env("EMBEDDING_BASE_URL"),
        embedding_api_key=_env("EMBEDDING_API_KEY") or openai_api_key,
        embedding_dim=_env_int("EMBEDDING_DIM", 0),
        eval_model=_env("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
        gen_model=_env("OPENAI_GEN_MODEL", "gpt-4o"),
        rewrite_model=_env("OPENAI_REWRITE_MODEL", "gpt-4o"),
    )
