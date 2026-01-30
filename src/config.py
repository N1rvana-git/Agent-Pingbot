"""Project configuration.

Reads environment variables and exposes typed settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the project.

    Args:
        openai_api_key: OpenAI API key.
        tavily_api_key: Tavily API key.
        chroma_persist_dir: Path for Chroma persistence.
        retriever_k: Number of documents to retrieve.
        search_k: Number of web search results.
        upper_threshold: CRAG upper threshold.
        lower_threshold: CRAG lower threshold.
        embedding_model: Embedding model name.
        eval_model: Evaluator model name.
        gen_model: Generator model name.
        rewrite_model: Query rewrite model name.
    """

    openai_api_key: str
    tavily_api_key: str
    chroma_persist_dir: str
    retriever_k: int
    search_k: int
    upper_threshold: float
    lower_threshold: float
    embedding_model: str
    eval_model: str
    gen_model: str
    rewrite_model: str


def load_settings(require_keys: bool = False) -> Settings:
    """Load settings from environment variables.

    Args:
        require_keys: Whether to enforce API keys presence.

    Returns:
        Settings: Loaded settings object.

    Raises:
        ValueError: If a required environment variable is missing.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma").strip()
    retriever_k = int(os.getenv("RETRIEVER_K", "5"))
    search_k = int(os.getenv("SEARCH_K", "5"))
    upper_threshold = float(os.getenv("CRAG_UPPER_THRESHOLD", "0.5"))
    lower_threshold = float(os.getenv("CRAG_LOWER_THRESHOLD", "-0.5"))
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
    eval_model = os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini").strip()
    gen_model = os.getenv("OPENAI_GEN_MODEL", "gpt-4o").strip()
    rewrite_model = os.getenv("OPENAI_REWRITE_MODEL", "gpt-4o").strip()

    if require_keys and not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if require_keys and not tavily_api_key:
        raise ValueError("TAVILY_API_KEY is required")

    return Settings(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        chroma_persist_dir=chroma_persist_dir,
        retriever_k=retriever_k,
        search_k=search_k,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        embedding_model=embedding_model,
        eval_model=eval_model,
        gen_model=gen_model,
        rewrite_model=rewrite_model,
    )
