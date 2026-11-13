"""FastAPI service for Rail-CRAG."""
from __future__ import annotations

import dataclasses
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .graph.builder import build_crag_graph
from .graph.nodes import CRAGNodes
from .ingestion.pipeline import Ingestor
from .rag.retriever import HybridRetriever
from .rag.store import IndexMismatchError, get_store
from .utils.logging_utils import setup_logging

logger = setup_logging(name="rail-crag.api")
settings = load_settings(require_keys=False)
store = get_store(settings)

_RETRIEVER: Optional[HybridRetriever] = None
_GRAPH_CACHE: Dict[tuple[str, bool], Any] = {}


def _default_retriever() -> HybridRetriever:
    """Return the process-wide retriever used by ``/retrieve`` and warmup.

    Returns:
        HybridRetriever: Cached retriever configured from settings.
    """
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever(
            store,
            rrf_k=settings.rrf_k,
            candidate_pool=settings.candidate_pool,
            default_mode=settings.retrieval_mode,
            default_rerank=settings.rerank,
            logger=logger,
        )
    return _RETRIEVER


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the retrieval index so the first query does not pay for it.

    Args:
        _app: FastAPI application.

    Yields:
        None: Control back to the server.
    """
    try:
        logger.info("retrieval warmup: %s", _default_retriever().warmup())
    except Exception:  # noqa: BLE001 - an empty index is a valid state
        logger.warning("retrieval warmup skipped", exc_info=True)
    yield


app = FastAPI(title="Rail-CRAG API", version="1.1", lifespan=lifespan)

RetrievalMode = Literal["hybrid", "bm25", "vector"]


class ChatRequest(BaseModel):
    """Chat request payload."""

    query: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[RetrievalMode] = None
    rerank: Optional[bool] = None


class RetrieveRequest(BaseModel):
    """Retrieval-only request payload (ablation / debugging)."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=10)


class IngestRequest(BaseModel):
    """Markdown ingest request payload."""

    markdown_content: str = Field(..., min_length=1)
    source_name: str = Field("api_upload", min_length=1, max_length=200)


def _graph_for(mode: Optional[str], rerank: Optional[bool]) -> Any:
    """Return a compiled graph for the requested retrieval configuration.

    Args:
        mode: Optional retrieval mode override.
        rerank: Optional rerank override.

    Returns:
        Any: Compiled LangGraph application.
    """
    resolved_mode = mode or settings.retrieval_mode
    resolved_rerank = settings.rerank if rerank is None else rerank
    key = (resolved_mode, resolved_rerank)
    if key not in _GRAPH_CACHE:
        config: Settings = dataclasses.replace(
            settings, retrieval_mode=resolved_mode, rerank=resolved_rerank
        )
        _GRAPH_CACHE[key] = build_crag_graph(CRAGNodes(config, logger=logger))
    return _GRAPH_CACHE[key]


@app.get("/")
def health_check() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "active", "system": "Rail-CRAG"}


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Expose index, retrieval and model configuration."""
    stats = store.stats()
    return {
        "status": "ok",
        "index": stats,
        "retrieval": {
            "mode": settings.retrieval_mode,
            "rerank": settings.rerank,
            "rrf_k": settings.rrf_k,
            "candidate_pool": settings.candidate_pool,
            "retriever_k": settings.retriever_k,
        },
        "models": {
            "chat": settings.has_chat_model,
            "web_search": settings.has_web_search,
            "eval_model": settings.eval_model,
            "gen_model": settings.gen_model,
        },
    }


@app.post("/chat")
def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    """Run the CRAG pipeline for one question.

    Args:
        req: Chat request.

    Returns:
        Dict[str, Any]: Answer with provenance and retrieval diagnostics.
    """
    started = time.perf_counter()
    try:
        result = _graph_for(req.mode, req.rerank).invoke({"question": req.query})
    except IndexMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - keep the API contract stable
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail="chat_failed") from exc

    documents = result.get("retrieved_documents", [])
    return {
        "answer": result.get("final_answer", ""),
        "confidence": result.get("confidence", "unknown"),
        "retrieval": result.get("retrieval_trace", {}),
        "sources": [
            {
                "id": doc.get("id"),
                "path": doc.get("path"),
                "source": doc.get("source"),
                "score": doc.get("score"),
            }
            for doc in documents
        ],
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest) -> Dict[str, Any]:
    """Return BM25 / vector / hybrid / rerank results for one query.

    Args:
        req: Retrieval request.

    Returns:
        Dict[str, Any]: Four-way comparison with latencies.
    """
    return _default_retriever().retrieve_with_trace(req.query, top_k=req.top_k)


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest) -> Dict[str, Any]:
    """Ingest raw Markdown produced by MinerU or written by hand.

    Args:
        req: Ingest request.

    Returns:
        Dict[str, Any]: Ingest outcome.
    """
    ingestor = Ingestor(store, logger=logger)
    outcome = ingestor.ingest_text(req.markdown_content, req.source_name)
    if outcome.error:
        raise HTTPException(status_code=400, detail=outcome.error)
    # Drop cached BM25 indexes and graphs so the next query sees the new corpus.
    _GRAPH_CACHE.clear()
    _default_retriever().reload()
    return {"status": "ok", "source": outcome.source, "chunks_added": outcome.chunks}
