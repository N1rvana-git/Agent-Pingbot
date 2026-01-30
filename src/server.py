"""FastAPI service for Rail-CRAG."""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .components.vector_store import VectorStore
from .config import load_settings
from .graph.builder import build_crag_graph
from .ingestion.mineru_parser import MarkdownHierarchySplitter
from .utils.logging_utils import setup_logging

app = FastAPI(title="Rail-CRAG API", version="1.0")
logger = setup_logging(name="rail-crag.api")
settings = load_settings(require_keys=False)
vector_store = VectorStore(settings, logger=logger)
graph = build_crag_graph()


class ChatRequest(BaseModel):
    """Chat request payload."""

    query: str


class IngestRequest(BaseModel):
    """Ingest request payload."""

    markdown_content: str
    source_name: str = "api_upload"


@app.get("/")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "active", "system": "Rail-CRAG"}


@app.post("/chat")
def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    """Run CRAG pipeline for a question.

    Args:
        req: Chat request.

    Returns:
        Dict[str, Any]: Response payload.
    """
    try:
        result = graph.invoke({"question": req.query})
        return {
            "answer": result.get("final_answer", ""),
            "context_source": result.get("confidence", "unknown"),
            "steps": result.get("knowledge_strips", []) + result.get("search_results", []),
        }
    except Exception as exc:
        logger.exception("Chat endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail="chat_failed") from exc


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest) -> Dict[str, Any]:
    """Ingest raw markdown from MinerU.

    Args:
        req: Ingest request.

    Returns:
        Dict[str, Any]: Response payload.
    """
    try:
        splitter = MarkdownHierarchySplitter()
        chunks = splitter.parse(req.markdown_content)
        count = vector_store.add_chunks(chunks, source_name=req.source_name)
        return {"status": "success", "chunks_added": count}
    except Exception as exc:
        logger.exception("Ingest endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail="ingest_failed") from exc


def run() -> None:
    """Entrypoint for ASGI servers."""
    logging.getLogger(__name__).info("Rail-CRAG API ready")
