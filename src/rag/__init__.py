"""Retrieval layer for Rail-CRAG.

Public entry points:

* :class:`~src.rag.store.ChromaStore` — the only module that talks to ChromaDB;
* :class:`~src.rag.retriever.HybridRetriever` — BM25 + vector + RRF + rerank;
* :func:`~src.rag.evaluation.compare_modes` — the ablation used in CI.
"""
from .embeddings import build_embedding_function, describe_embedding
from .evaluation import EvalReport, compare_modes, evaluate, load_eval_cases
from .reranker import HeuristicReranker, Reranker
from .retriever import HybridRetriever
from .store import ChunkRecord, ChromaStore, IndexMismatchError, VectorHit, get_store

__all__ = [
    "ChunkRecord",
    "ChromaStore",
    "EvalReport",
    "HeuristicReranker",
    "HybridRetriever",
    "IndexMismatchError",
    "Reranker",
    "VectorHit",
    "build_embedding_function",
    "compare_modes",
    "describe_embedding",
    "evaluate",
    "get_store",
    "load_eval_cases",
]
