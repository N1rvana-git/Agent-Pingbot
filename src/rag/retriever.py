"""Hybrid retrieval: BM25 + dense vectors + Reciprocal Rank Fusion.

Why two legs
------------
Chinese railway queries mix exact identifiers (``TB 10621``, ``CRTS III``,
``SIL4``) with paraphrases ("列车跑起来会不会撞到隧道壁"). BM25 wins on the
first kind, dense vectors on the second, and neither is reliable alone.

Why RRF
-------
BM25 scores and cosine similarities live on different scales, so a weighted
average needs per-corpus tuning. Reciprocal Rank Fusion only reads ranks:

    score(chunk) = sum(1 / (k + rank_i)) for every leg that returned it

which needs no score calibration and is the standard default for hybrid search.

Failure behaviour
-----------------
The dense leg is allowed to fail (stale index, embedding gateway down): the
retriever degrades to BM25 and reports ``vector_error`` in the trace instead of
failing the request. A query still gets an answer, and the degradation is
visible rather than silent.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .reranker import HeuristicReranker, Reranker
from .store import ChunkRecord, ChromaStore
from .text import tokenize

BM25_K1 = 1.5
BM25_B = 0.75
MODES = frozenset({"hybrid", "bm25", "vector"})


@dataclass(frozen=True)
class Bm25Index:
    """Pre-computed corpus statistics.

    Args:
        term_frequencies: Per-chunk token counts.
        document_lengths: Per-chunk token counts total.
        document_frequency: Number of chunks containing each token.
        average_length: Mean chunk length.
    """

    term_frequencies: tuple[Counter[str], ...]
    document_lengths: tuple[int, ...]
    document_frequency: dict[str, int]
    average_length: float


def build_bm25_index(chunks: Sequence[ChunkRecord]) -> Bm25Index:
    """Build the BM25 statistics once per corpus version.

    Args:
        chunks: Corpus chunks.

    Returns:
        Bm25Index: Pre-computed statistics.
    """
    term_frequencies: list[Counter[str]] = []
    document_frequency: dict[str, int] = {}
    document_lengths: list[int] = []

    for chunk in chunks:
        tokens = tokenize(f"{chunk.content}\n{chunk.metadata.get('path', '')}")
        counts = Counter(tokens)
        term_frequencies.append(counts)
        document_lengths.append(len(tokens))
        for token in counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    average = sum(document_lengths) / max(1, len(document_lengths))
    return Bm25Index(tuple(term_frequencies), tuple(document_lengths), document_frequency, average)


def bm25_rank(
    query: str,
    chunks: Sequence[ChunkRecord],
    index: Bm25Index,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[tuple[float, ChunkRecord]]:
    """Rank chunks with Okapi BM25.

    Args:
        query: Query text.
        chunks: Corpus chunks (same order as the index).
        index: Pre-computed statistics.
        k1: Term-frequency saturation.
        b: Length-normalisation strength.

    Returns:
        list[tuple[float, ChunkRecord]]: Positive scores, descending.
    """
    if not query.strip() or not chunks:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    document_count = len(chunks)
    idf: dict[str, float] = {}
    for token in set(query_tokens):
        frequency = index.document_frequency.get(token, 0)
        if frequency:
            idf[token] = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )

    scored: list[tuple[float, ChunkRecord]] = []
    for position, chunk in enumerate(chunks):
        counts = index.term_frequencies[position]
        length = index.document_lengths[position]
        norm = k1 * (1.0 - b + b * length / max(1.0, index.average_length))
        score = 0.0
        for token in query_tokens:
            term_frequency = counts.get(token, 0)
            if not term_frequency:
                continue
            token_idf = idf.get(token)
            if token_idf is None:
                continue
            score += token_idf * (term_frequency * (k1 + 1.0)) / (term_frequency + norm)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return scored


class HybridRetriever:
    """BM25 + dense retrieval with RRF fusion and optional reranking.

    Args:
        store: Knowledge store.
        rrf_k: RRF constant.
        candidate_pool: Fused candidates kept before reranking.
        default_mode: ``hybrid``, ``bm25`` or ``vector``.
        default_rerank: Whether fused results are reranked by default.
        reranker: Optional reranker override.
        logger: Optional logger.
    """

    def __init__(
        self,
        store: ChromaStore,
        *,
        rrf_k: int = 60,
        candidate_pool: int = 20,
        default_mode: str = "hybrid",
        default_rerank: bool = True,
        reranker: Optional[Reranker] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if default_mode not in MODES:
            raise ValueError(f"unknown retrieval mode: {default_mode!r}")
        self._store = store
        self._rrf_k = max(1, rrf_k)
        self._candidate_pool = max(1, candidate_pool)
        self._default_mode = default_mode
        self._default_rerank = default_rerank
        self._reranker = reranker or HeuristicReranker()
        self._logger = logger or logging.getLogger(__name__)

        self._chunks: list[ChunkRecord] = []
        self._chunk_by_id: dict[str, ChunkRecord] = {}
        self._index: Optional[Bm25Index] = None
        self._corpus_size: Optional[int] = None

    # -- corpus -----------------------------------------------------------
    def _ensure_corpus(self) -> None:
        """Rebuild the BM25 index when the collection size changed."""
        size = self._store.count()
        if self._index is not None and self._corpus_size == size:
            return
        self._chunks = self._store.all_chunks()
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._index = build_bm25_index(self._chunks)
        self._corpus_size = size
        self._logger.info("BM25 index built over %d chunks", len(self._chunks))

    def reload(self) -> None:
        """Force the next query to rebuild the corpus cache."""
        self._index = None
        self._corpus_size = None
        self._chunks = []
        self._chunk_by_id = {}

    def warmup(self) -> dict[str, Any]:
        """Build the BM25 index before the first request.

        Returns:
            dict[str, Any]: Corpus and configuration metadata.
        """
        self._ensure_corpus()
        return self.stats()

    # -- retrieval --------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve and fuse chunks.

        Args:
            query: User query.
            top_k: Number of chunks to return.
            mode: Retrieval mode override.
            rerank: Rerank override.

        Returns:
            list[dict[str, Any]]: Fused (and optionally reranked) results.

        Raises:
            ValueError: If the mode is unknown.
        """
        selected_mode = (mode or self._default_mode).lower()
        if selected_mode not in MODES:
            raise ValueError(f"unknown retrieval mode: {selected_mode!r}")
        should_rerank = self._default_rerank if rerank is None else rerank
        top_k = max(1, top_k)
        query_text = (query or "").strip()
        if not query_text:
            return []

        self._ensure_corpus()
        if not self._chunks:
            return []

        # Both legs must retrieve at least the candidate pool, otherwise a
        # small top_k would starve the fusion (RRF over one vector hit is not
        # the same ranking as RRF over twenty).
        leg_k = max(top_k, self._candidate_pool) if selected_mode == "hybrid" else top_k
        pool = self._candidate_pool if (should_rerank and selected_mode == "hybrid") else top_k

        bm25_ranked: list[tuple[float, ChunkRecord]] = []
        if selected_mode != "vector":
            bm25_ranked = bm25_rank(query_text, self._chunks, self._index)

        vector_hits: list[tuple[float, str]] = []
        vector_error: Optional[str] = None
        if selected_mode != "bm25":
            try:
                hits = self._store.vector_search(query_text, leg_k)
                vector_hits = [(hit.score, hit.chunk_id) for hit in hits]
            except Exception as exc:  # noqa: BLE001 - degrade to BM25
                vector_error = str(exc)
                self._logger.warning("Dense retrieval unavailable, using BM25 only: %s", exc)

        bm25_ranks = {
            chunk.chunk_id: rank for rank, (_score, chunk) in enumerate(bm25_ranked, start=1)
        }
        vector_ranks = {
            chunk_id: rank for rank, (_score, chunk_id) in enumerate(vector_hits, start=1)
        }
        bm25_score = {chunk.chunk_id: score for score, chunk in bm25_ranked}
        vector_score = {chunk_id: score for score, chunk_id in vector_hits}

        if selected_mode == "bm25":
            ordered = [(score, chunk) for score, chunk in bm25_ranked[:pool]]
        elif selected_mode == "vector":
            ordered = [
                (score, self._chunk_by_id[chunk_id])
                for score, chunk_id in vector_hits
                if chunk_id in self._chunk_by_id
            ]
        else:
            fused: dict[str, float] = {}
            for chunk_id, rank in bm25_ranks.items():
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
            for chunk_id, rank in vector_ranks.items():
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
            ordered = [
                (fused[chunk_id], self._chunk_by_id[chunk_id])
                for chunk_id in sorted(fused)
                if chunk_id in self._chunk_by_id
            ]
            ordered.sort(key=lambda item: (-item[0], item[1].chunk_id))
            ordered = ordered[:pool]

        if not ordered:
            return []

        results = self._to_results(
            ordered,
            mode=selected_mode,
            bm25_rank=bm25_ranks,
            vector_rank=vector_ranks,
            bm25_score=bm25_score,
            vector_score=vector_score,
            vector_error=vector_error,
        )

        if should_rerank and selected_mode == "hybrid" and len(results) > 1:
            results = self._reranker.rerank(query_text, results, top_k=len(results))
            best = max(1e-6, float(results[0].get("rerank_score") or 0.0))
            for item in results:
                raw = max(0.0, float(item.get("rerank_score") or 0.0))
                item["score"] = round(min(0.99, raw / best * 0.99), 4)

        return results[:top_k]

    def _to_results(
        self,
        ordered: Sequence[tuple[float, ChunkRecord]],
        *,
        mode: str,
        bm25_rank: dict[str, int],
        vector_rank: dict[str, int],
        bm25_score: dict[str, float],
        vector_score: dict[str, float],
        vector_error: Optional[str],
    ) -> list[dict[str, Any]]:
        """Convert ranked chunks into the public result shape.

        Args:
            ordered: (raw score, chunk) pairs.
            mode: Retrieval mode used.
            bm25_rank: BM25 rank per chunk id.
            vector_rank: Dense rank per chunk id.
            bm25_score: BM25 score per chunk id.
            vector_score: Dense score per chunk id.
            vector_error: Dense-leg failure message, if any.

        Returns:
            list[dict[str, Any]]: Result dictionaries.
        """
        top_score = ordered[0][0] or 1.0
        results: list[dict[str, Any]] = []
        for raw_score, chunk in ordered:
            path = str(chunk.metadata.get("path") or "")
            results.append(
                {
                    "id": chunk.chunk_id,
                    "content": chunk.content,
                    "path": path,
                    "source": str(chunk.metadata.get("source") or ""),
                    "metadata": dict(chunk.metadata),
                    "score": round(min(0.99, raw_score / top_score * 0.99), 4),
                    "rrf_score": round(raw_score, 6),
                    "bm25_rank": bm25_rank.get(chunk.chunk_id),
                    "vector_rank": vector_rank.get(chunk.chunk_id),
                    "bm25_score": round(bm25_score.get(chunk.chunk_id, 0.0), 4),
                    "vector_score": round(vector_score.get(chunk.chunk_id, 0.0), 4),
                    "retrieval_mode": mode,
                    "vector_error": vector_error,
                }
            )
        return results

    # -- observability ----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Return corpus and configuration metadata.

        Returns:
            dict[str, Any]: Index statistics.
        """
        self._ensure_corpus()
        return {
            **self._store.stats(),
            "sources": len({chunk.metadata.get("source") for chunk in self._chunks}),
            "rrf_k": self._rrf_k,
            "candidate_pool": self._candidate_pool,
            "default_mode": self._default_mode,
            "default_rerank": self._default_rerank,
            "reranker": self._reranker.name,
        }

    def retrieve_with_trace(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """Run the four ablation configurations for one query.

        Args:
            query: User query.
            top_k: Number of results per configuration.

        Returns:
            dict[str, Any]: Results and latency per configuration.
        """
        configurations = (
            ("hybrid_rerank", "hybrid", True),
            ("hybrid", "hybrid", False),
            ("bm25", "bm25", False),
            ("vector", "vector", False),
        )
        results: dict[str, Any] = {}
        for name, mode, rerank in configurations:
            started = time.perf_counter()
            items = self.retrieve(query, top_k=top_k, mode=mode, rerank=rerank)
            results[name] = {
                "mode": mode,
                "rerank": rerank,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "results": items,
            }
        return {"query": query, "corpus": self.stats(), "runs": results}
