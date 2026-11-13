"""CRAG graph nodes.

Pipeline: retrieve → evaluate → (refine | web search | both) → generate.

Retrieval is hybrid (BM25 + dense vectors + RRF) and may rerank the fused
candidate pool; see :mod:`src.rag.retriever`. The evaluation step implements the
CRAG paper's Correct / Incorrect / Ambiguous decision, and the correction step
either refines the retrieved knowledge or falls back to web search.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..components.evaluator import RetrievalEvaluator, determine_crag_action
from ..components.generator import AnswerGenerator
from ..components.refiner import KnowledgeRefiner
from ..components.rewriter import QueryRewriter
from ..components.search import WebSearcher
from ..config import Settings, load_settings
from ..rag.retriever import HybridRetriever
from ..rag.store import get_store
from ..utils.logging_utils import setup_logging
from .state import AgentState

_MAX_CONTEXT_CHARS = 4000
_OFFLINE_MAX_BLOCKS = 2
_OFFLINE_MAX_CHARS = 900
_MAX_REFINE_DOCS = 3


class CRAGNodes:
    """Collection of graph nodes for the CRAG workflow.

    Args:
        settings: Optional settings override (tests).
        logger: Optional logger.
    """

    def __init__(self, settings: Settings | None = None, logger: logging.Logger | None = None) -> None:
        self._logger = logger or setup_logging(name="rail-crag")
        self._settings = settings or load_settings(require_keys=False)
        self._retriever = HybridRetriever(
            get_store(self._settings),
            rrf_k=self._settings.rrf_k,
            candidate_pool=self._settings.candidate_pool,
            default_mode=self._settings.retrieval_mode,
            default_rerank=self._settings.rerank,
            logger=self._logger,
        )
        self._evaluator = RetrievalEvaluator(self._settings, logger=self._logger)
        self._refiner = KnowledgeRefiner(self._evaluator)
        self._searcher = WebSearcher(self._settings.tavily_api_key, logger=self._logger)
        self._rewriter = QueryRewriter(self._settings, logger=self._logger)
        self._generator = AnswerGenerator(self._settings, logger=self._logger)

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve chunks with hybrid search.

        Args:
            state: Current state (``question`` required).

        Returns:
            Dict[str, Any]: ``retrieved_documents`` and ``retrieval_trace``.
        """
        question = state["question"]
        self._logger.info("[Node] retrieve mode=%s", self._settings.retrieval_mode)
        documents = self._retriever.retrieve(
            question,
            top_k=self._settings.retriever_k,
            mode=self._settings.retrieval_mode,
            rerank=self._settings.rerank,
        )
        trace = {
            "mode": self._settings.retrieval_mode,
            "rerank": self._settings.rerank,
            "top_k": self._settings.retriever_k,
            "returned": len(documents),
            "vector_error": next(
                (doc.get("vector_error") for doc in documents if doc.get("vector_error")), None
            ),
            "hits": [
                {
                    "id": doc.get("id"),
                    "path": doc.get("path"),
                    "source": doc.get("source"),
                    "score": doc.get("score"),
                    "bm25_rank": doc.get("bm25_rank"),
                    "vector_rank": doc.get("vector_rank"),
                }
                for doc in documents
            ],
        }
        return {"retrieved_documents": documents, "retrieval_trace": trace}

    def evaluate(self, state: AgentState) -> Dict[str, Any]:
        """Score retrieved chunks and decide the CRAG action.

        Args:
            state: Current state.

        Returns:
            Dict[str, Any]: ``evaluation_scores`` and ``confidence``.
        """
        self._logger.info("[Node] evaluate")
        documents = state.get("retrieved_documents", [])
        results = self._evaluator.score_documents(
            state["question"], [doc.get("content", "") for doc in documents]
        )
        scores = {
            str(doc.get("id") or index): result.score
            for index, (doc, result) in enumerate(zip(documents, results))
        }
        confidence = determine_crag_action(
            list(scores.values()),
            upper=self._settings.upper_threshold,
            lower=self._settings.lower_threshold,
        )
        return {"evaluation_scores": scores, "confidence": confidence}

    def refine_knowledge(self, state: AgentState) -> Dict[str, Any]:
        """Refine retrieved chunks into citation-prefixed strips.

        Args:
            state: Current state.

        Returns:
            Dict[str, Any]: ``knowledge_strips``.
        """
        self._logger.info("[Node] refine_knowledge")
        documents = state.get("retrieved_documents", [])
        scores = state.get("evaluation_scores", {})
        # Refining every chunk multiplies the evaluator calls (one per strip).
        # Keep the best chunks by the score the evaluate node already computed.
        ranked = sorted(
            documents,
            key=lambda doc: scores.get(str(doc.get("id")), 0.0),
            reverse=True,
        )[:_MAX_REFINE_DOCS]

        refined: List[str] = []
        for doc in ranked:
            content = str(doc.get("content") or "")
            if not content:
                continue
            refined_text = self._refiner.refine(state["question"], content)
            if not refined_text:
                continue
            path = str(doc.get("path") or "")
            refined.append(f"【{path}】\n{refined_text}" if path else refined_text)
        return {"knowledge_strips": refined}

    def web_search(self, state: AgentState) -> Dict[str, Any]:
        """Search the web when internal knowledge is insufficient.

        Args:
            state: Current state.

        Returns:
            Dict[str, Any]: ``search_results``.
        """
        self._logger.info("[Node] web_search")
        rewritten = self._rewriter.rewrite(state["question"])
        results = self._searcher.search(rewritten, top_k=self._settings.search_k)
        return {"search_results": results}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final answer from refined and searched context.

        Args:
            state: Current state.

        Returns:
            Dict[str, Any]: ``final_context`` and ``final_answer``.
        """
        self._logger.info("[Node] generate")
        context_parts = list(state.get("knowledge_strips", [])) + list(state.get("search_results", []))
        # With a model the context is a prompt budget; without one it is the
        # answer itself, so keep the extract short enough to read.
        char_budget = _MAX_CONTEXT_CHARS if self._settings.has_chat_model else _OFFLINE_MAX_CHARS
        block_limit = len(context_parts) if self._settings.has_chat_model else _OFFLINE_MAX_BLOCKS
        selected: List[str] = []
        total = 0
        for part in context_parts:
            if not part:
                continue
            if len(selected) >= block_limit:
                break
            if selected and total + len(part) > char_budget:
                break
            selected.append(part)
            total += len(part)
        context = "\n\n".join(selected)
        answer = self._generator.generate(state["question"], context)
        return {"final_context": context, "final_answer": answer}
