"""CRAG graph node implementations (skeleton)."""
from __future__ import annotations

from typing import Dict, List

from ..components.evaluator import RetrievalEvaluator, determine_crag_action
from ..components.generator import AnswerGenerator
from ..components.refiner import KnowledgeRefiner
from ..components.rewriter import QueryRewriter
from ..components.search import WebSearcher
from ..components.vector_store import VectorStore
from ..config import load_settings
from ..utils.logging_utils import setup_logging
from .state import AgentState


class CRAGNodes:
    """Collection of graph nodes for CRAG workflow.

    Each node is a pure-ish function that takes state and returns a partial update.
    """

    def __init__(self) -> None:
        """Initialize CRAG nodes with logger and settings."""
        self._logger = setup_logging(name="rail-crag")
        self._settings = load_settings(require_keys=False)
        self._vector_store = VectorStore(self._settings, logger=self._logger)
        self._evaluator = RetrievalEvaluator(self._settings, logger=self._logger)
        self._refiner = KnowledgeRefiner(self._evaluator)
        self._searcher = WebSearcher(self._settings.tavily_api_key, logger=self._logger)
        self._rewriter = QueryRewriter(self._settings, logger=self._logger)
        self._generator = AnswerGenerator(self._settings, logger=self._logger)

    def retrieve(self, state: AgentState) -> Dict[str, object]:
        """Retrieve documents for the query.

        Args:
            state: Current agent state.

        Returns:
            Dict[str, object]: Partial state updates.
        """
        self._logger.info("[Node] retrieve")
        retrieved = self._vector_store.search(state["question"], self._settings.retriever_k)
        return {
            "retrieved_documents": [
                {"id": doc.doc_id, "content": doc.content, "metadata": doc.metadata} for doc in retrieved
            ]
        }

    def evaluate(self, state: AgentState) -> Dict[str, object]:
        """Evaluate retrieved documents and decide confidence.

        Args:
            state: Current agent state.

        Returns:
            Dict[str, object]: Partial state updates.
        """
        self._logger.info("[Node] evaluate")
        documents = [doc.get("content", "") for doc in state.get("retrieved_documents", [])]
        results = self._evaluator.score_documents(state["question"], documents)
        scores = {str(i): r.score for i, r in enumerate(results)}
        confidence = determine_crag_action(
            list(scores.values()),
            upper=self._settings.upper_threshold,
            lower=self._settings.lower_threshold,
        )
        return {"evaluation_scores": scores, "confidence": confidence}

    def refine_knowledge(self, state: AgentState) -> Dict[str, object]:
        """Refine retrieved knowledge (decompose-filter-recompose).

        Args:
            state: Current agent state.

        Returns:
            Dict[str, object]: Partial state updates.
        """
        self._logger.info("[Node] refine_knowledge")
        refined: List[str] = []
        for doc in state.get("retrieved_documents", []):
            content = doc.get("content", "")
            if not content:
                continue
            refined_text = self._refiner.refine(state["question"], content)
            if refined_text:
                refined.append(refined_text)
        return {"knowledge_strips": refined}

    def web_search(self, state: AgentState) -> Dict[str, object]:
        """Perform web search to supplement knowledge.

        Args:
            state: Current agent state.

        Returns:
            Dict[str, object]: Partial state updates.
        """
        self._logger.info("[Node] web_search")
        rewritten = self._rewriter.rewrite(state["question"])
        results = self._searcher.search(rewritten, top_k=self._settings.search_k)
        return {"search_results": results}

    def generate(self, state: AgentState) -> Dict[str, object]:
        """Generate final response based on context.

        Args:
            state: Current agent state.

        Returns:
            Dict[str, object]: Partial state updates.
        """
        self._logger.info("[Node] generate")
        context = "\n".join(state.get("knowledge_strips", []) + state.get("search_results", []))
        answer = self._generator.generate(state["question"], context)
        return {"final_context": context, "final_answer": answer}
