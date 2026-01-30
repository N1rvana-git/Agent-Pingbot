"""Build the LangGraph workflow (skeleton)."""
from __future__ import annotations

from typing import List

from ..components.evaluator import RetrievalEvaluator, determine_crag_action
from ..components.refiner import KnowledgeRefiner
from ..components.search import WebSearcher
from .state import AgentState


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve documents for the query.

    Args:
        state: Current agent state.

    Returns:
        AgentState: Updated state with documents.
    """
    state["documents"] = []
    return state


def evaluate_node(state: AgentState, evaluator: RetrievalEvaluator) -> AgentState:
    """Evaluate retrieved documents.

    Args:
        state: Current agent state.
        evaluator: Retrieval evaluator.

    Returns:
        AgentState: Updated state with scores and action.
    """
    results = evaluator.score_documents(state["query"], state["documents"])
    scores = [r.score for r in results]
    state["scores"] = scores
    state["action"] = determine_crag_action(scores)
    return state


def refine_node(state: AgentState, refiner: KnowledgeRefiner) -> AgentState:
    """Refine knowledge if action is correct or ambiguous.

    Args:
        state: Current agent state.
        refiner: Knowledge refiner.

    Returns:
        AgentState: Updated state with refined context.
    """
    if state["documents"]:
        state["refined_context"] = refiner.refine(state["query"], "\n".join(state["documents"]))
    return state


def search_node(state: AgentState, searcher: WebSearcher) -> AgentState:
    """Perform web search if needed.

    Args:
        state: Current agent state.
        searcher: Web searcher.

    Returns:
        AgentState: Updated state with search results.
    """
    state["search_results"] = searcher.search(state["query"], top_k=5)
    return state
