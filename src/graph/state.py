"""LangGraph state for the CRAG workflow.

Every node returns a partial update, so the state is declared with
``total=False``: only ``question`` is guaranteed on entry, and downstream nodes
read optional keys defensively.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """State shared by the CRAG nodes.

    Args:
        question: User question.
        retrieved_documents: Retrieved chunks (content + provenance + scores).
        retrieval_trace: Compact retrieval diagnostics for the UI/API.
        evaluation_scores: Relevance score per chunk id.
        confidence: CRAG decision (correct/incorrect/ambiguous).
        knowledge_strips: Refined, citation-prefixed context strips.
        search_results: Web search snippets.
        final_context: Final blended context.
        final_answer: Final answer.
    """

    question: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_trace: Dict[str, Any]
    evaluation_scores: Dict[str, float]
    confidence: str
    knowledge_strips: List[str]
    search_results: List[str]
    final_context: str
    final_answer: str
