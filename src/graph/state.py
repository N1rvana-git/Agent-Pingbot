"""LangGraph state definitions."""
from __future__ import annotations

from typing import List, TypedDict


class AgentState(TypedDict):
    """State for CRAG graph.

    Args:
        question: User query.
        retrieved_documents: Retrieved documents.
        evaluation_scores: Relevance scores per document id.
        confidence: CRAG decision (correct/incorrect/ambiguous).
        knowledge_strips: Refined strips.
        search_results: Web search results.
        final_context: Final blended context.
        final_answer: Final LLM response.
    """

    question: str
    retrieved_documents: List[dict]
    evaluation_scores: dict
    confidence: str
    knowledge_strips: List[str]
    search_results: List[str]
    final_context: str
    final_answer: str
