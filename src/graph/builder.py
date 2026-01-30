"""Build and compile the CRAG LangGraph workflow."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import CRAGNodes
from .state import AgentState


def build_crag_graph() -> object:
    """Build and compile the CRAG graph.

    Returns:
        object: Compiled graph application.
    """
    nodes = CRAGNodes()
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("evaluate", nodes.evaluate)
    workflow.add_node("knowledge_refinement", nodes.refine_knowledge)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("generate", nodes.generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")

    def router(state: AgentState) -> str:
        """Route after evaluation.

        Args:
            state: Current agent state.

        Returns:
            str: Route key.
        """
        confidence = state["confidence"]
        if confidence == "correct":
            return "to_refine"
        if confidence == "incorrect":
            return "to_search"
        return "to_both"

    workflow.add_conditional_edges(
        "evaluate",
        router,
        {
            "to_refine": "knowledge_refinement",
            "to_search": "web_search",
            "to_both": "knowledge_refinement",
        },
    )

    def route_after_refine(state: AgentState) -> str:
        """Route after refinement.

        Args:
            state: Current agent state.

        Returns:
            str: Route key.
        """
        return "web_search" if state["confidence"] == "ambiguous" else "generate"

    workflow.add_conditional_edges(
        "knowledge_refinement",
        route_after_refine,
        {"web_search": "web_search", "generate": "generate"},
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
