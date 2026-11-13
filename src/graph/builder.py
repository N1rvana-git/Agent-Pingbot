"""Build and compile the CRAG LangGraph workflow."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import CRAGNodes
from .state import AgentState

_CORRECT = "to_refine"
_INCORRECT = "to_search"
_AMBIGUOUS = "to_both"


def build_crag_graph(nodes: CRAGNodes | None = None) -> object:
    """Build and compile the CRAG graph.

    Args:
        nodes: Optional node collection (tests inject a preconfigured instance).

    Returns:
        object: Compiled graph application.
    """
    graph_nodes = nodes or CRAGNodes()
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", graph_nodes.retrieve)
    workflow.add_node("evaluate", graph_nodes.evaluate)
    workflow.add_node("knowledge_refinement", graph_nodes.refine_knowledge)
    workflow.add_node("web_search", graph_nodes.web_search)
    workflow.add_node("generate", graph_nodes.generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")

    def route_after_evaluate(state: AgentState) -> str:
        """Map the CRAG confidence to a correction path.

        Args:
            state: Current state.

        Returns:
            str: Route key.
        """
        confidence = state.get("confidence", "ambiguous")
        if confidence == "correct":
            return _CORRECT
        if confidence == "incorrect":
            return _INCORRECT
        return _AMBIGUOUS

    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            _CORRECT: "knowledge_refinement",
            _INCORRECT: "web_search",
            _AMBIGUOUS: "knowledge_refinement",
        },
    )

    def route_after_refine(state: AgentState) -> str:
        """Continue to web search for ambiguous queries, otherwise generate.

        Args:
            state: Current state.

        Returns:
            str: Route key.
        """
        return "web_search" if state.get("confidence") == "ambiguous" else "generate"

    workflow.add_conditional_edges(
        "knowledge_refinement",
        route_after_refine,
        {"web_search": "web_search", "generate": "generate"},
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
