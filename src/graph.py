"""
Wires the nodes into a LangGraph StateGraph:

    planner -> search -> validate_extract -> synthesizer -> END
"""

from langgraph.graph import StateGraph, END

from src.state import ResearchState
from src.nodes.planner import planner_node
from src.nodes.search import search_node
from src.nodes.validate_extract import validate_extract_node
from src.nodes.synthesizer import synthesizer_node


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("validate_extract", validate_extract_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "validate_extract")
    graph.add_edge("validate_extract", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()