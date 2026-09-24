"""Assemble the LangGraph state graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", nodes.route_intent)
    graph.add_node("qa", nodes.qa_node)
    graph.add_node("create", nodes.create_node)
    graph.add_node("history", nodes.history_node)
    graph.add_node("chat", nodes.chat_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        nodes.intent_edge,
        {"qa": "qa", "create": "create", "history": "history", "chat": "chat"},
    )
    for node in ("qa", "create", "history", "chat"):
        graph.add_edge(node, END)
    return graph.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
