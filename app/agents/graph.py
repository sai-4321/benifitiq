"""Wires the agents into a LangGraph state machine.

            START
              |
           router ---------------------------------------------+
     policy/  | member         \\ both                           | out_of_scope
           v  v                 v                               v
     retriever  lookup <---- retriever (then lookup)          refuse -> END
           \\     |
            v    v
          summarizer -> human_review (may PAUSE) -> END

The checkpointer saves state after every node. That is what lets the graph stop at
human_review, return to the API, and later resume the SAME run by thread_id.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    human_review_node,
    lookup_node,
    refuse_node,
    retriever_node,
    router_node,
    summarizer_node,
)
from app.agents.state import AgentState


def route_after_router(state: AgentState) -> str:
    return {"policy": "retriever", "both": "retriever", "member": "lookup"}.get(state.get("intent", ""), "refuse")


def route_after_retriever(state: AgentState) -> str:
    return "lookup" if state.get("intent") == "both" else "summarizer"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("lookup", lookup_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("refuse", refuse_node)
    g.add_node("human_review", human_review_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router,
                            {"retriever": "retriever", "lookup": "lookup", "refuse": "refuse"})
    g.add_conditional_edges("retriever", route_after_retriever, {"lookup": "lookup", "summarizer": "summarizer"})
    g.add_edge("lookup", "summarizer")
    g.add_edge("summarizer", "human_review")
    g.add_edge("human_review", END)
    g.add_edge("refuse", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


@lru_cache(maxsize=1)
def get_graph():
    """One graph + one in-memory checkpointer per process (so /approve can find paused runs)."""
    return build_graph()
