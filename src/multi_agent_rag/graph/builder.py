"""Build and compile the LangGraph multi-agent workflow."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from multi_agent_rag.graph.edges import route_after_supervisor, route_after_validation
from multi_agent_rag.graph.nodes import (
    reasoning_node,
    response_node,
    retrieval_node,
    supervisor_node,
    validation_node,
)
from multi_agent_rag.graph.state import GraphState


def build_graph(use_checkpointer: bool = True) -> StateGraph:
    """
    Assemble and compile the five-node LangGraph workflow.

    Graph topology
    ──────────────
    START → supervisor
               ├─(retrieval)─→ retrieval → reasoning → validation
               └─(reasoning)─────────────→ reasoning → validation
                                                           ├─(retry)─→ reasoning  (loop ≤ max_retries)
                                                           └─(pass) ─→ response → END

    With use_checkpointer=True an InMemorySaver is attached so that the graph
    supports multi-turn conversation via LangGraph thread IDs.
    LangSmith tracing is activated automatically when LANGCHAIN_TRACING_V2=true
    is present in the environment.
    """
    graph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("validation", validation_node)
    graph.add_node("response", response_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Supervisor → retrieval or reasoning ───────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"retrieval": "retrieval", "reasoning": "reasoning"},
    )

    # ── Retrieval always leads to reasoning ───────────────────────────────────
    graph.add_edge("retrieval", "reasoning")

    # ── Reasoning always leads to validation ─────────────────────────────────
    graph.add_edge("reasoning", "validation")

    # ── Validation → retry reasoning OR finalize ─────────────────────────────
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {"reasoning": "reasoning", "response": "response"},
    )

    # ── Response is the terminal node ─────────────────────────────────────────
    graph.add_edge("response", END)

    checkpointer = MemorySaver() if use_checkpointer else None
    return graph.compile(checkpointer=checkpointer)


# Module-level compiled graph — import this for one-shot usage
compiled_graph = build_graph(use_checkpointer=True)
