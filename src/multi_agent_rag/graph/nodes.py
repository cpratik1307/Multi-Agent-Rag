"""LangGraph node functions — one function per graph node."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from multi_agent_rag.agents.reasoning_agent import run_reasoning
from multi_agent_rag.agents.retrieval_agent import run_retrieval
from multi_agent_rag.agents.supervisor import supervisor_decision
from multi_agent_rag.agents.validation_agent import run_validation
from multi_agent_rag.graph.state import GraphState
from multi_agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


# ── Node 1: Supervisor ────────────────────────────────────────────────────────

def supervisor_node(state: GraphState) -> dict:
    """
    Decide whether the graph should retrieve new documents or go straight to
    reasoning using already-retrieved context.
    """
    has_docs = bool(state.get("retrieved_docs"))
    decision = supervisor_decision(state["query"], has_docs)
    logger.info("supervisor_node", next_agent=decision["next_agent"])
    return {"next_agent": decision["next_agent"]}


# ── Node 2: Retrieval ─────────────────────────────────────────────────────────

def retrieval_node(state: GraphState) -> dict:
    """Retrieve relevant document chunks via hybrid FAISS + BM25 search."""
    return run_retrieval(state["query"])


# ── Node 3: Reasoning ─────────────────────────────────────────────────────────

def reasoning_node(state: GraphState) -> dict:
    """Generate an answer using GPT-4o over the retrieved context."""
    return run_reasoning(
        query=state["query"],
        retrieved_docs=state.get("retrieved_docs", []),
    )


# ── Node 4: Validation ────────────────────────────────────────────────────────

def validation_node(state: GraphState) -> dict:
    """
    Validate the reasoning output against retrieved context.
    Increments retry_count — used by the edge function to cap retry loops.
    """
    result = run_validation(
        reasoning_output=state.get("reasoning_output", ""),
        retrieved_docs=state.get("retrieved_docs", []),
    )
    return {
        **result,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ── Node 5: Response ──────────────────────────────────────────────────────────

def response_node(state: GraphState) -> dict:
    """
    Format the final answer, append it to chat history, and return it.

    Falls back to the raw reasoning_output if validation never produced a
    grounded answer (e.g., after max retries were exhausted).
    """
    answer = state.get("validated_answer") or state.get("reasoning_output", "")
    score = state.get("validation_score", 0.0)
    sources = state.get("sources", [])
    error = state.get("error")

    if error:
        final_answer = f"⚠️ An error occurred: {error}"
    else:
        source_lines = "\n".join(f"  • {s}" for s in sources) or "  • No sources available"
        final_answer = (
            f"{answer}\n\n"
            f"**Sources:**\n{source_lines}\n\n"
            f"**Grounding confidence:** {score:.0%}"
        )

    logger.info(
        "response_node",
        confidence=f"{score:.0%}",
        sources=len(sources),
        retry_count=state.get("retry_count", 0),
    )

    return {
        "validated_answer": final_answer,
        # Append the AI reply to accumulated conversation history
        "chat_history": [AIMessage(content=final_answer)],
    }
