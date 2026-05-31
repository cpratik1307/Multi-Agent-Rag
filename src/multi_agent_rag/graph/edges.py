"""Conditional edge functions that control LangGraph routing."""

from __future__ import annotations

from typing import Literal

from multi_agent_rag.config import settings
from multi_agent_rag.graph.state import GraphState


def route_after_supervisor(
    state: GraphState,
) -> Literal["retrieval", "reasoning"]:
    """Route to retrieval or reasoning based on the supervisor's decision."""
    return state.get("next_agent", "retrieval")  # type: ignore[return-value]


def route_after_validation(
    state: GraphState,
) -> Literal["reasoning", "response"]:
    """
    Retry reasoning if the grounding score is below threshold AND we haven't
    exhausted the maximum allowed retries.  Otherwise, finalize the response.
    """
    score = state.get("validation_score", 0.0)
    retry_count = state.get("retry_count", 0)

    if score < settings.grounding_threshold and retry_count < settings.max_retries:
        return "reasoning"

    return "response"
