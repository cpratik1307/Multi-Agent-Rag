"""Tests for LangGraph edge routing logic."""

from __future__ import annotations


# ── route_after_supervisor ────────────────────────────────────────────────────

def test_route_supervisor_returns_retrieval():
    from multi_agent_rag.graph.edges import route_after_supervisor

    state = {"next_agent": "retrieval", "query": "test", "retrieved_docs": []}
    assert route_after_supervisor(state) == "retrieval"


def test_route_supervisor_returns_reasoning():
    from multi_agent_rag.graph.edges import route_after_supervisor

    state = {"next_agent": "reasoning", "query": "test", "retrieved_docs": []}
    assert route_after_supervisor(state) == "reasoning"


def test_route_supervisor_defaults_to_retrieval_when_missing():
    from multi_agent_rag.graph.edges import route_after_supervisor

    state = {"query": "test"}  # next_agent not set
    assert route_after_supervisor(state) == "retrieval"


# ── route_after_validation ────────────────────────────────────────────────────

def test_route_validation_retries_on_low_score():
    from multi_agent_rag.graph.edges import route_after_validation

    state = {"validation_score": 0.4, "retry_count": 1}
    assert route_after_validation(state) == "reasoning"


def test_route_validation_proceeds_on_high_score():
    from multi_agent_rag.graph.edges import route_after_validation

    state = {"validation_score": 0.85, "retry_count": 1}
    assert route_after_validation(state) == "response"


def test_route_validation_proceeds_on_exact_threshold():
    from multi_agent_rag.graph.edges import route_after_validation

    state = {"validation_score": 0.7, "retry_count": 1}
    assert route_after_validation(state) == "response"


def test_route_validation_stops_at_max_retries():
    from multi_agent_rag.graph.edges import route_after_validation

    # retry_count == max_retries (3) → should NOT retry, even if score is low
    state = {"validation_score": 0.1, "retry_count": 3}
    assert route_after_validation(state) == "response"


def test_route_validation_still_retries_below_max():
    from multi_agent_rag.graph.edges import route_after_validation

    state = {"validation_score": 0.2, "retry_count": 2}
    assert route_after_validation(state) == "reasoning"
