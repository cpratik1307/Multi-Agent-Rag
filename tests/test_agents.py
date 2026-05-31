"""Tests for individual agent functions (no LLM calls required for most)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── supervisor_decision ───────────────────────────────────────────────────────

def test_supervisor_no_docs_skips_llm():
    """When no docs are available, supervisor routes to retrieval without an LLM call."""
    from multi_agent_rag.agents.supervisor import supervisor_decision

    result = supervisor_decision("What was the revenue?", has_retrieved_docs=False)
    assert result["next_agent"] == "retrieval"


def test_supervisor_with_docs_calls_llm_and_parses_json():
    """When docs exist, supervisor calls GPT-4o and parses its JSON response."""
    from multi_agent_rag.agents.supervisor import supervisor_decision

    mock_response = MagicMock()
    mock_response.content = '{"next_agent": "reasoning", "reasoning": "Context available."}'

    with patch("multi_agent_rag.agents.supervisor.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_cls.return_value = mock_llm

        result = supervisor_decision("What was the revenue?", has_retrieved_docs=True)

    assert result["next_agent"] in ("retrieval", "reasoning")


def test_supervisor_invalid_json_falls_back_to_retrieval():
    """If the LLM returns invalid JSON, supervisor defaults to retrieval safely."""
    from multi_agent_rag.agents.supervisor import supervisor_decision

    mock_response = MagicMock()
    mock_response.content = "not valid json at all"

    with patch("multi_agent_rag.agents.supervisor.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_cls.return_value = mock_llm

        result = supervisor_decision("What was the revenue?", has_retrieved_docs=True)

    assert result["next_agent"] == "retrieval"


# ── run_validation ────────────────────────────────────────────────────────────

def test_validation_high_score_sets_validated_answer():
    from multi_agent_rag.agents.validation_agent import run_validation

    docs = [Document(page_content="Revenue was 100 billion dollars in Q4 fiscal year")]
    result = run_validation("Revenue was 100 billion in Q4", docs)

    assert result["validation_score"] >= 0.7
    assert result["validated_answer"] != ""


def test_validation_low_score_clears_validated_answer():
    from multi_agent_rag.agents.validation_agent import run_validation

    docs = [Document(page_content="Revenue was 100 billion dollars in Q4 fiscal year")]
    result = run_validation("The CEO resigned and launched a new product line", docs)

    assert result["validation_score"] < 0.7
    assert result["validated_answer"] == ""


# ── run_reasoning (mocked LLM) ────────────────────────────────────────────────

def test_reasoning_uses_fallback_on_primary_failure():
    from multi_agent_rag.agents.reasoning_agent import run_reasoning

    docs = [Document(page_content="Apple Q4 revenue was $119.6 billion.")]
    mock_response = MagicMock()
    mock_response.content = "Apple Q4 revenue was $119.6 billion."

    call_count = 0

    def side_effect(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Primary model error")
        return mock_response

    with patch("multi_agent_rag.agents.reasoning_agent.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = side_effect
        mock_cls.return_value = mock_llm

        result = run_reasoning("What was Apple Q4 revenue?", docs)

    assert result["error"] is None
    assert "119.6" in result["reasoning_output"]


def test_reasoning_returns_error_when_both_models_fail():
    from multi_agent_rag.agents.reasoning_agent import run_reasoning

    docs = [Document(page_content="Some financial data.")]

    with patch("multi_agent_rag.agents.reasoning_agent.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Model unavailable")
        mock_cls.return_value = mock_llm

        result = run_reasoning("What was the revenue?", docs)

    assert result["error"] is not None
    assert result["reasoning_output"] == ""
