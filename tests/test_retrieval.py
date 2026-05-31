"""Tests for the grounding tool and its scoring logic."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


# ── _compute_grounding_score unit tests ──────────────────────────────────────

def test_grounding_score_high_overlap():
    from multi_agent_rag.tools.grounding_tool import _compute_grounding_score

    answer = "Apple reported revenue of 100 billion dollars in Q4 2023"
    context = ["Apple reported quarterly revenue of 100 billion dollars in fiscal Q4 2023"]
    score = _compute_grounding_score(answer, context)
    assert score >= 0.7


def test_grounding_score_low_overlap():
    from multi_agent_rag.tools.grounding_tool import _compute_grounding_score

    answer = "Tesla launched a new rocket to Mars in 2023"
    context = ["Apple reported quarterly revenue of 100 billion dollars in fiscal Q4 2023"]
    score = _compute_grounding_score(answer, context)
    assert score < 0.7


def test_grounding_score_empty_answer():
    from multi_agent_rag.tools.grounding_tool import _compute_grounding_score

    score = _compute_grounding_score("", ["Some context text here"])
    assert score == 0.0


def test_grounding_score_empty_context():
    from multi_agent_rag.tools.grounding_tool import _compute_grounding_score

    score = _compute_grounding_score("Some answer here", [])
    assert score == 0.0


# ── grounding_check tool tests ────────────────────────────────────────────────

def test_grounding_check_tool_returns_expected_keys():
    from multi_agent_rag.tools.grounding_tool import grounding_check

    result = grounding_check.invoke(
        {
            "answer": "Revenue was 100 billion dollars",
            "context_chunks": ["The company had revenue of 100 billion dollars"],
        }
    )
    assert "score" in result
    assert "is_grounded" in result
    assert "explanation" in result
    assert 0.0 <= result["score"] <= 1.0


def test_grounding_check_grounded_flag():
    from multi_agent_rag.tools.grounding_tool import grounding_check

    high_overlap = grounding_check.invoke(
        {
            "answer": "revenue profit earnings quarter fiscal year",
            "context_chunks": ["revenue profit earnings quarter fiscal year results"],
        }
    )
    assert high_overlap["is_grounded"] is True

    low_overlap = grounding_check.invoke(
        {
            "answer": "aliens visited earth yesterday",
            "context_chunks": ["Apple quarterly revenue was 100 billion dollars"],
        }
    )
    assert low_overlap["is_grounded"] is False
