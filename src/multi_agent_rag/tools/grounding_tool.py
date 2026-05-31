"""Grounding / hallucination-check tool.

Computes token-level precision between the generated answer and the retrieved
context chunks.  A score ≥ 0.7 indicates the answer is well-grounded.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from langchain.tools import tool


class GroundingInput(BaseModel):
    answer: str = Field(description="The generated answer to validate.")
    context_chunks: list[str] = Field(
        description="List of retrieved document text chunks used to generate the answer."
    )


def _compute_grounding_score(answer: str, context_chunks: list[str]) -> float:
    """Token-overlap precision: fraction of answer tokens found in context."""

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0

    combined_context = " ".join(context_chunks)
    context_tokens = tokenize(combined_context)
    if not context_tokens:
        return 0.0

    overlap = answer_tokens & context_tokens
    precision = len(overlap) / len(answer_tokens)
    return round(precision, 4)


@tool(args_schema=GroundingInput)
def grounding_check(answer: str, context_chunks: list[str]) -> dict:
    """Validate whether the generated answer is grounded in the retrieved context.

    Returns a score between 0 and 1.  Scores below 0.7 indicate potential
    hallucination and should trigger answer regeneration.
    """
    score = _compute_grounding_score(answer, context_chunks)
    is_grounded = score >= 0.7
    return {
        "score": score,
        "is_grounded": is_grounded,
        "explanation": (
            f"Answer shares {score:.0%} token overlap with retrieved context. "
            + ("Grounded." if is_grounded else "Low grounding — may contain hallucinations.")
        ),
    }
