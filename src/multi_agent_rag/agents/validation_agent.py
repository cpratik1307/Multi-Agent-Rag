"""Validation / grounding agent: checks if the answer is supported by context."""

from __future__ import annotations

from langchain_core.documents import Document

from multi_agent_rag.config import settings
from multi_agent_rag.tools.grounding_tool import grounding_check
from multi_agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


def run_validation(reasoning_output: str, retrieved_docs: list[Document]) -> dict:
    """
    Validate the reasoning output against the retrieved context.

    Invokes the grounding_check tool, which computes token-overlap precision
    between the answer and the source document chunks.

    Returns a partial state update with validation_score and validated_answer.
    A blank validated_answer signals to the graph that a retry is needed.
    """
    context_chunks = [doc.page_content for doc in retrieved_docs]
    result: dict = grounding_check.invoke(
        {"answer": reasoning_output, "context_chunks": context_chunks}
    )
    score: float = result["score"]
    is_grounded: bool = result["is_grounded"]

    logger.info(
        "validation_result",
        score=score,
        threshold=settings.grounding_threshold,
        grounded=is_grounded,
    )

    return {
        "validation_score": score,
        # Only populate validated_answer if grounded; else blank triggers retry
        "validated_answer": reasoning_output if is_grounded else "",
    }
