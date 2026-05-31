"""Reasoning agent: GPT-4o over retrieved context with retry + fallback."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_rag.config import settings
from multi_agent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert financial analyst assistant.

Answer the user's question based ONLY on the provided document excerpts.
Be precise. Cite specific figures, dates, and sources when available.
If the context does not contain enough information, state that clearly
rather than guessing or fabricating data.

Format your response as:
1. A direct answer to the question.
2. Key supporting facts from the documents (with source citations).
3. Any important caveats or data limitations.
"""


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        parts.append(f"[Excerpt {i} — {source}, page {page + 1}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_llm(llm: ChatOpenAI, messages: list) -> str:
    response = llm.invoke(messages)
    return response.content


def run_reasoning(query: str, retrieved_docs: list[Document]) -> dict:
    """
    Generate an answer using GPT-4o over the retrieved documents.

    Retry strategy:
    - Retries up to 3× on RateLimitError / APIError with exponential backoff.
    - Falls back to GPT-3.5-turbo if GPT-4o fails entirely.
    - Returns error state if both models fail.
    """
    context = _format_context(retrieved_docs)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context Documents:\n\n{context}\n\n---\n\nQuestion: {query}"
        ),
    ]

    # ── Primary: GPT-4o ───────────────────────────────────────────────────────
    try:
        llm = ChatOpenAI(
            model=settings.gpt4o_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
        output = _call_llm(llm, messages)
        logger.info("reasoning_complete", model=settings.gpt4o_model)
        return {"reasoning_output": output, "error": None}
    except Exception as primary_err:
        logger.warning(
            "primary_model_failed",
            model=settings.gpt4o_model,
            error=str(primary_err),
        )

    # ── Fallback: GPT-3.5-turbo ───────────────────────────────────────────────
    try:
        fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
        output = _call_llm(fallback_llm, messages)
        logger.info("reasoning_complete", model=settings.fallback_model + " (fallback)")
        return {"reasoning_output": output, "error": None}
    except Exception as fallback_err:
        logger.error("fallback_model_failed", error=str(fallback_err))
        return {
            "reasoning_output": "",
            "error": f"Both models failed. Last error: {fallback_err}",
        }
