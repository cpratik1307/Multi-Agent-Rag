"""Supervisor agent: uses GPT-4o to decide whether to retrieve or reason."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from multi_agent_rag.config import settings
from multi_agent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a routing supervisor for a financial document QA system.

Analyze the user's query and decide the optimal next step.

Available routes:
- "retrieval": The query requires searching financial documents for specific data,
  numbers, facts, or events not already present in the conversation.
- "reasoning": The retrieved documents already contain sufficient context to
  answer the question directly — no new retrieval is needed.

Respond ONLY with valid JSON in this exact format (no markdown fences):
{"next_agent": "retrieval", "reasoning": "brief explanation"}
"""


def build_supervisor_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.gpt4o_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )


def supervisor_decision(query: str, has_retrieved_docs: bool) -> dict:
    """
    Decide whether to retrieve documents or proceed directly to reasoning.

    When no documents have been retrieved yet, skips the LLM call and routes
    straight to retrieval (avoids an unnecessary API round-trip).
    """
    if not has_retrieved_docs:
        return {"next_agent": "retrieval", "reasoning": "No documents retrieved yet."}

    llm = build_supervisor_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Query: {query}\n"
                "Retrieved documents are already available in context.\n"
                "Should we perform another retrieval pass or proceed to reasoning?"
            )
        ),
    ]
    response = llm.invoke(messages)
    try:
        result = json.loads(response.content)
        logger.info("supervisor_decision", decision=result)
        return result
    except (json.JSONDecodeError, KeyError):
        logger.warning("supervisor_invalid_json", raw=response.content)
        return {"next_agent": "retrieval", "reasoning": "JSON parse error; defaulting to retrieval."}
