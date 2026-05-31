"""LangGraph shared state definition."""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """Shared state threaded through every node in the LangGraph workflow."""

    # Core query
    query: str

    # Accumulated conversation history (operator.add appends instead of replacing)
    chat_history: Annotated[list[BaseMessage], operator.add]

    # Retrieval outputs
    retrieved_docs: list[Document]
    sources: list[str]

    # Agent outputs
    reasoning_output: str
    validated_answer: str
    validation_score: float

    # Control flow
    next_agent: str        # "retrieval" | "reasoning" — set by supervisor
    retry_count: int       # incremented by validation_node; caps at max_retries
    error: Optional[str]   # set when an LLM call fails irrecoverably
