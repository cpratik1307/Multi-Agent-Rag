"""Retrieval agent: wraps the hybrid search tool and formats results."""

from __future__ import annotations

from langchain_core.documents import Document

from multi_agent_rag.config import settings
from multi_agent_rag.tools.search_tool import hybrid_search
from multi_agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


def run_retrieval(query: str) -> dict:
    """
    Execute hybrid FAISS + BM25 search and return retrieved docs with sources.

    Returns a partial state update dict consumed by the LangGraph node.
    """
    logger.info("retrieval_start", query=query[:80])

    raw_results: list[dict] = hybrid_search.invoke(
        {"query": query, "top_k": settings.top_k}
    )

    docs = [
        Document(
            page_content=r["content"],
            metadata={"source": r["source"], "page": r["page"]},
        )
        for r in raw_results
    ]

    # Deduplicated source citations: "filename.pdf (page N)"
    sources = sorted(
        {f"{r['source']} (page {r['page'] + 1})" for r in raw_results}
    )

    logger.info("retrieval_complete", chunks=len(docs), sources=len(sources))
    return {"retrieved_docs": docs, "sources": sources}
