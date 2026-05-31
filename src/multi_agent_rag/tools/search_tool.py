"""Hybrid search tool with Pydantic schema validation.

The global retriever is lazy-initialized on first call.  The Streamlit app
(or tests) can inject a pre-built retriever via `_set_retriever()`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document


class SearchInput(BaseModel):
    query: str = Field(
        description="The natural-language question or keyword query to search financial reports."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of document chunks to return.",
    )


# Module-level retriever (dependency-injected from outside)
_retriever: EnsembleRetriever | None = None


def _get_retriever() -> EnsembleRetriever:
    global _retriever
    if _retriever is None:
        from multi_agent_rag.vectorstore.retriever import build_ensemble_retriever

        _retriever = build_ensemble_retriever()
    return _retriever


def _set_retriever(retriever: EnsembleRetriever) -> None:
    """Inject a pre-built retriever (called by Streamlit on PDF upload)."""
    global _retriever
    _retriever = retriever


@tool(args_schema=SearchInput)
def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Search financial reports using hybrid FAISS + BM25 retrieval.

    Returns a ranked list of document chunks with source and page metadata.
    The ranking uses Reciprocal Rank Fusion (RRF) to blend dense semantic
    similarity with sparse keyword relevance.
    """
    retriever = _get_retriever()
    # Update k on both sub-retrievers before calling
    retriever.retrievers[0].search_kwargs["k"] = top_k
    retriever.retrievers[1].k = top_k

    docs: list[Document] = retriever.invoke(query)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0),
        }
        for doc in docs
    ]
