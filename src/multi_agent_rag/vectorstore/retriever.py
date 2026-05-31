"""Build the hybrid EnsembleRetriever (FAISS dense + BM25 sparse → RRF merge)."""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from multi_agent_rag.config import settings
from multi_agent_rag.vectorstore.ingestion import load_and_split_pdfs


def build_ensemble_retriever(
    pdf_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    top_k: int | None = None,
) -> EnsembleRetriever:
    """
    Return an EnsembleRetriever that fuses dense (FAISS) and sparse (BM25)
    results using Reciprocal Rank Fusion (weights 0.6 / 0.4).

    Priority:
    1. Load saved FAISS index + docs.pkl from *index_path*.
    2. If index not found, build from *pdf_dir* and persist it.
    """
    k = top_k or settings.top_k
    index_path = Path(index_path or settings.faiss_index_path)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    docs_pkl = index_path / "docs.pkl"
    faiss_index_file = index_path / "index.faiss"

    if faiss_index_file.exists() and docs_pkl.exists():
        # Load pre-built artifacts
        faiss_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        with open(docs_pkl, "rb") as fh:
            docs: list[Document] = pickle.load(fh)
    else:
        # Build from scratch
        if pdf_dir is None:
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run ingestion first, or pass pdf_dir= to build it now."
            )
        docs = load_and_split_pdfs(pdf_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        faiss_store = FAISS.from_documents(docs, embeddings)
        faiss_store.save_local(str(index_path))
        with open(docs_pkl, "wb") as fh:
            pickle.dump(docs, fh)

    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(docs, k=k)

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
