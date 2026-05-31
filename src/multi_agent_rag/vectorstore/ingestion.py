"""PDF ingestion pipeline: load → split → embed → FAISS + save BM25 docs."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from multi_agent_rag.config import settings

logger = logging.getLogger(__name__)


def load_and_split_pdfs(pdf_dir: str | Path) -> list[Document]:
    """Load all PDFs in *pdf_dir* and split into overlapping text chunks."""
    pdf_dir = Path(pdf_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    all_docs: list[Document] = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'.")

    for pdf_path in pdf_files:
        logger.info("Loading %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)
        logger.info("  → %d chunks from %s", len(chunks), pdf_path.name)

    return all_docs


def build_vectorstore(
    pdf_dir: str | Path,
    index_path: str | Path | None = None,
) -> list[Document]:
    """
    Ingest PDFs from *pdf_dir*, build a FAISS vector index, and persist it.

    Also saves the raw document list as `docs.pkl` alongside the index so that
    the BM25 retriever can be reconstructed without re-embedding.
    """
    index_path = Path(index_path or settings.faiss_index_path)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    docs = load_and_split_pdfs(pdf_dir)
    logger.info("Building FAISS index from %d chunks…", len(docs))

    index_path.mkdir(parents=True, exist_ok=True)
    faiss_store = FAISS.from_documents(docs, embeddings)
    faiss_store.save_local(str(index_path))

    # Persist raw docs so BM25Retriever can be rebuilt without re-embedding
    docs_pkl = index_path / "docs.pkl"
    with open(docs_pkl, "wb") as fh:
        pickle.dump(docs, fh)

    logger.info("Index and docs saved to '%s'.", index_path)
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_vectorstore(
        pdf_dir="data/sample_reports",
        index_path=settings.faiss_index_path,
    )
    print("✅ Ingestion complete.")
