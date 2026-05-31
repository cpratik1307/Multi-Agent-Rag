"""Streamlit chat interface for the Multi-Agent RAG system."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent RAG — Financial QA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached pipeline builder ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔨 Building vector index — this may take a minute…")
def get_pipeline(pdf_dir: str):
    """Build the EnsembleRetriever and compiled LangGraph for a given pdf_dir."""
    from multi_agent_rag.graph.builder import build_graph
    from multi_agent_rag.tools.search_tool import _set_retriever
    from multi_agent_rag.vectorstore.retriever import build_ensemble_retriever

    retriever = build_ensemble_retriever(pdf_dir=pdf_dir)
    _set_retriever(retriever)
    return build_graph(use_checkpointer=True)


def _make_initial_state(query: str) -> dict:
    return {
        "query": query,
        "chat_history": [HumanMessage(content=query)],
        "retrieved_docs": [],
        "sources": [],
        "reasoning_output": "",
        "validated_answer": "",
        "validation_score": 0.0,
        "next_agent": "retrieval",
        "retry_count": 0,
        "error": None,
    }


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📊 Multi-Agent RAG — Financial Document QA")
    st.caption(
        "Powered by **LangGraph** · **GPT-4o** · **Chroma + BM25 Hybrid Search** · **LangSmith**"
    )

    # ── Sidebar: document upload ──────────────────────────────────────────────
    with st.sidebar:
        st.header("📁 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDF reports",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more financial PDF reports to query.",
        )

        pdf_dir: str | None = None
        index_ready = False

        if uploaded_files:
            # Persist files to a temp directory for the lifetime of the session
            if "tmp_dir" not in st.session_state:
                st.session_state.tmp_dir = tempfile.mkdtemp()
            tmp_dir = Path(st.session_state.tmp_dir)
            for uf in uploaded_files:
                dest = tmp_dir / uf.name
                if not dest.exists():
                    dest.write_bytes(uf.read())
            pdf_dir = str(tmp_dir)
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded.")
            index_ready = True

        elif Path("data/sample_reports").exists() and any(
            Path("data/sample_reports").glob("*.pdf")
        ):
            pdf_dir = "data/sample_reports"
            st.info("📂 Using sample reports from `data/sample_reports/`.")
            index_ready = True
        else:
            st.warning(
                "Upload PDFs above, or add them to `data/sample_reports/` and restart."
            )

        st.divider()
        st.markdown("**Model pipeline**")
        st.markdown("- Primary: `gpt-4o`")
        st.markdown("- Fallback: `gpt-3.5-turbo`")
        st.markdown("- Embeddings: `text-embedding-3-small`")
        st.markdown("- Retrieval: Chroma + BM25 (RRF)")
        st.markdown("- Tracing: LangSmith")

        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    # ── Session state ─────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # ── Render conversation history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")
            if msg.get("confidence") is not None:
                st.progress(
                    float(msg["confidence"]),
                    text=f"Grounding confidence: {msg['confidence']:.0%}",
                )

    # ── Query input ───────────────────────────────────────────────────────────
    if not index_ready:
        st.stop()

    if prompt := st.chat_input("Ask a question about the financial reports…"):
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the LangGraph pipeline
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking — supervisor → retrieval → reasoning → validation…"):
                graph = get_pipeline(pdf_dir)  # type: ignore[arg-type]
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke(_make_initial_state(prompt), config=config)

            answer = result.get("validated_answer") or "⚠️ No answer generated."
            score = float(result.get("validation_score", 0.0))
            sources = result.get("sources", [])

            st.markdown(answer)
            if sources:
                with st.expander("📄 Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")
            st.progress(score, text=f"Grounding confidence: {score:.0%}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": score,
            }
        )


if __name__ == "__main__":
    main()
