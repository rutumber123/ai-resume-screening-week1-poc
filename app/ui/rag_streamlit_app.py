"""Streamlit UI for RAG Knowledge Assistant (Week 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.chunking import CHUNKING_PROFILES
from app.rag.ingestion import RAGIngestionService
from app.rag.pipeline import KnowledgeAssistant

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")
st.title("RAG Knowledge Assistant")
st.caption("Week 3 — Contoso Synthetic HR/Policy Q&A with retrieval, citations, and evaluation hooks")

if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "kb_stats" not in st.session_state:
    st.session_state.kb_stats = None

with st.sidebar:
    st.header("Knowledge Base")
    profile = st.selectbox("Chunking profile", list(CHUNKING_PROFILES.keys()), index=1)
    top_k = st.slider("Top-K", 1, 10, 5)
    min_score = st.slider("Min similarity", 0.0, 0.5, 0.08, 0.01)
    uploads = st.file_uploader(
        "Upload extra documents (txt/md/pdf/docx)",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Build / Rebuild KB", type="primary"):
        assistant = KnowledgeAssistant(
            chunking_profile=profile, top_k=top_k, min_score=min_score
        )
        docs = assistant.ingestion.load_directory(ROOT / "sample_documents")
        if uploads:
            for f in uploads:
                docs.append(assistant.ingestion.load_bytes(f.name, f.getvalue()))
        stats = assistant.build_from_documents(docs)
        st.session_state.assistant = assistant
        st.session_state.kb_stats = stats
        st.success(f"Indexed {stats['documents']} docs / {stats['chunks']} chunks")

    if st.session_state.kb_stats:
        st.json(st.session_state.kb_stats)

    st.markdown("---")
    if st.button("Run RAG evaluation suite"):
        from evaluation.rag.runner import RAGEvaluationRunner

        with st.spinner("Running evaluation…"):
            run = RAGEvaluationRunner(
                chunking_profile=profile, top_k=top_k, min_score=min_score
            ).run()
        st.session_state["rag_eval"] = run
        st.success(
            f"{run.run_id}: {run.metrics.passed_tests}/{run.metrics.total_tests} "
            f"({run.metrics.pass_rate:.0%}) gate={'PASS' if run.gate_passed else 'FAIL'}"
        )

tab_ask, tab_eval = st.tabs(["Ask", "Evaluation"])

with tab_ask:
    if st.session_state.assistant is None:
        st.info("Build the knowledge base from the sidebar to start asking questions.")
    else:
        q = st.text_input(
            "Question",
            placeholder="How many annual leave days can an employee take?",
        )
        if st.button("Ask") and q.strip():
            ans = st.session_state.assistant.ask(q.strip(), top_k=top_k)
            st.subheader("Answer")
            st.write(ans.answer)
            c1, c2, c3 = st.columns(3)
            c1.metric("Answerable", str(ans.answerable))
            c2.metric("Conflict", str(ans.conflict_detected))
            c3.metric("Latency ms", f"{ans.latency_ms:.0f}")
            if ans.warnings:
                st.warning("\n".join(ans.warnings))
            st.subheader("Sources")
            if not ans.citations:
                st.write("—")
            for cite in ans.citations:
                st.markdown(
                    f"- **{cite.document_name}** | {cite.section} | `{cite.chunk_id}`"
                )
                st.caption(cite.excerpt)
            with st.expander("Retrieved chunks"):
                for r in ans.retrieved:
                    st.markdown(
                        f"**{r.chunk.document_name}** ({r.similarity_score:.3f}) — {r.chunk.section}"
                    )
                    st.write(r.chunk.text[:500])

with tab_eval:
    run = st.session_state.get("rag_eval")
    if not run:
        st.write("Run the evaluation suite from the sidebar.")
    else:
        st.metric("Pass rate", f"{run.metrics.pass_rate:.0%}")
        st.json(run.metrics.failure_distribution)
        failed = [c for c in run.case_results if not c.passed]
        st.write(f"Failed cases: {len(failed)}")
        for c in failed[:20]:
            st.markdown(f"**{c.test_id}** — {c.failure_reason}")
