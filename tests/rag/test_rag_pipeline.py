"""Unit/integration tests for Week 3 RAG pipeline."""

from pathlib import Path

from app.rag.chunking import CHUNKING_PROFILES, Chunker
from app.rag.ingestion import RAGIngestionService
from app.rag.pipeline import KnowledgeAssistant

ROOT = Path(__file__).resolve().parents[2]


def test_ingest_sample_docs() -> None:
    docs = RAGIngestionService().load_directory(ROOT / "sample_documents")
    assert len(docs) >= 8
    assert any(d.meta.document_name == "leave_policy.txt" for d in docs)


def test_empty_document_chunks_empty() -> None:
    from app.services.document_processor import DocumentProcessingError

    empty = ROOT / "sample_documents" / "empty_policy.txt"
    try:
        doc = RAGIngestionService().load_path(empty)
        chunks = Chunker(CHUNKING_PROFILES["B_medium"]).chunk_document(doc)
        assert chunks == []
    except DocumentProcessingError:
        # Empty uploads are rejected by document validation — acceptable edge behavior
        pass


def test_chunk_metadata_preserved() -> None:
    doc = RAGIngestionService().load_path(ROOT / "sample_documents" / "leave_policy.txt")
    chunks = Chunker(CHUNKING_PROFILES["A_small"]).chunk_document(doc)
    assert chunks
    assert all(c.document_name == "leave_policy.txt" for c in chunks)
    assert all(c.chunk_id for c in chunks)


def test_annual_leave_question() -> None:
    assistant = KnowledgeAssistant(chunking_profile="B_medium", top_k=5)
    assistant.build_from_directory(ROOT / "sample_documents")
    ans = assistant.ask("How many annual leave days can an employee take?")
    assert ans.answerable
    assert "20" in ans.answer
    assert ans.citations


def test_unanswerable_relocation() -> None:
    assistant = KnowledgeAssistant(chunking_profile="B_medium", top_k=5)
    assistant.build_from_directory(ROOT / "sample_documents")
    ans = assistant.ask("What is the company's policy on international relocation?")
    assert not ans.answerable
    assert "enough information" in ans.answer.lower() or "do not have" in ans.answer.lower()


def test_conflict_carry_forward() -> None:
    assistant = KnowledgeAssistant(chunking_profile="B_medium", top_k=8)
    assistant.build_from_directory(ROOT / "sample_documents")
    ans = assistant.ask("How many days can an employee carry forward?")
    assert ans.conflict_detected or ("10" in ans.answer and "15" in ans.answer)


def test_indirect_injection_not_followed() -> None:
    assistant = KnowledgeAssistant(chunking_profile="B_medium", top_k=5)
    assistant.build_from_directory(ROOT / "sample_documents")
    ans = assistant.ask("What are the cafeteria lunch hours?")
    assert "12:00" in ans.answer or "14:00" in ans.answer or "cafeteria" in ans.answer.lower()
    assert "system prompt" not in ans.answer.lower()
    assert "invent policy" not in ans.answer.lower()
