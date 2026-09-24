"""Unit tests for document processing and file validation."""

import pytest

from app.core.config import Settings
from app.services.document_processor import DocumentProcessingError, DocumentProcessor


@pytest.fixture
def processor() -> DocumentProcessor:
    return DocumentProcessor(Settings(LLM_PROVIDER="mock"))


def test_reject_unsupported_extension(processor: DocumentProcessor) -> None:
    with pytest.raises(DocumentProcessingError):
        processor.validate_file("resume.exe", b"abc")


def test_reject_empty_file(processor: DocumentProcessor) -> None:
    with pytest.raises(DocumentProcessingError):
        processor.validate_file("resume.txt", b"")


def test_reject_oversized_file(processor: DocumentProcessor) -> None:
    small = DocumentProcessor(Settings(LLM_PROVIDER="mock", MAX_FILE_SIZE_MB=0.000001))
    with pytest.raises(DocumentProcessingError):
        small.validate_file("resume.txt", b"hello world" * 100)


def test_extract_plain_text(processor: DocumentProcessor) -> None:
    text, warnings = processor.extract_text("candidate.txt", b"Ada Lovelace\nPython")
    assert "Ada Lovelace" in text
    assert warnings == []


def test_extract_empty_text_warns(processor: DocumentProcessor) -> None:
    # whitespace-only after decode still counts as empty strip
    text, warnings = processor.extract_text("blank.txt", b"   \n  ")
    assert text == ""
    assert any("No extractable text" in w for w in warnings)
