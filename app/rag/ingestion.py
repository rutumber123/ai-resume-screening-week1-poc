"""Document ingestion for RAG."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from app.core.logging import get_logger
from app.rag.models import DocumentMeta, DocumentRecord
from app.services.document_processor import DocumentProcessingError, DocumentProcessor

logger = get_logger(__name__)


class RAGIngestionService:
    def __init__(self) -> None:
        self.docs = DocumentProcessor()

    def load_path(self, path: Path) -> DocumentRecord:
        content = path.read_bytes()
        text, warnings = self.docs.extract_text(path.name, content)
        text = self.clean_text(text)
        doc_id = self._extract_doc_id(text) or path.stem.upper()
        meta = DocumentMeta(
            document_id=doc_id,
            document_name=path.name,
            document_type=path.suffix.lstrip(".").lower() or "txt",
            source_path=str(path),
            extra={"warnings": warnings},
        )
        if not text.strip():
            meta.extra["empty"] = True
        return DocumentRecord(meta=meta, text=text)

    def load_bytes(self, filename: str, content: bytes) -> DocumentRecord:
        text, warnings = self.docs.extract_text(filename, content)
        text = self.clean_text(text)
        doc_id = self._extract_doc_id(text) or Path(filename).stem.upper()
        return DocumentRecord(
            meta=DocumentMeta(
                document_id=doc_id,
                document_name=filename,
                document_type=Path(filename).suffix.lstrip(".").lower() or "txt",
                source_path=filename,
                extra={"warnings": warnings},
            ),
            text=text,
        )

    def load_directory(self, directory: Path) -> List[DocumentRecord]:
        records: List[DocumentRecord] = []
        for path in sorted(directory.glob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
                continue
            try:
                records.append(self.load_path(path))
            except DocumentProcessingError as exc:
                logger.error("Skipping %s: %s", path.name, exc)
        return records

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_doc_id(text: str) -> str | None:
        match = re.search(r"Document ID:\s*([A-Z0-9\-]+)", text, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None
