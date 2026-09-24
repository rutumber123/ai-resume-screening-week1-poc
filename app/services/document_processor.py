"""Document text extraction for common resume formats."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessingError(Exception):
    """Raised when a document cannot be processed."""


class DocumentProcessor:
    """Extract plain text from uploaded resume bytes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate_file(self, filename: str, content: bytes) -> None:
        if not filename:
            raise DocumentProcessingError("Filename is required.")
        suffix = Path(filename).suffix.lower()
        if suffix not in self.settings.supported_extension_list:
            raise DocumentProcessingError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(self.settings.supported_extension_list)}"
            )
        if len(content) == 0:
            raise DocumentProcessingError("File is empty.")
        if len(content) > self.settings.max_file_size_bytes:
            raise DocumentProcessingError(
                f"File exceeds max size of {self.settings.max_file_size_mb} MB."
            )

    def extract_text(self, filename: str, content: bytes) -> Tuple[str, list[str]]:
        """
        Return (text, warnings).
        Does not invent content; empty extraction yields a warning.
        """
        self.validate_file(filename, content)
        suffix = Path(filename).suffix.lower()
        warnings: list[str] = []
        logger.info("Extracting text from file type=%s size=%s", suffix, len(content))

        try:
            if suffix in {".txt", ".md"}:
                text = self._extract_plain(content)
            elif suffix == ".pdf":
                text = self._extract_pdf(content)
            elif suffix == ".docx":
                text = self._extract_docx(content)
            else:
                raise DocumentProcessingError(f"Unsupported file type: {suffix}")
        except DocumentProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DocumentProcessingError(
                f"Failed to parse document '{filename}': {exc}"
            ) from exc

        text = (text or "").strip()
        if not text:
            warnings.append("No extractable text found in document.")
        if len(text) > 100_000:
            warnings.append("Document is unusually long; truncated for processing.")
            text = text[:100_000]
        return text, warnings

    def _extract_plain(self, content: bytes) -> str:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")

    def _extract_pdf(self, content: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentProcessingError("pypdf is required for PDF support.") from exc

        reader = PdfReader(io.BytesIO(content))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    def _extract_docx(self, content: bytes) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise DocumentProcessingError(
                "python-docx is required for DOCX support."
            ) from exc

        document = Document(io.BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs if p.text)
