"""Configurable document chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from app.rag.models import Chunk, DocumentRecord


@dataclass(frozen=True)
class ChunkingConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    strategy: str = "recursive_section"  # recursive_section | fixed


CHUNKING_PROFILES: Dict[str, ChunkingConfig] = {
    "A_small": ChunkingConfig(name="A_small", chunk_size=350, chunk_overlap=50),
    "B_medium": ChunkingConfig(name="B_medium", chunk_size=700, chunk_overlap=100),
    "C_large": ChunkingConfig(name="C_large", chunk_size=1200, chunk_overlap=150),
}


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or CHUNKING_PROFILES["B_medium"]

    def chunk_document(self, document: DocumentRecord) -> List[Chunk]:
        if not document.text.strip():
            return []
        sections = self._split_sections(document.text)
        chunks: List[Chunk] = []
        counter = 0
        for section_name, section_text in sections:
            pieces = self._window(section_text)
            for piece, start, end in pieces:
                counter += 1
                chunk_id = f"{document.meta.document_id}::C{counter:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.meta.document_id,
                        document_name=document.meta.document_name,
                        section=section_name,
                        page_number=None,
                        source_identifier=f"{document.meta.document_name}#{section_name}",
                        text=piece.strip(),
                        start_char=start,
                        end_char=end,
                        metadata={
                            "chunking_profile": self.config.name,
                            "document_type": document.meta.document_type,
                        },
                    )
                )
        return chunks

    def _split_sections(self, text: str) -> List[tuple[str, str]]:
        pattern = re.compile(r"(?m)^(#{1,3}\s+.+|[A-Z][A-Za-z0-9 ./\-]{3,80})$")
        lines = text.splitlines()
        sections: List[tuple[str, list[str]]] = [("Preamble", [])]
        for line in lines:
            if line.startswith("## ") or re.match(r"^##?\s+\d+\.", line):
                sections.append((line.lstrip("# ").strip()[:80], []))
            else:
                sections[-1][1].append(line)
        out = []
        for name, body_lines in sections:
            body = "\n".join(body_lines).strip()
            if body:
                out.append((name, body))
        return out or [("Body", text)]

    def _window(self, text: str) -> List[tuple[str, int, int]]:
        size = self.config.chunk_size
        overlap = min(self.config.chunk_overlap, size // 2)
        if len(text) <= size:
            return [(text, 0, len(text))]
        pieces = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            # prefer break on paragraph
            if end < len(text):
                break_at = text.rfind("\n\n", start, end)
                if break_at > start + size // 3:
                    end = break_at
            pieces.append((text[start:end], start, end))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return pieces
