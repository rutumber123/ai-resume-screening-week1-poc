"""RAG domain models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    document_id: str
    document_name: str
    document_type: str = "txt"
    source_path: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    meta: DocumentMeta
    text: str


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    section: str = "Unknown"
    page_number: Optional[int] = None
    source_identifier: str
    text: str
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    similarity_score: float


class Citation(BaseModel):
    document_name: str
    section: str
    page_number: Optional[int] = None
    chunk_id: str
    source_identifier: str
    excerpt: str = ""


class RAGAnswer(BaseModel):
    question: str
    answer: str
    answerable: bool
    citations: List[Citation] = Field(default_factory=list)
    retrieved: List[RetrievedChunk] = Field(default_factory=list)
    conflict_detected: bool = False
    warnings: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_top_k: int = 0
    chunking_profile: str = ""
    prompt_version: str = "rag_v1"
