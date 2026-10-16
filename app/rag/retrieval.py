"""Retrieval service."""

from __future__ import annotations

from typing import List

from app.rag.models import RetrievedChunk
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(
        self, question: str, top_k: int = 5, min_score: float = 0.05
    ) -> List[RetrievedChunk]:
        return self.store.search(question, top_k=top_k, min_score=min_score)
