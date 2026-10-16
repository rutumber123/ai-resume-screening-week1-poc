"""In-memory vector store abstraction."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from app.rag.embeddings import Embedder
from app.rag.models import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self.matrix: np.ndarray | None = None

    def clear(self) -> None:
        self.chunks = []
        self.matrix = None

    def add_chunks(self, chunks: Sequence[Chunk], refit: bool = True) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        texts = [c.text for c in self.chunks]
        if refit:
            self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        if not self.chunks or self.matrix is None or self.matrix.size == 0:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)
        results: List[RetrievedChunk] = []
        for idx in order[: max(1, top_k)]:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(
                RetrievedChunk(chunk=self.chunks[int(idx)], similarity_score=round(score, 4))
            )
        return results

    def stats(self) -> dict:
        return {
            "chunk_count": len(self.chunks),
            "documents": sorted({c.document_id for c in self.chunks}),
        }
