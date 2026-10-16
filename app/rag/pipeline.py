"""End-to-end RAG knowledge assistant pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from app.core.logging import get_logger
from app.rag.chunking import CHUNKING_PROFILES, Chunker, ChunkingConfig
from app.rag.embeddings import get_embedder
from app.rag.generation import RAGGenerator
from app.rag.ingestion import RAGIngestionService
from app.rag.models import Chunk, DocumentRecord, RAGAnswer
from app.rag.retrieval import Retriever
from app.rag.vector_store import VectorStore
from evaluation.observability import evaluation_span

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "sample_documents"


class KnowledgeAssistant:
    def __init__(
        self,
        chunking_profile: str = "B_medium",
        embedder_provider: str = "tfidf",
        top_k: int = 5,
        min_score: float = 0.08,
    ) -> None:
        self.chunking_profile = chunking_profile
        self.top_k = top_k
        self.min_score = min_score
        config = CHUNKING_PROFILES.get(chunking_profile, CHUNKING_PROFILES["B_medium"])
        self.chunker = Chunker(config)
        self.ingestion = RAGIngestionService()
        self.embedder = get_embedder(embedder_provider)
        self.store = VectorStore(self.embedder)
        self.retriever = Retriever(self.store)
        self.generator = RAGGenerator()
        self.documents: List[DocumentRecord] = []
        self.chunks: List[Chunk] = []

    def build_from_directory(self, directory: Path | None = None) -> dict:
        directory = directory or DEFAULT_DOCS
        docs = self.ingestion.load_directory(directory)
        return self.build_from_documents(docs)

    def build_from_documents(self, documents: Sequence[DocumentRecord]) -> dict:
        self.documents = list(documents)
        self.store.clear()
        self.chunks = []
        for doc in documents:
            self.chunks.extend(self.chunker.chunk_document(doc))
        self.store.add_chunks(self.chunks, refit=True)
        stats = {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "chunking_profile": self.chunking_profile,
            "empty_documents": sum(1 for d in self.documents if not d.text.strip()),
        }
        logger.info("Knowledge base built %s", stats)
        return stats

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        k = top_k if top_k is not None else self.top_k
        start = time.perf_counter()
        query = self._expand_query(question)
        with evaluation_span(
            "rag_ask",
            {"question_preview": (question or "")[:120], "top_k": k},
        ) as span:
            retrieved = self.retriever.retrieve(
                query, top_k=k, min_score=self.min_score
            )
            answer = self.generator.generate(
                question, retrieved, min_score=self.min_score
            )
            answer.latency_ms = round((time.perf_counter() - start) * 1000, 2)
            answer.retrieval_top_k = k
            answer.chunking_profile = self.chunking_profile
            span["events"].append(
                {
                    "retrieved_chunk_ids": [r.chunk.chunk_id for r in retrieved],
                    "scores": [r.similarity_score for r in retrieved],
                    "answerable": answer.answerable,
                }
            )
            return answer

    @staticmethod
    def _expand_query(question: str) -> str:
        q = question
        lower = question.lower()
        extras = []
        if any(w in lower for w in ("wfh", "remote", "work from home", "hybrid")):
            extras.append("work from home VPN stipend core hours")
        if "carry" in lower:
            extras.append("carry forward annual leave 10 days 15 days")
        if "password" in lower or "mfa" in lower:
            extras.append("password MFA VPN IT usage")
        if "cafeteria" in lower or "office open" in lower:
            extras.append("cafeteria facility office access lunch")
        if "reloc" in lower:
            extras.append("international relocation")
        if extras:
            q = question + " " + " ".join(extras)
        return q

    def rebuild_with_profile(self, profile: str) -> dict:
        self.chunking_profile = profile
        self.chunker = Chunker(CHUNKING_PROFILES.get(profile, CHUNKING_PROFILES["B_medium"]))
        return self.build_from_documents(self.documents)
