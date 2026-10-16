"""Embedding abstractions — TF-IDF offline default; optional OpenAI."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#\-]*", re.I)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Lightweight local embedder — no API key, no model download."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> None:
        docs = [tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        # Keep frequent-enough terms; always keep something
        terms = sorted(df.keys())
        self.vocab = {t: i for i, t in enumerate(terms)}
        n = max(1, len(docs))
        idf = np.zeros(len(terms), dtype=np.float64)
        for t, i in self.vocab.items():
            idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0
        self.idf = idf
        logger.info("TF-IDF embedder fitted vocab=%s docs=%s", len(self.vocab), n)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf is None or not self.vocab:
            self.fit(texts or ["placeholder"])
        rows = []
        dim = len(self.vocab)
        for text in texts:
            tokens = tokenize(text)
            counts = Counter(tokens)
            vec = np.zeros(dim, dtype=np.float64)
            for t, c in counts.items():
                idx = self.vocab.get(t)
                if idx is not None:
                    vec[idx] = c
            # tf normalize
            total = vec.sum()
            if total > 0:
                vec = vec / total
            vec = vec * self.idf
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec)
        return np.vstack(rows) if rows else np.zeros((0, dim))


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.model = model
        self._dim = 1536

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim))
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        vectors = [np.array(item.embedding, dtype=np.float64) for item in response.data]
        arr = np.vstack(vectors)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_embedder(provider: str = "tfidf") -> Embedder:
    provider = (provider or "tfidf").lower()
    if provider == "openai":
        return OpenAIEmbedder()
    return TfidfEmbedder()
