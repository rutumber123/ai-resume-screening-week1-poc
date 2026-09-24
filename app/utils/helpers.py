"""Utility helpers."""

from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from typing import Generator, Iterable, List, Optional


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_skill(skill: str) -> str:
    return re.sub(r"[^a-z0-9+#./\s-]", "", (skill or "").lower()).strip()


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def safe_filename_hash(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]


def extract_years(text: str) -> Optional[float]:
    """Extract a years-of-experience number from free text when present."""
    if not text:
        return None
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)",
        r"(?:experience|exp)[:\s]+(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


@contextmanager
def timed_operation(label: str, logger) -> Generator[dict, None, None]:
    start = time.perf_counter()
    meta: dict = {"label": label}
    try:
        yield meta
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s completed in %.1f ms", label, elapsed_ms)
        meta["elapsed_ms"] = elapsed_ms
