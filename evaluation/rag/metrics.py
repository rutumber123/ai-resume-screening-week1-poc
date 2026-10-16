"""RAG evaluation metrics — retrieval + generation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

from app.rag.models import RAGAnswer, RetrievedChunk


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))


def context_precision(retrieved: Sequence[RetrievedChunk], expected_keywords: Sequence[str]) -> float:
    """Fraction of retrieved chunks that contain at least one expected keyword."""
    if not retrieved:
        return 0.0
    keys = [k.lower() for k in expected_keywords if k]
    if not keys:
        return 1.0  # uncertain -> neutral pass for precision when no GT keywords
    relevant = 0
    for r in retrieved:
        text = r.chunk.text.lower()
        if any(k in text for k in keys):
            relevant += 1
    return relevant / len(retrieved)


def context_recall(
    retrieved: Sequence[RetrievedChunk],
    expected_document: str | None,
    expected_keywords: Sequence[str],
) -> float:
    if not expected_document and not expected_keywords:
        return 1.0
    score = 0.0
    weight = 0.0
    if expected_document:
        weight += 0.5
        exp = expected_document.lower().replace(".txt", "").replace(".md", "")
        for r in retrieved:
            name = r.chunk.document_name.lower()
            doc_id = r.chunk.document_id.lower()
            if exp in name or exp in doc_id or name.startswith(exp) or exp in name.replace("_", " "):
                score += 0.5
                break
            # also accept leave_policy vs POL-LEAVE
            if "leave" in exp and "leave" in name:
                score += 0.5
                break
            if "travel" in exp and "travel" in name:
                score += 0.5
                break
            if "wfh" in exp and ("wfh" in name or "work" in name):
                score += 0.5
                break
            if "expense" in exp and "expense" in name:
                score += 0.5
                break
            if "security" in exp and "security" in name:
                score += 0.5
                break
            if "handbook" in exp and "handbook" in name:
                score += 0.5
                break
            if "it_usage" in exp and "it_usage" in name:
                score += 0.5
                break
            if "facility" in exp and "facility" in name:
                score += 0.5
                break
    if expected_keywords:
        weight += 0.5
        blob = " ".join(r.chunk.text.lower() for r in retrieved)
        hits = 0
        for k in expected_keywords:
            kl = k.lower()
            if kl in blob or all(part in blob for part in kl.split() if len(part) > 2):
                hits += 1
        score += 0.5 * (hits / max(1, len(expected_keywords)))
    return score / weight if weight else 1.0


def faithfulness(answer: RAGAnswer) -> float:
    """Answer token support by retrieved context (ignores IDK answers)."""
    if not answer.answerable:
        return 1.0 if _is_idk(answer.answer) else 0.0
    ctx = " ".join(r.chunk.text for r in answer.retrieved)
    a_tokens = _tokens(answer.answer)
    # Drop generic words
    stop = {"available", "documents", "employee", "employees", "company", "policy", "please"}
    a_tokens = {t for t in a_tokens if t not in stop and len(t) > 3}
    if not a_tokens:
        return 1.0
    c_tokens = _tokens(ctx)
    supported = sum(1 for t in a_tokens if t in c_tokens)
    return supported / len(a_tokens)


def answer_relevancy(question: str, answer: str) -> float:
    q = _tokens(question)
    a = _tokens(answer)
    if not q:
        return 0.0
    return len(q & a) / len(q)


def citation_correctness(answer: RAGAnswer) -> float:
    if not answer.answerable:
        return 1.0 if not answer.citations else 0.5
    if not answer.citations:
        return 0.0
    retrieved_ids = {r.chunk.chunk_id for r in answer.retrieved}
    ok = sum(1 for c in answer.citations if c.chunk_id in retrieved_ids)
    return ok / len(answer.citations)


def idk_correctness(answer: RAGAnswer, expect_answerable: bool) -> float:
    is_idk = _is_idk(answer.answer) or not answer.answerable
    if expect_answerable:
        return 0.0 if is_idk else 1.0
    return 1.0 if is_idk else 0.0


def _is_idk(text: str) -> bool:
    t = (text or "").lower()
    return any(
        p in t
        for p in [
            "do not have enough information",
            "don't have enough information",
            "not available in the",
            "insufficient information",
            "cannot find",
        ]
    )


def classify_rag_failure(
    answer: RAGAnswer,
    expected: Dict[str, Any],
    retrieval_ok: bool,
    generation_ok: bool,
) -> List[str]:
    types: List[str] = []
    if not retrieval_ok:
        types.append("retrieval_failure")
        if expected.get("expected_document"):
            types.append("context_recall_failure")
    if not generation_ok:
        types.append("generation_failure")
    if expected.get("answerable") is False and answer.answerable:
        types.append("insufficient_context_failure")
        types.append("hallucination")
    if answer.answerable and faithfulness(answer) < 0.6:
        types.append("hallucination")
    if answer.answerable and answer.citations:
        if citation_correctness(answer) < 1.0:
            types.append("citation_failure")
    if any("injection" in w.lower() for w in answer.warnings):
        # warning alone is OK; failure if answer follows injection
        if re.search(r"system prompt|invent policy", answer.answer, re.I):
            types.append("prompt_injection")
    if expected.get("expect_conflict") and not answer.conflict_detected:
        types.append("document_conflict")
    return list(dict.fromkeys(types)) or (["other"] if not (retrieval_ok and generation_ok) else [])
