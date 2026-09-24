"""Job description parsing into structured requirements."""

from __future__ import annotations

import re
from typing import List, Optional

from app.core.constants import NOT_SPECIFIED
from app.core.logging import get_logger
from app.models.schemas import JobDescription
from app.utils.helpers import extract_years, normalize_text, unique_preserve_order

logger = get_logger(__name__)


class JDProcessor:
    """Parse natural-language JDs into structured fields (rule-based + optional LLM)."""

    SECTION_PATTERNS = {
        "required_skills": [
            r"required\s+skills?",
            r"must[- ]have",
            r"mandatory\s+skills?",
            r"requirements?",
            r"technical\s+skills?",
        ],
        "preferred_skills": [
            r"preferred\s+skills?",
            r"nice[- ]to[- ]have",
            r"bonus\s+skills?",
            r"good\s+to\s+have",
        ],
        "responsibilities": [
            r"responsibilities",
            r"what\s+you.?ll\s+do",
            r"role\s+overview",
            r"key\s+duties",
        ],
        "education": [
            r"education",
            r"qualifications?",
            r"degree",
        ],
        "certifications": [
            r"certifications?",
            r"certificates?",
        ],
    }

    def process(self, raw_text: str, llm_structured: dict | None = None) -> JobDescription:
        text = normalize_text(raw_text)
        if not text:
            raise ValueError("Job description text is empty.")

        logger.info("Processing JD length=%s", len(text))
        notes: List[str] = []

        if llm_structured:
            jd = self._from_llm(raw_text, llm_structured, notes)
        else:
            jd = self._from_rules(raw_text, notes)

        if jd.min_experience_years is None:
            notes.append("No clearly stated experience requirement found.")
        if not jd.required_skills and not jd.preferred_skills:
            notes.append(
                "Skills sections were ambiguous; extracted skills may be incomplete."
            )
        jd.parsing_notes = unique_preserve_order(notes)
        return jd

    def _from_llm(
        self, raw_text: str, data: dict, notes: List[str]
    ) -> JobDescription:
        notes.append("JD structured with LLM assistance.")
        return JobDescription(
            title=data.get("title") or self._extract_title(raw_text),
            required_skills=unique_preserve_order(data.get("required_skills") or []),
            preferred_skills=unique_preserve_order(data.get("preferred_skills") or []),
            min_experience_years=data.get("min_experience_years"),
            education_requirements=unique_preserve_order(
                data.get("education_requirements") or []
            ),
            certifications=unique_preserve_order(data.get("certifications") or []),
            responsibilities=unique_preserve_order(data.get("responsibilities") or []),
            other_requirements=unique_preserve_order(
                data.get("other_requirements") or []
            ),
            raw_text=raw_text,
        )

    def _from_rules(self, raw_text: str, notes: List[str]) -> JobDescription:
        notes.append("JD structured with deterministic parser.")
        sections = self._split_sections(raw_text)
        required = self._parse_skill_bullets(
            sections.get("required_skills", "") or self._skills_fallback(raw_text)
        )
        preferred = self._parse_skill_bullets(sections.get("preferred_skills", ""))

        # Mixed skills: if only one skills blob, put into required and note ambiguity
        if required and not preferred and "preferred" not in raw_text.lower():
            notes.append(
                "Preferred vs required skills may be mixed; treating listed skills as required."
            )

        # Remove preferred items that duplicate required
        preferred = [s for s in preferred if s.lower() not in {r.lower() for r in required}]

        experience = extract_years(raw_text)
        education = self._parse_bullets(sections.get("education", ""))
        certifications = self._parse_bullets(sections.get("certifications", ""))
        responsibilities = self._parse_bullets(sections.get("responsibilities", ""))

        return JobDescription(
            title=self._extract_title(raw_text),
            required_skills=required,
            preferred_skills=preferred,
            min_experience_years=experience,
            education_requirements=education or [],
            certifications=certifications or [],
            responsibilities=responsibilities
            or self._default_responsibilities(raw_text),
            other_requirements=[],
            raw_text=raw_text,
        )

    def _extract_title(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return NOT_SPECIFIED
        first = lines[0]
        if len(first) < 120:
            # Strip common prefixes
            first = re.sub(
                r"^(job\s*title|position|role)\s*[:\-]\s*",
                "",
                first,
                flags=re.IGNORECASE,
            )
            return first.strip() or NOT_SPECIFIED
        return NOT_SPECIFIED

    def _split_sections(self, text: str) -> dict[str, str]:
        lines = text.splitlines()
        current = "preamble"
        buckets: dict[str, list[str]] = {current: []}
        for line in lines:
            matched_section = None
            for section, patterns in self.SECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(rf"^\s*{pattern}\b", line, flags=re.IGNORECASE):
                        matched_section = section
                        break
                if matched_section:
                    break
            if matched_section:
                current = matched_section
                buckets.setdefault(current, [])
                continue
            buckets.setdefault(current, []).append(line)
        return {k: "\n".join(v) for k, v in buckets.items()}

    def _parse_skill_bullets(self, section_text: str) -> List[str]:
        items = self._parse_bullets(section_text)
        skills: List[str] = []
        for item in items:
            # Split comma/semicolon-separated skill lines
            parts = re.split(r"[,;/|]", item)
            for part in parts:
                cleaned = part.strip(" -•*\t")
                cleaned = re.sub(
                    r"^(experience\s+with|proficient\s+in|knowledge\s+of)\s+",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                if 1 < len(cleaned) <= 60:
                    skills.append(cleaned)
        # Split "A or B" style phrases; normalize "unit testing with pytest" → pytest
        expanded: List[str] = []
        for skill in skills:
            if re.search(r"\bpytest\b", skill, flags=re.IGNORECASE):
                expanded.append("pytest")
            if re.search(r"\b(or|and)\b", skill, flags=re.IGNORECASE) and len(skill) < 80:
                parts = re.split(r"\s+(?:or|and)\s+", skill, flags=re.IGNORECASE)
                for part in parts:
                    part = re.split(r"\s+with\s+", part, maxsplit=1)[0]
                    part = re.sub(r"\s+or equivalent.*$", "", part, flags=re.IGNORECASE)
                    part = part.strip(" -")
                    if 1 < len(part) <= 60 and "equivalent" not in part.lower():
                        expanded.append(part)
            else:
                cleaned = re.sub(r"\s+or equivalent.*$", "", skill, flags=re.IGNORECASE)
                cleaned = cleaned.strip()
                if cleaned and "equivalent" not in cleaned.lower():
                    expanded.append(cleaned)
        return unique_preserve_order(expanded)

    def _parse_bullets(self, section_text: str) -> List[str]:
        if not section_text:
            return []
        items: List[str] = []
        for line in section_text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
            if line:
                items.append(line)
        return unique_preserve_order(items)

    def _skills_fallback(self, text: str) -> str:
        # Capture lines that look like tech stacks when no section headers exist
        known = [
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "node",
            "aws",
            "azure",
            "docker",
            "kubernetes",
            "sql",
            "fastapi",
            "django",
            "flask",
            "postgresql",
            "mongodb",
            "tensorflow",
            "pytorch",
            "llm",
            "rag",
        ]
        found = []
        lower = text.lower()
        for skill in known:
            if re.search(rf"\b{re.escape(skill)}\b", lower):
                found.append(skill)
        return "\n".join(found)

    def _default_responsibilities(self, text: str) -> List[str]:
        # Keep short excerpts that look like duty statements
        duties = []
        for line in text.splitlines():
            if re.search(r"\b(develop|design|build|own|collaborate|lead)\b", line, re.I):
                cleaned = line.strip(" -•*")
                if 20 < len(cleaned) < 200:
                    duties.append(cleaned)
        return unique_preserve_order(duties)[:8]
