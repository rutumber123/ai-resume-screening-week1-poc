"""Resume processing and candidate information extraction."""

from __future__ import annotations

import re
from typing import List, Optional

from app.core.constants import NOT_SPECIFIED
from app.core.logging import get_logger
from app.models.schemas import CandidateProfile
from app.utils.helpers import extract_years, unique_preserve_order

logger = get_logger(__name__)

TECH_CATALOG = {
    "programming_languages": [
        "python",
        "java",
        "javascript",
        "typescript",
        "go",
        "golang",
        "c#",
        "c++",
        "ruby",
        "rust",
        "kotlin",
        "scala",
        "r",
        "sql",
    ],
    "frameworks": [
        "react",
        "angular",
        "vue",
        "django",
        "flask",
        "fastapi",
        "spring",
        "spring boot",
        "express",
        "next.js",
        "nestjs",
        "dotnet",
        ".net",
        "langchain",
        "llamaindex",
    ],
    "cloud_technologies": [
        "aws",
        "azure",
        "gcp",
        "google cloud",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "github actions",
        "ci/cd",
    ],
    "databases": [
        "postgresql",
        "postgres",
        "mysql",
        "mongodb",
        "redis",
        "dynamodb",
        "elasticsearch",
        "snowflake",
        "bigquery",
    ],
}


class ResumeProcessor:
    """Extract structured candidate information without inventing facts."""

    def process(
        self,
        raw_text: str,
        filename: str = "",
        extraction_warnings: Optional[List[str]] = None,
        llm_structured: dict | None = None,
    ) -> CandidateProfile:
        warnings = list(extraction_warnings or [])
        text = (raw_text or "").strip()
        logger.info(
            "Processing resume filename_hash_len=%s text_len=%s",
            len(filename),
            len(text),
        )

        if not text:
            warnings.append("Resume text is empty; all fields marked Not specified.")
            return CandidateProfile(
                name=NOT_SPECIFIED,
                raw_text="",
                source_filename=filename,
                extraction_warnings=warnings,
            )

        if llm_structured:
            profile = self._from_llm(text, filename, llm_structured, warnings)
        else:
            profile = self._from_rules(text, filename, warnings)

        # Never invent: blank name stays Not specified
        if not profile.name or not profile.name.strip():
            profile.name = NOT_SPECIFIED
        profile.extraction_warnings = unique_preserve_order(warnings)
        return profile

    def _from_llm(
        self,
        text: str,
        filename: str,
        data: dict,
        warnings: List[str],
    ) -> CandidateProfile:
        warnings.append("Candidate profile extracted with LLM assistance.")
        # Guard: only keep skills that appear in resume text (anti-hallucination)
        allowed = self._skills_mentioned_in_text(text)
        skills = [
            s
            for s in unique_preserve_order(data.get("skills") or [])
            if self._skill_supported(s, text, allowed)
        ]
        langs = [
            s
            for s in unique_preserve_order(data.get("programming_languages") or [])
            if self._skill_supported(s, text, allowed)
        ]
        frameworks = [
            s
            for s in unique_preserve_order(data.get("frameworks") or [])
            if self._skill_supported(s, text, allowed)
        ]
        cloud = [
            s
            for s in unique_preserve_order(data.get("cloud_technologies") or [])
            if self._skill_supported(s, text, allowed)
        ]
        databases = [
            s
            for s in unique_preserve_order(data.get("databases") or [])
            if self._skill_supported(s, text, allowed)
        ]

        dropped = (
            len(data.get("skills") or [])
            + len(data.get("programming_languages") or [])
            + len(data.get("frameworks") or [])
            - len(skills)
            - len(langs)
            - len(frameworks)
        )
        if dropped > 0:
            warnings.append(
                f"Removed {dropped} LLM-suggested skill(s) not evidenced in resume text."
            )

        return CandidateProfile(
            name=data.get("name") or self._extract_name(text),
            total_experience_years=data.get("total_experience_years")
            if data.get("total_experience_years") is not None
            else extract_years(text),
            skills=skills,
            programming_languages=langs,
            frameworks=frameworks,
            cloud_technologies=cloud,
            databases=databases,
            certifications=unique_preserve_order(data.get("certifications") or []),
            education=unique_preserve_order(data.get("education") or []),
            previous_roles=unique_preserve_order(data.get("previous_roles") or []),
            projects=unique_preserve_order(data.get("projects") or []),
            domain_experience=unique_preserve_order(data.get("domain_experience") or []),
            raw_text=text,
            source_filename=filename,
        )

    def _from_rules(
        self, text: str, filename: str, warnings: List[str]
    ) -> CandidateProfile:
        warnings.append("Candidate profile extracted with deterministic parser.")
        if not re.search(r"skills?", text, re.I):
            warnings.append("No explicit skills section detected.")

        found = {k: [] for k in TECH_CATALOG}
        lower = text.lower()
        for category, skills in TECH_CATALOG.items():
            for skill in skills:
                if re.search(rf"\b{re.escape(skill)}\b", lower):
                    found[category].append(skill)

        # Generic skills line extraction
        skills_section = self._extract_section(text, r"skills?")
        extra_skills = self._split_skills(skills_section) if skills_section else []

        experience = extract_years(text)
        if experience is None:
            warnings.append("Total experience years not specified.")

        education = self._extract_bullets_section(text, r"education")
        certifications = self._extract_bullets_section(text, r"certifications?")
        roles = self._extract_bullets_section(text, r"(experience|employment|work\s+history)")
        projects = self._extract_bullets_section(text, r"projects?")

        return CandidateProfile(
            name=self._extract_name(text),
            total_experience_years=experience,
            skills=unique_preserve_order(extra_skills),
            programming_languages=unique_preserve_order(found["programming_languages"]),
            frameworks=unique_preserve_order(found["frameworks"]),
            cloud_technologies=unique_preserve_order(found["cloud_technologies"]),
            databases=unique_preserve_order(found["databases"]),
            certifications=certifications,
            education=education,
            previous_roles=roles[:10],
            projects=projects[:10],
            domain_experience=[],
            raw_text=text,
            source_filename=filename,
        )

    def _extract_name(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return NOT_SPECIFIED
        candidate = lines[0]
        # Skip obvious headers
        if re.search(r"resume|curriculum vitae|\bcv\b", candidate, re.I):
            candidate = lines[1] if len(lines) > 1 else ""
        if not candidate or len(candidate) > 80:
            return NOT_SPECIFIED
        if re.search(r"@|http|linkedin|github", candidate, re.I):
            return NOT_SPECIFIED
        # Reject injection-looking first lines
        if re.search(r"ignore|instruction|system\s*:", candidate, re.I):
            return NOT_SPECIFIED
        return candidate

    def _extract_section(self, text: str, header_pattern: str) -> str:
        pattern = rf"(?im)^\s*{header_pattern}\s*:?\s*$"
        match = re.search(pattern, text)
        if not match:
            # Inline "Skills: a, b, c"
            inline = re.search(
                rf"(?im)^\s*{header_pattern}\s*:\s*(.+)$", text
            )
            return inline.group(1).strip() if inline else ""
        start = match.end()
        rest = text[start:]
        next_header = re.search(
            r"(?im)^\s*(experience|education|projects?|certifications?|summary|work)\b",
            rest,
        )
        end = next_header.start() if next_header else len(rest)
        return rest[:end].strip()

    def _extract_bullets_section(self, text: str, header_pattern: str) -> List[str]:
        section = self._extract_section(text, header_pattern)
        if not section:
            return []
        items = []
        for line in section.splitlines():
            line = re.sub(r"^[-*•\d.)\s]+", "", line.strip())
            if line:
                items.append(line)
        return unique_preserve_order(items)

    def _split_skills(self, blob: str) -> List[str]:
        parts = re.split(r"[,;/|•\n]", blob)
        return unique_preserve_order(
            [p.strip(" -*\t") for p in parts if 1 < len(p.strip(" -*\t")) <= 40]
        )

    def _skills_mentioned_in_text(self, text: str) -> set[str]:
        lower = text.lower()
        mentioned = set()
        for skills in TECH_CATALOG.values():
            for skill in skills:
                if re.search(rf"\b{re.escape(skill)}\b", lower):
                    mentioned.add(skill.lower())
        return mentioned

    def _skill_supported(self, skill: str, text: str, allowed: set[str]) -> bool:
        s = skill.lower().strip()
        if s in allowed:
            return True
        # Require literal presence for non-catalog skills
        return bool(re.search(rf"\b{re.escape(s)}\b", text, flags=re.IGNORECASE))
