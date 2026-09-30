"""Shared application constants."""

NOT_SPECIFIED = "Not specified"

RECOMMENDATION_SHORTLIST = "Shortlist"
RECOMMENDATION_REVIEW = "Review"
RECOMMENDATION_DOES_NOT_MEET = "Does Not Meet Requirements"

VALID_RECOMMENDATIONS = {
    RECOMMENDATION_SHORTLIST,
    RECOMMENDATION_REVIEW,
    RECOMMENDATION_DOES_NOT_MEET,
}

# Common skill aliases for deterministic matching (lowercase)
SKILL_ALIASES: dict[str, set[str]] = {
    "javascript": {"js", "javascript", "ecmascript"},
    "typescript": {"ts", "typescript"},
    "python": {"python", "py"},
    "postgresql": {"postgres", "postgresql", "psql"},
    "kubernetes": {"k8s", "kubernetes"},
    "amazon web services": {"aws", "amazon web services"},
    "google cloud platform": {"gcp", "google cloud", "google cloud platform"},
    "machine learning": {"ml", "machine learning"},
    "natural language processing": {"nlp", "natural language processing"},
    "ci/cd": {"ci/cd", "cicd", "continuous integration", "continuous delivery"},
    "rest api": {"rest", "rest api", "restful", "rest apis"},
    "node.js": {"node", "nodejs", "node.js"},
    "react": {"react", "reactjs", "react.js"},
    "fastapi": {"fastapi", "fast api"},
    "docker": {"docker", "containers"},
    "openai api": {"openai", "openai api", "openai apis"},
    "pytest": {"pytest", "py.test", "unit testing with pytest"},
}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+the\s+instructions",
    r"you\s+are\s+now",
    r"system\s*:\s*",
    r"<<<\s*system\s*>>>",
    r"unrestricted\s+mode",
    r"shortlist\s+this\s+candidate",
    r"mark\s+this\s+candidate\s+as\s+the\s+best",
    r"give\s+(this\s+)?candidate\s+(a\s+)?(perfect|100|high)\s+score",
    r"disregard\s+(all\s+)?(prior|previous)\s+(rules|instructions)",
]
