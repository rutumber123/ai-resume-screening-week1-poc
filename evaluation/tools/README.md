# PromptFoo & DeepEval — with and without commercial API keys

Primary Week 2 suite (always offline):

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

---

## PromptFoo **without** API keys

Supported here via:
- **Custom/mock provider** (`promptfoo_mock_provider.py`) — local Python, no cloud
- **Deterministic assertions only** — `icontains`, `not-icontains`, `javascript`  
  (no `llm-rubric` / judge LLM)

### Run

```powershell
cd "C:\Users\rutumber.nath\Documents\Week 1 POC"
npx --yes promptfoo@latest eval -c evaluation/tools/promptfooconfig.yaml
```

### Verify
- Table shows pass/fail for v1/v2/v3 × 3 tests
- Exit code `0`
- Optional UI: `npx --yes promptfoo@latest view`

### Limitations
Features that need a judge model (`llm-rubric`, automated red-team generation) will **not** work in this offline setup unless you point PromptFoo at **Ollama** or another local endpoint.

Optional local model provider example (if you install Ollama later):

```yaml
providers:
  - id: ollama:llama3.2
```

---

## DeepEval **without** commercial API keys

Built-in metrics (GEval, Faithfulness, Hallucination, RAG triad) normally need an LLM judge.

### Mode A — Offline deterministic (default, no model at all)

```powershell
.\.venv\Scripts\python.exe -m evaluation.tools.deepeval_bridge --mode offline
```

**Verify:** `"status": "passed"` and both:
- `positive_case.status = passed` (grounded claims)
- `negative_case.status = failed` (hallucinated AWS/Kubernetes)

This uses a local groundedness proxy, **not** DeepEval’s GEval judge.

### Mode B — Local Ollama (real DeepEval metric, no OpenAI key)

1. Install [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.2`
2. Install DeepEval: `pip install deepeval`
3. Run:

```powershell
.\.venv\Scripts\python.exe -m pip install "deepeval>=1.0.0"
$env:OLLAMA_MODEL = "llama3.2"
.\.venv\Scripts\python.exe -m evaluation.tools.deepeval_bridge --mode ollama
```

**Verify:** `"status": "passed"`, `"mode": "ollama"`.

### Mode C — Commercial OpenAI (optional)

```powershell
$env:OPENAI_API_KEY = "sk-..."
.\.venv\Scripts\python.exe -m evaluation.tools.deepeval_bridge --mode openai
```

---

## Quick matrix

| Tool | Offline / no cloud key | Local Ollama | OpenAI key |
|------|------------------------|--------------|------------|
| Custom eval runner | Yes (primary) | N/A | Optional `--use-llm` |
| PromptFoo | Yes (mock provider + deterministic asserts) | Optional | Optional |
| DeepEval bridge | Yes (`--mode offline`) | Yes (`--mode ollama`) | Yes (`--mode openai`) |
