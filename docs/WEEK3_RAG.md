# Week 3 — RAG Knowledge Assistant & Evaluation

## Principle

Week 1: build an AI app.  
Week 2: systematically test LLM apps.  
Week 3: systematically test **retrieval + generation** in a RAG app.

## Application

**Contoso Synthetic Knowledge / HR Policy Assistant**

- Ingest sample policies under `sample_documents/`
- Chunk → TF-IDF embeddings → in-memory vector store
- Retrieve Top-K → generate grounded answer + citations
- Decline when documents lack the answer
- Detect leave carry-forward conflicts (10 vs 15 days)
- Treat retrieved injection text as data

### Run UI

```powershell
.\.venv\Scripts\streamlit.exe run app\ui\rag_streamlit_app.py
```

### API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --port 8000
```

- `POST /api/v1/rag/build`
- `POST /api/v1/rag/ask` JSON `{"question":"...","top_k":5}`

## Evaluation

```powershell
.\.venv\Scripts\python.exe scripts\generate_rag_catalog.py
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py
.\.venv\Scripts\python.exe scripts\run_rag_experiments.py
```

### Metrics (offline RAGAS proxies by default)

| Metric | Meaning | Limitation |
|--------|---------|------------|
| Context precision | Share of retrieved chunks matching expected keywords | Keywords ≠ true relevance |
| Context recall | Expected doc/keywords covered by retrieval | Misses semantic paraphrase |
| Faithfulness | Answer tokens supported by retrieved text | Lexical heuristic |
| Answer relevancy | Question/answer token overlap | Weak for paraphrase |
| Citation correctness | Citations map to retrieved chunk IDs | Does not prove quote sufficiency |
| IDK correctness | Unanswerable questions declined | Depends on topic heuristics |

Official RAGAS LLM-judge metrics are optional via `evaluation/rag/ragas_bridge.py` when `ragas` + API key are available.

## Experiments

`scripts/run_rag_experiments.py` compares chunking profiles `A_small` / `B_medium` / `C_large` and Top-K `3/5/8`, writing `experiments/rag/chunking_topk_summary.json`.

## Framework integration

```text
Evaluation Framework (Week 2)
        ├── LLM ScreeningEvaluationRunner
        └── RAGEvaluationRunner  (Week 3)
```

Shared: result storage, markdown reports, dimension scores, failure taxonomy, observability spans.
