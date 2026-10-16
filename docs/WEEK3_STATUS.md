# Week 3 Manager Status Summary

**Week:** 3 — RAG Knowledge Assistant + Retrieval/Generation Evaluation  
**Status:** Complete (runnable POC, offline by default)

## Delivered
- RAG policy assistant (ingest → chunk → embed → retrieve → answer + citations)
- Synthetic HR/policy corpus + conflict + injection documents
- 50+ RAG evaluation cases across direct/multi-doc/unavailable/adversarial categories
- Retrieval + generation metrics with offline RAGAS-style proxies
- Chunking & Top-K experiments
- Streamlit UI + API endpoints
- Security tests for indirect prompt injection
- Integrated with Week 2 evaluation result/report framework

## Demo path
1. Build KB in RAG UI  
2. Ask leave / travel / unanswerable / conflict questions  
3. Run `scripts/run_rag_evaluation.py`  
4. Show retrieval vs generation failures and experiment summary  

## Next (Week 4)
Observability depth, advanced regression, CI evaluation gates, broader LLM QA framework.
