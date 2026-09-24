# Week 2 Extension Points

The Week 1 app is the **system under test**. Week 2 should build evaluation around it—not rewrite it.

## Suggested Week 2 deliverables

1. **Eval harness** (`app/evaluation/runner.py`)  
   - Execute `baseline_scenarios.json` against live models  
   - Persist outputs + diffs per model/prompt version  

2. **Metrics**  
   - Skill faithfulness (precision/recall vs ground truth)  
   - Hallucination rate  
   - Injection resistance rate  
   - Score stability (repeat N runs)  
   - Schema validity rate  

3. **Prompt regression**  
   - Version prompts in `prompts/`  
   - Compare structured outputs to golden files  

4. **Security pack**  
   - Expand adversarial corpus  
   - Automate pass/fail on override attempts  

5. **CI gate**  
   - `pytest` + eval smoke on PR  
   - Fail if faithfulness drops below threshold  

6. **Observability**  
   - Structured logs with request IDs (no PII)  
   - Token/latency metrics  

7. **Toward Agentic QA**  
   - Agent proposes new edge resumes  
   - Agent runs screening API  
   - Agent files a markdown defect report when invariants break  

## Non-goals for Week 2

- Full ATS productization  
- Candidate PII storage  
- Fine-tuning  
