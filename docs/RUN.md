# Quick Run Instructions

## 1. Install (once)

```powershell
cd "c:\Users\rutumber.nath\Documents\Week 1 POC"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## 2. Demo UI

```powershell
streamlit run app/ui/streamlit_app.py
```

Upload resumes from `sample_data\resumes\` (try `01`, `02`, `03`, `18`).

## 3. API

```powershell
uvicorn app.api.main:app --reload --port 8000
```

Open http://localhost:8000/docs

## 4. Tests

```powershell
pytest -q
```

## 5. CLI demo

```powershell
python scripts\run_demo.py
python scripts\run_baseline_eval.py
```

Default `LLM_PROVIDER=mock` needs no API key.
