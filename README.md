# job_application_agent

FastAPI service plus Tailwind UI for the "Job Applicant" agent. Current features:
- Structured profile JSON (truth-locked), loaded from `data/profile.json`.
- Resume ingestion endpoint to create a tagged profile from an uploaded PDF.
- Job ingestion with canonical URL resolution (prefers direct posting links over aggregator pages).
- Explainable scoring with component breakdown, keyword coverage, and missing keyword gaps.
- Resume tailoring with highlighted preview plus downloadable TXT/HTML artifacts.
- Auto-apply workflow with browser automation for common ATS forms (Playwright based).
- Tailwind UI at `/web` to upload a resume, search jobs, inspect score rationale, tailor resumes, and trigger auto-apply.

## Run locally
```bash
pip install -r requirements.txt
python -m playwright install chromium
uvicorn src.app:app --reload --port 8000
```
Then open http://localhost:8000/web for the UI.
