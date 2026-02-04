# job_application_agent

FastAPI service plus Tailwind UI for the "Job Applicant" agent. Current features:
- Structured profile JSON (truth-locked), loaded from `data/profile.json`.
- Resume ingestion endpoint to create a tagged profile from an uploaded PDF.
- Sample job objects + matcher (hard filters + weighted subscores).
- Tailwind UI at `/web` to upload a resume, pick targets, and view sample matches.

## Run locally
```bash
pip install -r requirements.txt
uvicorn src.app:app --reload --port 8000
```
Then open http://localhost:8000/web for the UI.
