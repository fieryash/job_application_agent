# job_application_agent

FastAPI service plus Tailwind UI for the “Job Applicant Ashtik” agent. Current features:
- Structured profile JSON (truth-locked), loaded from `data/profile.json`.
- Resume ingestion endpoint to create a tagged profile from an uploaded PDF.
- Sample job objects + matcher (hard filters + weighted subscores).
- Job ingestion to DB (SQLite) via Tavily + OpenAI parser, with sample fallback.
- Tailor endpoint with guardrails (no new numbers/tools) + validation report.
- Tailwind UI at `/web` to upload a resume, pick targets (persisted), view jobs, score matches, and tailor bullets. If `API_KEY` is set, UI requests must include `X-API-Key` (there’s an API key input in the UI).
- Search + ingest real jobs via Tavily (with ATS fetchers for Greenhouse/Lever/Ashby) and store in SQLite.
- API key auth and simple rate limiting (per key/IP per minute).

## Run locally
```bash
pip install -r requirements.txt
uvicorn src.app:app --reload --port 8000
```
Then open http://localhost:8000/web for the UI.

## Environment
- `OPENAI_API_KEY` (required for parsing/tailoring)
- `TAVILY_API_KEY` (optional; job search; falls back to sample jobs)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `API_KEY` (optional; if set, clients must send `X-API-Key`)
- `RATE_LIMIT_PER_MIN` (optional; default 60)
- `.env` is loaded automatically (via `python-dotenv`); copy `.env.example` to `.env` and set your keys before starting the server.

## API highlights
- `GET /profiles` — list profiles; `POST /profiles/from-resume` to ingest a PDF (returns tagged profile, stored under `data/profiles/`).
- `POST /profiles/{tag}/targets` — update targets and persist.
- `POST /jobs/ingest` - body `{queries[], urls[], limit, use_sample, profile_tag}`; uses Tavily + ATS fetchers and stores parsed jobs in SQLite `data/jobs.db`.
- `GET /jobs` - list stored jobs (fallback to `data/sample_jobs.json`).
- `POST /match?profile_tag=...` - score a job payload against a profile.
- `POST /match/batch?profile_tag=...` - score multiple job payloads in one call (UI uses this to avoid request fan-out/rate limits).
- `POST /tailor?profile_tag=...` - tailor bullets for a job with validation.
- `GET /tailor/{profile_tag}/{job_id}` - fetch stored tailored resume (or compute/store if missing).
