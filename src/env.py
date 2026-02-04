from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load() -> None:
    """
    Load environment variables from .env (project root) and process env.
    """
    # Load from current working directory if present
    load_dotenv()
    # Explicitly load the repo-level .env to be safe when working from subdirs
    repo_dotenv = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(repo_dotenv, override=False)


# Auto-load on import so any module that imports env gets .env populated.
load()
