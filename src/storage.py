from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Profile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PROFILE_PATH = DATA_DIR / "profile.json"
PROFILES_DIR = DATA_DIR / "profiles"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> Profile:
    ensure_data_dirs()
    with path.open("r", encoding="utf-8") as f:
        raw = f.read()
    return Profile.model_validate_json(raw)


def save_profile(profile: Profile, path: Optional[Path] = None) -> Path:
    ensure_data_dirs()
    target_path = path or DEFAULT_PROFILE_PATH
    with target_path.open("w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))
    return target_path


def save_profile_by_tag(profile: Profile) -> Path:
    """Persist a profile under data/profiles/<tag>.json."""
    ensure_data_dirs()
    target_path = PROFILES_DIR / f"{profile.tag}.json"
    return save_profile(profile, target_path)


def list_profiles() -> List[Dict[str, str]]:
    """Return minimal metadata for available profiles."""
    ensure_data_dirs()
    profiles: List[Dict[str, str]] = []
    if DEFAULT_PROFILE_PATH.exists():
        try:
            profile = load_profile(DEFAULT_PROFILE_PATH)
            profiles.append({"tag": profile.tag, "path": str(DEFAULT_PROFILE_PATH)})
        except Exception:
            profiles.append({"tag": "default_invalid", "path": str(DEFAULT_PROFILE_PATH)})

    for path in PROFILES_DIR.glob("*.json"):
        try:
            profile = load_profile(path)
            profiles.append({"tag": profile.tag, "path": str(path)})
        except Exception:
            profiles.append({"tag": path.stem, "path": str(path)})
    return profiles


def load_profile_by_tag(tag: str) -> Optional[Profile]:
    ensure_data_dirs()
    candidates = [DEFAULT_PROFILE_PATH] + list(PROFILES_DIR.glob("*.json"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            profile = load_profile(path)
        except Exception:
            continue
        if profile.tag == tag:
            return profile
    return None
