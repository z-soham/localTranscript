import json
from pathlib import Path

from src.constants import LOG_DIR

SETTINGS_FILE = LOG_DIR / "settings.json"

DEFAULT_SETTINGS: dict = {
    "model": "large-v3",
    "device": "cuda",
    "llm_url": "https://openrouter.ai/api/v1",
    "llm_api_key": "",
    "llm_model": "openai/gpt-4o-mini",
}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Back-fill any keys added after the file was first written
            for key, default in DEFAULT_SETTINGS.items():
                if key not in data:
                    data[key] = default
            return data
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)
