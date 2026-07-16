import json
import os
from pathlib import Path

from src.constants import DEFAULT_OUTPUT_DIR, LOG_DIR
from src.summarizer import (
    DEFAULT_GENERAL_PROMPT,
    DEFAULT_GENERAL_SYSTEM_MSG,
    DEFAULT_MEETING_PROMPT,
    DEFAULT_MEETING_SYSTEM_MSG,
)

SETTINGS_FILE = LOG_DIR / "settings.json"

DEFAULT_SETTINGS: dict = {
    "model": "large-v3",
    "device": "cuda",
    "llm_url": "https://openrouter.ai/api/v1",
    "llm_api_key": "",
    "llm_model": "openai/gpt-4o-mini",
    "diarize_enabled": False,
    "hf_token": "",
    "meeting_system_msg": DEFAULT_MEETING_SYSTEM_MSG,
    "meeting_prompt": DEFAULT_MEETING_PROMPT,
    "general_system_msg": DEFAULT_GENERAL_SYSTEM_MSG,
    "general_prompt": DEFAULT_GENERAL_PROMPT,
    "last_browser_dir": str(Path.home()),
    "output_dir": str(DEFAULT_OUTPUT_DIR),
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
            return _apply_env_defaults(data)
        except Exception:
            pass
    return _apply_env_defaults(DEFAULT_SETTINGS.copy())


def _apply_env_defaults(data: dict) -> dict:
    """Auto-populate API keys from environment variables when fields are empty."""
    if not data.get("llm_api_key"):
        data["llm_api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
    if not data.get("hf_token"):
        data["hf_token"] = os.environ.get("HF_TOKEN", "")
    return data


def save_settings(settings: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)
