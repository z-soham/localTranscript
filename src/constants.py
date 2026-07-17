from pathlib import Path

APP_TITLE = "Local Transcript Studio"
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".flac"
}
MODEL_OPTIONS = ["medium", "large-v3"]
ENGINE_OPTIONS = ["faster-whisper", "parakeet"]
PARAKEET_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"
LOG_DIR = Path.home() / "LocalTranscriptLogs"
YT_DOWNLOAD_DIR = Path.home() / "LocalTranscriptLogs" / "yt_downloads"
DEFAULT_OUTPUT_DIR = Path.home() / "LocalTranscriptLogs" / "transcripts"
