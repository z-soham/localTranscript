import math
import shutil
import subprocess
from pathlib import Path


def format_timestamp(seconds: float, always_include_hours: bool = True) -> str:
    if seconds < 0:
        seconds = 0

    total_seconds = int(seconds)
    milliseconds = int(round((seconds - total_seconds) * 1000))

    if milliseconds == 1000:
        total_seconds += 1
        milliseconds = 0

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if always_include_hours or hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
    return f"{minutes:02}:{secs:02},{milliseconds:03}"


def seconds_to_human(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h:d}h {m:02}m {s:02}s"
    if m > 0:
        return f"{m:d}m {s:02}s"
    return f"{s:d}s"


def get_media_duration_seconds(input_file: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_file),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None


def write_txt(output_path: Path, segments) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            text = segment.text.strip()
            if text:
                f.write(text + "\n")


def write_srt(output_path: Path, segments) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
