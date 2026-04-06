"""
src/youtube.py

Downloads audio from a YouTube URL using yt-dlp and returns a Path to
the resulting .mp3 file. Runs inside the existing worker thread — all
output is routed through QueueLogger to the GUI's queue.
"""
import threading
from pathlib import Path

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    yt_dlp = None  # type: ignore[assignment]
    YT_DLP_AVAILABLE = False

from src.constants import YT_DOWNLOAD_DIR
from src.logging_setup import QueueLogger, TranscriptionError


def download_youtube_audio(
    url: str,
    logger: QueueLogger,
    stop_event: threading.Event,
) -> Path:
    """
    Download audio from `url` to YT_DOWNLOAD_DIR as an .mp3 file.

    Returns the Path of the downloaded file.
    Raises TranscriptionError on failure or if yt-dlp is not installed.
    Checks stop_event before starting the network request.
    """
    if not YT_DLP_AVAILABLE:
        raise TranscriptionError("yt-dlp is not installed. Run: pip install yt-dlp")

    if stop_event.is_set():
        raise TranscriptionError("Download cancelled before it started.")

    YT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    class _Capture:
        path: Path | None = None

    capture = _Capture()

    def _post_hook(filepath: str) -> None:
        capture.path = Path(filepath)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(YT_DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "progress_hooks": [_make_progress_hook(logger, stop_event)],
        "post_hooks": [_post_hook],
    }

    logger.log(f"Downloading audio from: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise TranscriptionError(f"yt-dlp download failed: {exc}") from exc
    except Exception as exc:
        if stop_event.is_set():
            raise TranscriptionError("Download was cancelled.") from exc
        raise TranscriptionError(f"Unexpected download error: {exc}") from exc

    if capture.path is None or not capture.path.exists():
        candidates = sorted(
            YT_DOWNLOAD_DIR.glob("*.mp3"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise TranscriptionError(
                "yt-dlp produced no output file. Ensure FFmpeg is installed and on PATH."
            )
        capture.path = candidates[0]

    logger.log(f"Download complete: {capture.path.name}")
    return capture.path


def _make_progress_hook(logger: QueueLogger, stop_event: threading.Event):
    def _hook(d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            speed = d.get("speed")
            if total and total > 0:
                speed_str = f"{speed / 1024:.0f} KB/s" if speed else "--"
                logger.log(f"  Downloading... {downloaded / total * 100:.1f}%  ({speed_str})")
            else:
                logger.log(f"  Downloading... {downloaded / 1024 / 1024:.1f} MB received")
            if stop_event.is_set():
                raise Exception("Download interrupted by stop event.")
        elif status == "finished":
            logger.log("  Download finished, converting to MP3...")

    return _hook
