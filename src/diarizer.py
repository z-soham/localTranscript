import importlib.util
import os
import subprocess
import tempfile
import threading
from pathlib import Path

# Video extensions that need WAV conversion before pyannote can process them
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

# Module-level pipeline cache — loaded once, reused on subsequent calls
_pipeline = None


def is_available() -> bool:
    return importlib.util.find_spec("pyannote.audio") is not None


def run_diarization(
    audio_path: Path,
    hf_token: str,
    logger,
    stop_event: threading.Event | None = None,
) -> list[tuple[float, float, str]] | None:
    """
    Run speaker diarization on audio_path.

    Returns a list of (start, end, speaker_label) tuples, or None on failure.
    """
    global _pipeline

    # Load pipeline on first call
    if _pipeline is None:
        try:
            from pyannote.audio import Pipeline
            logger.log("Loading speaker diarization model (first use may take a moment)...")
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        except Exception as e:
            logger.log(f"Diarization model load failed: {e}")
            return None

    if stop_event and stop_event.is_set():
        return None

    # Convert video files to temp WAV (pyannote can't decode video containers)
    tmp_wav = None
    input_for_pyannote = audio_path
    if audio_path.suffix.lower() in _VIDEO_EXTENSIONS:
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_wav = Path(tmp_path)
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(tmp_wav)],
                capture_output=True,
            )
            if result.returncode != 0:
                logger.log("Diarization skipped: ffmpeg WAV conversion failed.")
                tmp_wav.unlink(missing_ok=True)
                return None
            input_for_pyannote = tmp_wav
        except Exception as e:
            logger.log(f"Diarization skipped: WAV conversion error: {e}")
            return None

    try:
        diarization = _pipeline(str(input_for_pyannote))
        return [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as e:
        logger.log(f"Diarization failed: {e}")
        return None
    finally:
        if tmp_wav:
            tmp_wav.unlink(missing_ok=True)
