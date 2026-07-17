import collections
import importlib.util
import math
import threading
import time
import wave
from pathlib import Path

from src.constants import PARAKEET_MODEL_ID
from src.logging_setup import QueueLogger
from src.utils import convert_to_wav16k_mono

CHUNK_SEC = 30

Segment = collections.namedtuple("Segment", "start end text")

# onnxruntime never raises when a requested execution provider is unavailable —
# it just warns and silently falls back — so CUDA availability must be checked
# upfront via get_available_providers() rather than caught with try/except.
_model_cache: dict[str, object] = {}


def is_available() -> bool:
    return importlib.util.find_spec("onnx_asr") is not None


def _cuda_available() -> bool:
    import onnxruntime
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def load_parakeet_model(prefer_cuda: bool, logger: QueueLogger):
    want_cuda = prefer_cuda and _cuda_available()
    device_used = "cuda" if want_cuda else "cpu"

    if device_used in _model_cache:
        logger.log(f"Reusing cached Parakeet model on {device_used.upper()}.")
        return _model_cache[device_used], device_used

    import onnx_asr

    logger.log(f"Loading Parakeet model on {device_used.upper()}...")
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if want_cuda else ["CPUExecutionProvider"]
        model = onnx_asr.load_model(PARAKEET_MODEL_ID, providers=providers)
    except Exception as e:
        if not want_cuda:
            raise
        logger.log(f"CUDA load failed: {e}. Falling back to CPU...")
        device_used = "cpu"
        model = onnx_asr.load_model(PARAKEET_MODEL_ID, providers=["CPUExecutionProvider"])

    _model_cache[device_used] = model
    return model, device_used


def transcribe_with_parakeet(
    input_path: Path,
    prefer_cuda: bool,
    logger: QueueLogger,
    stop_event: "threading.Event | None",
    duration_seconds: float | None,
) -> "list[Segment] | None":
    import numpy as np

    model, device_used = load_parakeet_model(prefer_cuda, logger)
    logger.log(f"Device in use: {device_used}")

    wav_path = convert_to_wav16k_mono(input_path)
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        wav_path.unlink(missing_ok=True)

    total_sec = len(audio) / sample_rate
    duration_seconds = duration_seconds or total_sec

    segments = []
    start_time = time.time()

    for chunk_start, chunk_end in _chunk_bounds(total_sec):
        if stop_event is not None and stop_event.is_set():
            return None

        chunk_audio = audio[int(chunk_start * sample_rate):int(chunk_end * sample_rate)]
        text = model.recognize(chunk_audio, sample_rate=sample_rate)
        segments.append(Segment(chunk_start, chunk_end, text))

        elapsed = time.time() - start_time
        progress = max(0.0, min(1.0, chunk_end / duration_seconds)) if duration_seconds else 0.0
        speed_x = (chunk_end / elapsed) if elapsed > 0 else None
        eta = ((duration_seconds - chunk_end) / speed_x) if speed_x and duration_seconds else None
        logger.progress(progress, chunk_end, duration_seconds, elapsed, eta, speed_x)

    return segments


def _chunk_bounds(total_sec: float) -> "list[tuple[float, float]]":
    n_chunks = max(1, math.ceil(total_sec / CHUNK_SEC))
    return [(i * CHUNK_SEC, min((i + 1) * CHUNK_SEC, total_sec)) for i in range(n_chunks)]


if __name__ == "__main__":
    assert _chunk_bounds(0) == [(0, 0)]
    assert _chunk_bounds(30) == [(0, 30)]
    assert _chunk_bounds(45) == [(0, 30), (30, 45)]
    assert _chunk_bounds(90) == [(0, 30), (30, 60), (60, 90)]
    print("parakeet chunk-bounds self-check passed")
