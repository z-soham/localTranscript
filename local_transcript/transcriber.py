import threading
import time
import traceback
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from local_transcript.constants import SUPPORTED_EXTENSIONS
from local_transcript.cuda import locate_cudnn_hint, preload_cuda_paths, should_force_cpu_after_cuda_error
from local_transcript.logging_setup import QueueLogger, TranscriptionError
from local_transcript.utils import get_media_duration_seconds, seconds_to_human, write_srt, write_txt


def transcribe_file(
    input_path: Path,
    model_name: str,
    prefer_cuda: bool,
    logger: QueueLogger,
    stop_event: threading.Event | None = None,
) -> None:
    if WhisperModel is None:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.log(f"Warning: Unexpected file extension '{input_path.suffix}'. Attempting anyway.")

    duration_seconds = get_media_duration_seconds(input_path)
    if duration_seconds is None:
        logger.log("ffprobe not found or duration could not be detected. ETA will be approximate.")
    else:
        logger.log(f"Media duration: {seconds_to_human(duration_seconds)}")

    model = None
    device_used = None

    if prefer_cuda:
        preload_cuda_paths(logger)
        try:
            logger.log(f"Loading model '{model_name}' on CUDA...")
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
            device_used = "cuda"
        except BaseException as e:
            error_text = str(e)
            logger.log(f"CUDA load failed: {error_text}")
            cudnn_hint = locate_cudnn_hint()
            if "cudnn_ops64_9.dll" in error_text or "Cannot load symbol cudnn" in error_text:
                logger.log("cuDNN 9 appears to be missing from PATH for the current environment.")
                if cudnn_hint:
                    logger.log(f"Possible cuDNN bin directory detected at: {cudnn_hint}")
                    logger.log(
                        "The app will continue on CPU. To restore GPU support, "
                        "add that directory to PATH before launching Python."
                    )
            if should_force_cpu_after_cuda_error(error_text):
                logger.log("Recoverable CUDA startup failure detected. Disabling GPU for this run and falling back to CPU.")
            else:
                logger.log("Unexpected CUDA startup failure detected. The app will still attempt CPU fallback.")
            model = None
            device_used = None
            logger.log("Falling back to CPU...")

    if model is None:
        try:
            logger.log(f"Loading model '{model_name}' on CPU...")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            device_used = "cpu"
        except Exception as e:
            raise TranscriptionError(f"CPU transcription engine could not be initialized: {e}") from e

    logger.log(f"Device in use: {device_used}")
    logger.log(f"Transcribing: {input_path}")

    segments_generator, info = model.transcribe(
        str(input_path),
        beam_size=5,
        vad_filter=True,
        language="en",
        condition_on_previous_text=True,
    )

    segments = []
    start_time = time.time()
    last_end = 0.0

    cancelled = False
    for segment in segments_generator:
        if stop_event is not None and stop_event.is_set():
            cancelled = True
            break

        segments.append(segment)
        last_end = max(last_end, float(segment.end))
        elapsed = time.time() - start_time

        if duration_seconds and duration_seconds > 0:
            progress = max(0.0, min(1.0, last_end / duration_seconds))
            speed_x = (last_end / elapsed) if elapsed > 0 else None
            remaining_audio = max(0.0, duration_seconds - last_end)
            eta = (remaining_audio / speed_x) if speed_x and speed_x > 0 else None
            logger.progress(progress, last_end, duration_seconds, elapsed, eta, speed_x)
        else:
            logger.progress(0.0, last_end, None, elapsed, None, None)

    if cancelled:
        logger.log("Transcription cancelled by user.")
        logger.done(False, "cancelled")
        return

    total_wall = time.time() - start_time
    logger.log("")
    logger.log(f"Detected language: {info.language} (probability: {info.language_probability:.3f})")
    logger.log(f"Number of segments: {len(segments)}")
    logger.log(f"Total transcription time: {seconds_to_human(total_wall)}")
    if duration_seconds and total_wall > 0:
        logger.log(f"Overall speed: {duration_seconds / total_wall:.2f}x real-time")

    base_output = input_path.with_suffix("")
    txt_output = base_output.with_name(base_output.name + "_transcript.txt")
    srt_output = base_output.with_name(base_output.name + "_subtitles.srt")

    write_txt(txt_output, segments)
    write_srt(srt_output, segments)

    logger.log(f"Transcript written to: {txt_output}")
    logger.log(f"Subtitles written to:  {srt_output}")
    logger.done(True, "Transcription completed successfully.")
