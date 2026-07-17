# AGENTS.md

## Run

```bash
python main.py          # GUI entry (sets KMP_DUPLICATE_LIB_OK + faulthandler crash logging)
```

No test, lint, or typecheck commands exist.

## Dependencies

```bash
pip install .           # core: faster-whisper, tkinterdnd2, yt-dlp, pyannote-audio, onnx-asr
```

System-level: FFmpeg (for ETA + YouTube audio extraction + Parakeet input normalization). Optional: CUDA/cuDNN for GPU.

## Python version

Code uses `X | Y` union syntax — requires **Python 3.10+** despite `pyproject.toml` stating `>=3.9`.

## Threading

Transcription runs in a daemon thread. **Never touch Tkinter widgets from the worker thread.** All cross-thread communication goes through `queue.Queue` via `QueueLogger` (`logging_setup.py`). The worker puts typed tuples: `("log", str)`, `("progress", dict)`, `("done", dict)`. The GUI polls the queue every 120ms via `root.after`.

## CUDA gotcha

`transcriber.py` caches CUDA models in `_cuda_model_cache` by name and **never frees them**. CTranslate2's CUDA teardown calls `abort()` on Windows — `del` and `unload_model()` are both unsafe. VRAM is reclaimed at process exit. CPU models are released normally.

`transcribe_file()` tries CUDA with `float16`; on any exception it falls back to CPU with `int8`. Path injection for Windows cuDNN is in `cuda.py` (`preload_cuda_paths`, no-op off Windows).

## Parakeet engine (optional, `engine="parakeet"`)

`parakeet.py` wraps `onnx-asr` (NVIDIA Parakeet TDT) as an alternative to faster-whisper — optional, gated by `is_available()`. Unlike CTranslate2, `onnxruntime` does **not** raise when a requested execution provider is unavailable; it silently falls back. So CUDA use is decided upfront via `onnxruntime.get_available_providers()`, not try/except. Input is normalized to 16kHz mono WAV (`utils.convert_to_wav16k_mono`, shared with `diarizer.py`), chunked into fixed 30s windows for progress reporting only (no silence detection — `onnx-asr` VAD handles correctness), and each chunk's `model.recognize(...)` call becomes one `Segment(start, end, text)` namedtuple. `transcriber.transcribe_file` branches on `engine`; the Parakeet path leaves `model = None` in that function (its own model lives in `parakeet._model_cache`) and fakes `info = types.SimpleNamespace(language="en", language_probability=1.0)` so the shared tail stays branch-free.

## Settings

Persisted as JSON at `~/LocalTranscriptLogs/settings.json`. `settings_manager.py` imports from `summarizer.py` (not circular — `summarizer` uses only stdlib `urllib`).

## Summarizer

Uses **stdlib `urllib` only** — no `requests` or `httpx`. Calls any OpenAI-compatible chat-completions endpoint (OpenRouter, Ollama, etc.).

## Diarization

`pyannote.audio` is optional. Requires a HuggingFace token (persisted in settings) for initial model download; inference runs offline after that. Video files need FFmpeg WAV conversion before pyannote can process them.

## Output

Given `input.mp4` → `input_transcript.txt` + `input_subtitles.srt`, written to the configured output directory (`output_dir` setting, defaults to `~/LocalTranscriptLogs/transcripts`) or alongside the input if unset.

## main.py

Does two things before `src.gui.main()`:
1. Sets `KMP_DUPLICATE_LIB_OK=TRUE` (workaround for duplicate OpenMP runtimes)
2. Enables `faulthandler` → writes C-level crash traces to `~/LocalTranscriptLogs/crash.log`

## Import order

Strictly one-way, no cycles. Bottom-up:

```
constants → logging_setup → {cuda, settings_manager, youtube}
utils (stdlib only, no internal deps)
diarizer imports utils (convert_to_wav16k_mono)
parakeet imports constants, logging_setup, utils
summarizer (stdlib only)
transcriber (imports constants, cuda, diarizer, logging_setup, parakeet, utils)
gui (imports everything above)
main.py (imports gui)
```

`filebrowser.py` is a helper imported by `gui.py` only.

## CLAUDE.md

Up to date — reflects the current package layout, import graph, and Parakeet engine. Threading and CUDA model caching details here are consistent with it.
