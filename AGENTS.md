# AGENTS.md

## Run

```bash
python main.py          # GUI entry (sets KMP_DUPLICATE_LIB_OK + faulthandler crash logging)
```

No test, lint, or typecheck commands exist.

## Dependencies

```bash
pip install .           # core: faster-whisper, tkinterdnd2, yt-dlp, pyannote-audio
```

System-level: FFmpeg (for ETA + YouTube audio extraction). Optional: CUDA/cuDNN for GPU.

## Python version

Code uses `X | Y` union syntax — requires **Python 3.10+** despite `pyproject.toml` stating `>=3.9`.

## Threading

Transcription runs in a daemon thread. **Never touch Tkinter widgets from the worker thread.** All cross-thread communication goes through `queue.Queue` via `QueueLogger` (`logging_setup.py`). The worker puts typed tuples: `("log", str)`, `("progress", dict)`, `("done", dict)`. The GUI polls the queue every 120ms via `root.after`.

## CUDA gotcha

`transcriber.py` caches CUDA models in `_cuda_model_cache` by name and **never frees them**. CTranslate2's CUDA teardown calls `abort()` on Windows — `del` and `unload_model()` are both unsafe. VRAM is reclaimed at process exit. CPU models are released normally.

`transcribe_file()` tries CUDA with `float16`; on any exception it falls back to CPU with `int8`. Path injection for Windows cuDNN is in `cuda.py` (`preload_cuda_paths`).

## Settings

Persisted as JSON at `~/LocalTranscriptLogs/settings.json`. `settings_manager.py` imports from `summarizer.py` (not circular — `summarizer` uses only stdlib `urllib`).

## Summarizer

Uses **stdlib `urllib` only** — no `requests` or `httpx`. Calls any OpenAI-compatible chat-completions endpoint (OpenRouter, Ollama, etc.).

## Diarization

`pyannote.audio` is optional. Requires a HuggingFace token (persisted in settings) for initial model download; inference runs offline after that. Video files need FFmpeg WAV conversion before pyannote can process them.

## Output

Given `input.mp4` → `input_transcript.txt` + `input_subtitles.srt` in the same directory.

## main.py

Does two things before `src.gui.main()`:
1. Sets `KMP_DUPLICATE_LIB_OK=TRUE` (workaround for duplicate OpenMP runtimes)
2. Enables `faulthandler` → writes C-level crash traces to `~/LocalTranscriptLogs/crash.log`

## Import order

Strictly one-way, no cycles. Bottom-up:

```
constants → logging_setup → {cuda, settings_manager, utils, diarizer, youtube}
summarizer (stdlib only)
transcriber (imports constants, cuda, diarizer, logging_setup, utils)
gui (imports everything above)
main.py (imports gui)
```

`filebrowser.py` is a helper imported by `gui.py` only.

## CLAUDE.md

Contains older layout details. Import graph section is stale (missing `settings_manager`, `summarizer`, `diarizer`, `filebrowser`). Threading and CUDA model caching details here supersede it.
