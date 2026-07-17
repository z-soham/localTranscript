# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python main.py
```

No test, lint, or typecheck commands exist in this repo.

## Dependencies

```bash
pip install .           # core: faster-whisper, tkinterdnd2, yt-dlp, pyannote-audio, onnx-asr
```

- `faster-whisper` — required for transcription (CTranslate2 under the hood)
- `tkinterdnd2` — optional, drag-and-drop file input
- `yt-dlp` — optional, YouTube URL audio download (requires FFmpeg for MP3 conversion)
- `pyannote-audio` — optional, multi-speaker diarization (requires a HuggingFace token; video files need FFmpeg WAV conversion first)
- `onnx-asr` (`onnx-asr[cpu,hub]`) — optional, alternative "Parakeet" ASR engine (NVIDIA Parakeet TDT via ONNX Runtime); faster than faster-whisper on CPU, English-only
- `ffprobe`/`ffmpeg` (system-level, not pip) — optional, used for media duration/ETA and required by YouTube download, diarization of video files, and the Parakeet engine (any non-WAV input)

Code uses `X | Y` union syntax, which requires **Python 3.10+** despite `pyproject.toml` declaring `>=3.9`.

## Package layout

```
src/
├── constants.py        # APP_TITLE, SUPPORTED_EXTENSIONS, MODEL_OPTIONS, LOG_DIR, YT_DOWNLOAD_DIR, DEFAULT_OUTPUT_DIR
├── logging_setup.py    # setup_logging(), LOGGER, SESSION_LOG_PATH, QueueLogger, TranscriptionError
├── utils.py            # format_timestamp, seconds_to_human, get_media_duration_seconds, write_txt, write_srt
├── cuda.py             # locate_cudnn_hint, preload_cuda_paths, should_force_cpu_after_cuda_error (Windows-only DLL path fixes)
├── settings_manager.py # load_settings/save_settings — JSON persistence at ~/LocalTranscriptLogs/settings.json
├── summarizer.py        # summarize_transcript() — LLM summarisation via any OpenAI-compatible endpoint, stdlib urllib only
├── diarizer.py          # run_diarization() — speaker diarization via pyannote.audio (optional dependency)
├── parakeet.py          # transcribe_with_parakeet() — alternative ASR engine via onnx-asr (optional dependency)
├── youtube.py           # download_youtube_audio(), YT_DLP_AVAILABLE — requires yt-dlp + FFmpeg
├── filebrowser.py       # in-app file/folder browser widget, used only by gui.py
└── gui.py               # TranscriptApp, build_root, main()
main.py                  # entry point — sets KMP_DUPLICATE_LIB_OK, enables faulthandler crash logging, calls src.gui.main()
```

Import graph is strictly one-way, no cycles:

```
constants → logging_setup → {cuda, settings_manager, youtube}
utils (stdlib-only, no internal deps)
diarizer imports utils (convert_to_wav16k_mono) — not circular, utils has no deps
parakeet imports constants, logging_setup, utils
summarizer (stdlib-only, no internal deps)
settings_manager imports summarizer (for default prompt/system-message constants — not circular)
transcriber (imports constants, cuda, diarizer, logging_setup, parakeet, utils)
gui (imports everything above, including filebrowser)
main.py (imports gui)
```

## Architecture

**Threading model:** Transcription runs in a daemon thread (`TranscriptApp.worker_thread`). All cross-thread communication goes through `self.queue` (a `queue.Queue`). The UI only reads from it in `_poll_queue`, called every 120ms on the main thread via `root.after`. Never call Tkinter widgets directly from the worker thread.

**QueueLogger** (`logging_setup.py`) is the thread-boundary contract. It puts typed tuples onto the queue: `("log", str)`, `("progress", dict)`, `("done", dict)`. The GUI's `_handle_progress` and `_handle_done` methods consume these.

**transcribe_file** (`transcriber.py`) tries CUDA first with `float16`; on any exception during model load it falls back to CPU with `int8`. If CUDA OOMs mid-transcription, it releases the model and retries the whole transcription on CPU. Output is written and `logger.done(...)` is signaled *before* model cleanup runs, since CUDA teardown can itself crash.

**CUDA model caching:** CTranslate2's CUDA teardown calls `abort()` on Windows, so `del`/`unload_model()` on a CUDA model is unsafe. `_cuda_model_cache` in `transcriber.py` keeps loaded CUDA models alive by name for reuse across transcriptions; VRAM is only reclaimed at process exit. CPU models are released normally after each run. Windows cuDNN/CUDA DLL path injection happens in `cuda.py` via `preload_cuda_paths` (`os.add_dll_directory` + prepending bin dirs to `PATH`); it's a no-op off Windows.

**Settings** (`settings_manager.py`) persist as JSON at `~/LocalTranscriptLogs/settings.json`, covering model/device choice, LLM summarizer config (URL, API key, model, prompts), diarization toggle + HF token, last browser directory, and output directory. Missing keys in an existing file are back-filled from `DEFAULT_SETTINGS`. Empty `llm_api_key`/`hf_token` fall back to the `OPENROUTER_API_KEY`/`HF_TOKEN` env vars.

**Summarizer** (`summarizer.py`) calls any OpenAI-compatible `/chat/completions` endpoint (OpenRouter, Ollama, etc.) using stdlib `urllib` only — no `requests`/`httpx`. Two modes (`Meeting`, `General Video`) each with their own default system message + prompt template, overridable via settings. SRT input has timestamps/sequence numbers stripped before being sent to the LLM.

**Diarization** (`diarizer.py`) is fully optional — gated by `is_available()` checking for `pyannote.audio`. The pipeline is loaded once and cached at module scope. Video files are converted to a temp 16kHz mono WAV via `utils.convert_to_wav16k_mono` first since pyannote can't decode video containers. Speaker turns are matched back to segments by maximum time-overlap (`_match_segments_to_speakers` in `transcriber.py`).

**Parakeet engine** (`parakeet.py`) is a second, fully optional ASR engine — gated by `is_available()` checking for `onnx_asr`, selected via the `engine` setting (`"faster-whisper"` default, or `"parakeet"`). Input is normalized to 16kHz mono WAV via `utils.convert_to_wav16k_mono`, loaded as a float32 array with stdlib `wave`, and split into fixed 30s chunks (progress checkpoints only — `onnx-asr` runs its own internal VAD, so chunk boundaries aren't correctness-sensitive). Each chunk is transcribed with `model.recognize(...)` and wrapped into a `Segment(start, end, text)` namedtuple matching the shape `write_txt`/`write_srt`/`_match_segments_to_speakers` already expect. `onnxruntime` does not raise when a requested execution provider (e.g. CUDA) is unavailable — it silently falls back — so CUDA availability is checked upfront via `onnxruntime.get_available_providers()` rather than via try/except. `_model_cache` in `parakeet.py` mirrors `transcriber._cuda_model_cache`'s "never explicitly unload a CUDA model" precedent. In `transcriber.transcribe_file`, the Parakeet path leaves `model = None` (the `_release_model` cleanup no-ops) and stands in `types.SimpleNamespace(language="en", language_probability=1.0)` for `info` so the shared language-log line stays branch-free.

**Output files** are written alongside the input (or to the configured output directory): `<name>_transcript.txt` (plain text) and `<name>_subtitles.srt` (timed subtitles), both prefixed with speaker labels when diarization is enabled.

**Logs** are written per-session to `~/LocalTranscriptLogs/local_transcript_<timestamp>.log` via a `RotatingFileHandler` (2 MB, 3 backups). `main.py` additionally enables `faulthandler` to catch native (C-level) crashes to `~/LocalTranscriptLogs/crash.log`, and sets `KMP_DUPLICATE_LIB_OK=TRUE` as a workaround for duplicate OpenMP runtimes.

## Model options

Engine: `faster-whisper` (default) or `parakeet`, selectable in UI (`ENGINE_OPTIONS`).

`faster-whisper` models: `medium`, `large-v3` (selectable in UI). CUDA uses `float16`; CPU uses `int8`. The Whisper model choice is ignored when engine is `parakeet` — that engine always uses `nemo-parakeet-tdt-0.6b-v3` (`PARAKEET_MODEL_ID`).

## Supported media formats

`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.flac`
