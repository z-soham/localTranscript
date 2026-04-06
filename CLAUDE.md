# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python main.py
```

## Dependencies

Install required packages:

```bash
pip install faster-whisper tkinterdnd2
```

- `faster-whisper` — required for transcription (uses CTranslate2 under the hood)
- `tkinterdnd2` — optional, enables drag-and-drop file input
- `ffprobe` (from FFmpeg) — optional, used to detect media duration for ETA calculation

## Package layout

```
local_transcript/
├── constants.py      # APP_TITLE, SUPPORTED_EXTENSIONS, MODEL_OPTIONS, LOG_DIR
├── logging_setup.py  # setup_logging(), LOGGER, SESSION_LOG_PATH, QueueLogger, TranscriptionError
├── utils.py          # format_timestamp, seconds_to_human, get_media_duration_seconds, write_txt, write_srt
├── cuda.py           # locate_cudnn_hint, preload_cuda_paths, should_force_cpu_after_cuda_error
├── transcriber.py    # transcribe_file (core logic, runs in background thread)
└── gui.py            # TranscriptApp, build_root, main()
main.py               # thin entry point — sets KMP_DUPLICATE_LIB_OK, calls local_transcript.gui.main()
```

Import graph is strictly one-way: `constants → logging_setup → cuda → transcriber → gui → main.py`.

## Architecture

**Threading model:** Transcription runs in a daemon thread (`TranscriptApp.worker_thread`). All cross-thread communication goes through `self.queue` (a `queue.Queue`). The UI only reads from it in `_poll_queue`, called every 120ms on the main thread via `root.after`. Never call Tkinter widgets directly from the worker thread.

**QueueLogger** (`logging_setup.py`) is the thread-boundary contract. It puts typed tuples onto the queue: `("log", str)`, `("progress", dict)`, `("done", dict)`. The GUI's `_handle_progress` and `_handle_done` methods consume these.

**transcribe_file** (`transcriber.py`) tries CUDA first with `float16`; on any exception it falls back to CPU with `int8`. CUDA path injection happens in `cuda.py` via `preload_cuda_paths`, which calls `os.add_dll_directory` and prepends cuDNN bin dirs to `PATH`.

**Output files** are written alongside the input: `<name>_transcript.txt` (plain text) and `<name>_subtitles.srt` (timed subtitles).

**Logs** are written per-session to `~/LocalTranscriptLogs/local_transcript_<timestamp>.log` via a `RotatingFileHandler` (2 MB, 3 backups).

## Model options

`medium`, `large-v3` (selectable in UI). CUDA uses `float16`; CPU uses `int8`.

## Supported media formats

`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.flac`
