# TODO: Add Parakeet (onnx-asr) as a second, optional ASR engine

## Context

The project currently transcribes exclusively via `faster-whisper`. For English-only
audio, NVIDIA's Parakeet TDT model (via the lightweight, MIT-licensed `onnx-asr`
package) beats Whisper large-v3 on WER and is dramatically faster on CPU (~50-100x
realtime vs faster-whisper's ~2-5x), while `onnx-asr` is portable (Windows/Linux/macOS,
CPU/CUDA) unlike raw NVIDIA NeMo.

Goal: let a user pick "Parakeet" as an alternative engine from Settings, with full
progress-bar/ETA parity and CUDA support, **without changing default behavior** —
existing installs with only `faster-whisper` keep working exactly as today. Parakeet
is an optional dependency, following the same graceful-degradation pattern already
used for `tkinterdnd2`, `yt-dlp`, and `pyannote.audio`.

Scope decisions already made:
- **Full progress parity**: chunk audio ourselves and update %/ETA per chunk (not a
  single blocking call with no feedback).
- **Wire up CUDA in v1**, with graceful fallback to CPU, mirroring the existing
  faster-whisper try/except pattern.
- Chunking is fixed-size (30s), not silence-aware — `onnx-asr` already runs its own
  VAD-based long-form recognition internally, so chunking here only exists for
  progress checkpoints, not correctness. Simplified out of the initial design per
  a `/ponytail` review (was going to add an `ffmpeg silencedetect` pass +
  snap-to-silence logic — cut as unnecessary complexity).

Two things are genuinely unconfirmed from docs and must be verified against the
installed package before writing real code — don't guess at these:
1. The exact shape of `model.recognize(...)`'s return value (does it expose
   per-token/segment timestamps, or just a text string?).
2. The exact CUDA wiring for `onnx_asr.load_model(...)` (provider list param name).

## Tasks

- [ ] **Spike first**: `pip install "onnx-asr[cpu,hub]"`, load
  `nemo-parakeet-tdt-0.6b-v3`, call `recognize()` on a short WAV, print the raw
  result object. Settles the two unconfirmed API points above before any other
  work starts.
- [ ] Add `onnx-asr` to `pyproject.toml` `dependencies` (same convention as
  `pyannote-audio` — optional at runtime despite being a hard pip dependency) and to
  the manual-install list in `CLAUDE.md`/`README.md`. Import lazily behind
  `try/except ImportError`, same as `WhisperModel` in `transcriber.py` and
  `pyannote.audio` in `diarizer.py`.
- [ ] Extract the inline ffmpeg-to-WAV conversion block in `diarizer.run_diarization`
  into `utils.py` as `convert_to_wav16k_mono(input_path) -> Path`; call it from both
  `diarizer.py` and the new `parakeet.py` (avoid duplicating ffmpeg-shelling logic
  for a second engine).
- [ ] New module `src/parakeet.py` (same import layer as `src/diarizer.py`: imports
  only `constants`, `logging_setup`, `utils`; imported by `transcriber.py`):
  - [ ] `is_available() -> bool` via `importlib.util.find_spec("onnx_asr")`, same
    shape as `diarizer.is_available()`.
  - [ ] Module-level `_model_cache: dict[str, object] = {}` keyed by `device_used`
    ("cuda"/"cpu") — same "never explicitly unload a CUDA model" precedent as
    `transcriber._cuda_model_cache`. Defensive only; revisit/delete if the spike or
    later testing shows onnxruntime has no equivalent teardown crash.
  - [ ] `load_parakeet_model(prefer_cuda, logger) -> (model, device_used)` — try CUDA
    execution provider first if requested, `except` → log + fall back to CPU, same
    defensive shape as the existing faster-whisper CUDA/CPU fallback. Confirm the
    real execution-provider param name against the installed `onnx_asr` source
    (e.g. `inspect.signature`) before hardcoding it.
  - [ ] `transcribe_with_parakeet(input_path, prefer_cuda, logger, stop_event, duration_seconds) -> list[Segment] | None`:
    1. Normalize input via `utils.convert_to_wav16k_mono`.
    2. Load the WAV into a numpy float32 array (stdlib `wave` module).
    3. Build fixed-size chunk boundaries, one-liner, no silence detection:
       `[(i, min(i + CHUNK_SEC, duration)) for i in range(0, duration, CHUNK_SEC)]`,
       `CHUNK_SEC = 30`.
    4. Per chunk: slice the array, call `model.recognize(chunk_array)`, check
       `stop_event`, call `logger.progress(...)` using the existing six-field
       `QueueLogger.progress()` contract (no GUI changes needed for this).
    5. Wrap each chunk's result into
       `Segment = collections.namedtuple("Segment", "start end text")` — matches the
       minimal shape `write_txt`/`write_srt`/`_match_segments_to_speakers` already
       read (`.text`, `.start`, `.end`). If the spike found finer sub-chunk
       timestamps, use those offset by chunk-start instead of one segment per chunk.
- [ ] `constants.py`: add `ENGINE_OPTIONS = ["faster-whisper", "parakeet"]` and
  `PARAKEET_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"`.
- [ ] `settings_manager.py`: add `"engine": "faster-whisper"` to `DEFAULT_SETTINGS`
  (existing back-fill logic already handles old settings.json files missing it).
- [ ] `transcriber.py`:
  - [ ] Add `engine: str = "faster-whisper"` keyword-only param to `transcribe_file`
    (default preserves current behavior exactly).
  - [ ] Add availability check mirroring the `WhisperModel is None` check: if
    `engine == "parakeet"` and `not parakeet.is_available()`, raise
    `TranscriptionError('onnx-asr is not installed. Run: pip install "onnx-asr[cpu,hub]"')`.
  - [ ] Branch model-loading/transcription: `engine == "parakeet"` calls
    `parakeet.transcribe_with_parakeet(...)` and leaves `model = None` (existing
    `finally: _release_model(...)` already no-ops on `model is None`, no change
    needed there). `engine == "faster-whisper"` branch is untouched.
  - [ ] For the Parakeet path, stand in
    `types.SimpleNamespace(language="en", language_probability=1.0)` for `info` so
    the shared tail's language-log line stays branch-free.
  - [ ] Confirm out of scope for v1 (no code needed, just don't add it): no mid-run
    CUDA-OOM chunk-level fallback for Parakeet.
- [ ] `gui.py`:
  - [ ] `self.engine_var = tk.StringVar(value=_s.get("engine", "faster-whisper"))`
    near the existing `model_var`/`device_pref_var` init.
  - [ ] Add `ttk.Combobox(..., values=ENGINE_OPTIONS, state="readonly")` in the
    "Transcription" section of `_build_settings_tab`, next to the model/device
    combos.
  - [ ] `_save_settings`: add `"engine": self.engine_var.get()`.
  - [ ] `start_transcription`: add `engine=self.engine_var.get()` to both
    `transcribe_file(...)` call sites (local-file and YouTube branches).
  - [ ] Optional/skip-if-not-cheap: grey out the Model combobox when engine is
    "parakeet" (its model id is fixed).
- [ ] Docs: update `CLAUDE.md` and reconcile `AGENTS.md` (currently documents the
  faster-whisper-only architecture) — new optional dependency, `src/parakeet.py` in
  the import graph, new `engine` setting.

## Known limitations (accepted for v1, not bugs)

- Cancellation granularity for Parakeet is per-chunk (~30s worst case), coarser than
  faster-whisper's per-segment check.
- Diarization speaker labeling on the Parakeet path is only as fine-grained as the
  ~30s chunk segments (unless the spike finds finer timestamp data).
- Fixed-size chunk boundaries mean a word can occasionally split across a chunk seam.
  Don't pre-build silence detection for this — revisit only on a real complaint.
- No mid-run CUDA OOM fallback for Parakeet.

## Verification checklist

- [ ] Spike: confirm `recognize()` return shape + CUDA provider param (see Tasks).
- [ ] Regression: Engine at default ("faster-whisper"), transcribe a short file —
  behavior identical to before this change.
- [ ] Switch to "parakeet", transcribe a short (~1 min) English file — progress bar
  advances per chunk, `_transcript.txt`/`_subtitles.srt` produced and readable,
  diarization (if enabled) still labels speakers.
- [ ] Transcribe a longer (~20-30 min) file on Parakeet — no obviously
  duplicated/missing words at chunk seams.
- [ ] Uninstall `onnx-asr`, select "parakeet" — clear `TranscriptionError`, no crash.
- [ ] If CUDA available: Parakeet loads on CUDA, falls back to CPU cleanly if forced.

## Rough time estimate (1hr meeting, unverified until the spike runs)

- ffmpeg normalize + WAV load: well under a minute combined, negligible vs. ASR
  compute (ffmpeg only needs the audio stream even for video input).
- ASR compute: dominant cost. Rough estimate — CUDA: ~20-70s; CPU: ~2-6 min. Treat
  as a placeholder; replace with real numbers from the spike.
