import os
import queue
import signal
import traceback
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = None

from src.constants import APP_TITLE, MODEL_OPTIONS
from src.logging_setup import LOGGER, SESSION_LOG_PATH, QueueLogger
from src.settings_manager import load_settings, save_settings
from src.summarizer import SUMMARY_MODES, summarize_transcript
from src.transcriber import WhisperModel, transcribe_file
from src.utils import seconds_to_human
from src.youtube import YT_DLP_AVAILABLE, download_youtube_audio


class TranscriptApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x800")
        self.root.minsize(860, 680)

        self.queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.session_log_path = SESSION_LOG_PATH

        # Load persisted settings and expose as tk vars (shared across all tabs)
        _s = load_settings()
        self.model_var = tk.StringVar(value=_s["model"])
        self.device_pref_var = tk.StringVar(value=_s["device"])
        self.llm_url_var = tk.StringVar(value=_s["llm_url"])
        self.llm_api_key_var = tk.StringVar(value=_s["llm_api_key"])
        self.llm_model_var = tk.StringVar(value=_s["llm_model"])
        self.diarize_var = tk.BooleanVar(value=_s["diarize_enabled"])
        self.hf_token_var = tk.StringVar(value=_s["hf_token"])

        # Transcription-tab state
        self.file_path_var = tk.StringVar()
        self.youtube_url_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.progress_text_var = tk.StringVar(value="No active transcription")

        # Summarisation-tab state
        self.summary_file_var = tk.StringVar()
        self.summary_mode_var = tk.StringVar(value=SUMMARY_MODES[0])
        self.summary_status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # Top-level UI skeleton
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(14, 12, 14, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Local speech-to-text transcription and AI-powered meeting summarisation.",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 0))

        self._build_transcription_tab()
        self._build_summarisation_tab()
        self._build_settings_tab()

        status_bar = ttk.Frame(self.root, padding=(14, 4, 14, 8))
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        # Startup messages
        self._log("Application initialized.")
        self._log(f"Session log: {self.session_log_path}")
        LOGGER.info("Application initialized")
        LOGGER.info("Session log: %s", self.session_log_path)
        if not DND_AVAILABLE:
            self._log("Drag-and-drop disabled — install tkinterdnd2 to enable: pip install tkinterdnd2")
        if WhisperModel is None:
            self._log("faster-whisper not installed — run: pip install faster-whisper")
        if not YT_DLP_AVAILABLE:
            self._log("yt-dlp not installed — YouTube URL input disabled. Run: pip install yt-dlp")

    # ------------------------------------------------------------------
    # Tab 1 — Transcription
    # ------------------------------------------------------------------

    def _build_transcription_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Transcribe  ")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        # --- Input ---
        file_frame = ttk.LabelFrame(tab, text="Input", padding=12)
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(1, weight=1)

        self.drop_zone = tk.Label(
            file_frame,
            text=self._drop_zone_text(),
            relief="groove",
            borderwidth=2,
            padx=16,
            pady=18,
            bg="#2d2d2d",
            fg="#cccccc",
            font=("Segoe UI", 11),
            justify="center",
        )
        self.drop_zone.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))

        if DND_AVAILABLE:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)

        ttk.Entry(file_frame, textvariable=self.file_path_var).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        ttk.Button(file_frame, text="Browse Media", command=self.browse_file).grid(row=1, column=2, padx=(0, 8))
        ttk.Button(file_frame, text="Clear", command=self.clear_file).grid(row=1, column=3)

        ttk.Separator(file_frame, orient="horizontal").grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(10, 8)
        )
        ttk.Label(file_frame, text="Or — YouTube URL:").grid(row=3, column=0, sticky="w", padx=(0, 8))
        self.youtube_url_entry = ttk.Entry(file_frame, textvariable=self.youtube_url_var)
        self.youtube_url_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(0, 8))
        ttk.Button(file_frame, text="Clear", command=self.clear_youtube_url).grid(row=3, column=3)

        # --- Buttons ---
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for col in range(3):
            btn_frame.columnconfigure(col, weight=1)

        self.start_btn = ttk.Button(btn_frame, text="Start Transcription", command=self.start_transcription)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_transcription, state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        ttk.Button(btn_frame, text="Open Output Folder", command=self.open_output_folder).grid(
            row=0, column=2, sticky="ew"
        )

        # --- Progress + console ---
        progress_frame = ttk.LabelFrame(tab, text="Progress", padding=12)
        progress_frame.grid(row=2, column=0, sticky="nsew")
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(2, weight=1)

        self.progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew")

        ttk.Label(progress_frame, textvariable=self.progress_text_var).grid(
            row=1, column=0, sticky="w", pady=(6, 10)
        )

        console_frame = ttk.Frame(progress_frame)
        console_frame.grid(row=2, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console = tk.Text(console_frame, wrap="word", height=14, bg="#0f0f0f", fg="#d8d8d8", insertbackground="#e0e0e0")
        self.console.grid(row=0, column=0, sticky="nsew")
        self.console.configure(state="disabled")

        scroll = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=scroll.set)

    # ------------------------------------------------------------------
    # Tab 2 — Summarisation
    # ------------------------------------------------------------------

    def _build_summarisation_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Summarise  ")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

        # --- File input ---
        file_frame = ttk.LabelFrame(tab, text="Transcript Input", padding=12)
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(0, weight=1)

        ttk.Label(
            file_frame,
            text="Select a .txt or .srt transcript to summarise with an LLM.",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Entry(file_frame, textvariable=self.summary_file_var).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(file_frame, text="Browse Transcript", command=self.browse_summary_file).grid(
            row=1, column=1, padx=(0, 8)
        )
        ttk.Button(file_frame, text="Clear", command=lambda: self.summary_file_var.set("")).grid(row=1, column=2)

        # --- Active LLM config (read-only display) ---
        cfg_frame = ttk.LabelFrame(tab, text="LLM Configuration  (edit in Settings tab)", padding=10)
        cfg_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        cfg_frame.columnconfigure(1, weight=1)
        cfg_frame.columnconfigure(3, weight=1)

        ttk.Label(cfg_frame, text="URL:").grid(row=0, column=0, sticky="w")
        ttk.Label(cfg_frame, textvariable=self.llm_url_var, foreground="#aaaaaa").grid(
            row=0, column=1, sticky="w", padx=(6, 24)
        )
        ttk.Label(cfg_frame, text="Model:").grid(row=0, column=2, sticky="w")
        ttk.Label(cfg_frame, textvariable=self.llm_model_var, foreground="#aaaaaa").grid(
            row=0, column=3, sticky="w", padx=(6, 0)
        )

        # --- Summary mode ---
        mode_frame = ttk.LabelFrame(tab, text="Summary Mode", padding=10)
        mode_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        for i, mode in enumerate(SUMMARY_MODES):
            ttk.Radiobutton(
                mode_frame,
                text=mode,
                variable=self.summary_mode_var,
                value=mode,
            ).grid(row=0, column=i, sticky="w", padx=(0, 24))

        mode_descriptions = {
            "Meeting": "Minutes of Meeting, key decisions, action items, participants, next steps.",
            "General Video": "Overview, key takeaways, important details, suggestions, further exploration.",
        }
        self._mode_desc_label = ttk.Label(
            mode_frame, text=mode_descriptions[SUMMARY_MODES[0]], foreground="#aaaaaa", font=("Segoe UI", 9)
        )
        self._mode_desc_label.grid(row=1, column=0, columnspan=len(SUMMARY_MODES), sticky="w", pady=(4, 0))

        def _update_mode_desc(*_):
            self._mode_desc_label.configure(text=mode_descriptions.get(self.summary_mode_var.get(), ""))

        self.summary_mode_var.trace_add("write", _update_mode_desc)

        # --- Action buttons ---
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.summarise_btn = ttk.Button(btn_frame, text="Generate Summary", command=self.start_summarisation)
        self.summarise_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.save_summary_btn = ttk.Button(
            btn_frame, text="Save Summary", command=self.save_summary, state="disabled"
        )
        self.save_summary_btn.grid(row=0, column=1, sticky="ew")

        # --- Output ---
        out_frame = ttk.LabelFrame(tab, text="Summary Output", padding=12)
        out_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 4))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)

        self.summary_output = tk.Text(out_frame, wrap="word", bg="#0f0f0f", fg="#d8d8d8", height=16, insertbackground="#e0e0e0")
        self.summary_output.grid(row=0, column=0, sticky="nsew")
        self.summary_output.configure(state="disabled")

        sum_scroll = ttk.Scrollbar(out_frame, orient="vertical", command=self.summary_output.yview)
        sum_scroll.grid(row=0, column=1, sticky="ns")
        self.summary_output.configure(yscrollcommand=sum_scroll.set)

        ttk.Label(tab, textvariable=self.summary_status_var, foreground="#aaaaaa").grid(
            row=5, column=0, sticky="w", pady=(2, 0)
        )

    # ------------------------------------------------------------------
    # Tab 3 — Settings
    # ------------------------------------------------------------------

    def _build_settings_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="  Settings  ")
        tab.columnconfigure(1, weight=1)

        r = 0

        # == Transcription ==
        ttk.Label(tab, text="Transcription", font=("Segoe UI", 11, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1
        ttk.Separator(tab).grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        r += 1

        ttk.Label(tab, text="Whisper model:").grid(row=r, column=0, sticky="w", padx=(0, 16), pady=5)
        ttk.Combobox(
            tab, textvariable=self.model_var, values=MODEL_OPTIONS, state="readonly", width=24
        ).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(tab, text="Preferred device:").grid(row=r, column=0, sticky="w", padx=(0, 16), pady=5)
        ttk.Combobox(
            tab, textvariable=self.device_pref_var, values=["cuda", "cpu"], state="readonly", width=24
        ).grid(row=r, column=1, sticky="w")
        r += 1

        # Spacer
        ttk.Label(tab, text="").grid(row=r, column=0)
        r += 1

        # == LLM / Summarisation ==
        ttk.Label(tab, text="LLM / Summarisation", font=("Segoe UI", 11, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1
        ttk.Separator(tab).grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        r += 1

        ttk.Label(tab, text="API base URL:").grid(row=r, column=0, sticky="w", padx=(0, 16), pady=5)
        ttk.Entry(tab, textvariable=self.llm_url_var, width=54).grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(
            tab,
            text="e.g.  https://openrouter.ai/api/v1   or   http://localhost:11434/v1  (Ollama)",
            foreground="#aaaaaa",
            font=("Segoe UI", 9),
        ).grid(row=r, column=1, sticky="w", pady=(0, 6))
        r += 1

        ttk.Label(tab, text="API key:").grid(row=r, column=0, sticky="w", padx=(0, 16), pady=5)
        self._api_key_entry = ttk.Entry(tab, textvariable=self.llm_api_key_var, show="*", width=54)
        self._api_key_entry.grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(
            tab,
            text="Leave blank when not required (e.g. local Ollama without auth).",
            foreground="#aaaaaa",
            font=("Segoe UI", 9),
        ).grid(row=r, column=1, sticky="w", pady=(0, 6))
        r += 1

        ttk.Label(tab, text="LLM model:").grid(row=r, column=0, sticky="w", padx=(0, 16), pady=5)
        ttk.Entry(tab, textvariable=self.llm_model_var, width=54).grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(
            tab,
            text="e.g.  openai/gpt-4o-mini   or   llama3.2   or   mistral",
            foreground="#aaaaaa",
            font=("Segoe UI", 9),
        ).grid(row=r, column=1, sticky="w", pady=(0, 6))
        r += 1

        # Show/hide key toggle
        show_key_var = tk.BooleanVar(value=False)

        def _toggle_key_visibility():
            self._api_key_entry.configure(show="" if show_key_var.get() else "*")

        ttk.Checkbutton(
            tab, text="Show API key", variable=show_key_var, command=_toggle_key_visibility
        ).grid(row=r, column=1, sticky="w", pady=(0, 12))
        r += 1

        # Spacer
        ttk.Label(tab, text="").grid(row=r, column=0)
        r += 1

        # == Speaker Diarization ==
        ttk.Label(tab, text="Speaker Diarization", font=("Segoe UI", 11, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1
        ttk.Separator(tab).grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        r += 1

        ttk.Checkbutton(
            tab, text="Enable speaker diarization", variable=self.diarize_var
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=5)
        r += 1

        ttk.Label(tab, text="HuggingFace token:").grid(row=r, column=0, sticky="w", padx=(0, 16), pady=5)
        self._hf_token_entry = ttk.Entry(tab, textvariable=self.hf_token_var, show="*", width=54)
        self._hf_token_entry.grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(
            tab,
            text="Required once to download model from HuggingFace. Inference is fully offline after that.",
            foreground="#aaaaaa",
            font=("Segoe UI", 9),
        ).grid(row=r, column=1, sticky="w", pady=(0, 6))
        r += 1

        show_hf_var = tk.BooleanVar(value=False)

        def _toggle_hf_visibility():
            self._hf_token_entry.configure(show="" if show_hf_var.get() else "*")

        ttk.Checkbutton(
            tab, text="Show token", variable=show_hf_var, command=_toggle_hf_visibility
        ).grid(row=r, column=1, sticky="w", pady=(0, 12))
        r += 1

        # Save row
        save_row = ttk.Frame(tab)
        save_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Button(save_row, text="Save Settings", command=self._save_settings).grid(row=0, column=0, padx=(0, 14))
        self._settings_status = ttk.Label(save_row, text="", foreground="#4caf50")
        self._settings_status.grid(row=0, column=1)

    def _save_settings(self) -> None:
        save_settings(
            {
                "model": self.model_var.get(),
                "device": self.device_pref_var.get(),
                "llm_url": self.llm_url_var.get().strip(),
                "llm_api_key": self.llm_api_key_var.get(),
                "llm_model": self.llm_model_var.get().strip(),
                "diarize_enabled": self.diarize_var.get(),
                "hf_token": self.hf_token_var.get(),
            }
        )
        self._settings_status.configure(text="Settings saved.")
        self.root.after(3000, lambda: self._settings_status.configure(text=""))

    # ------------------------------------------------------------------
    # Transcription helpers
    # ------------------------------------------------------------------

    def _drop_zone_text(self) -> str:
        if DND_AVAILABLE:
            return "Drag and drop an audio or video file here\n(MP4, MKV, MP3, WAV, M4A, FLAC, and more)\n(or use Browse Media below)"
        return "Browse Media below — supports MP4, MKV, MP3, WAV, M4A, FLAC, and more\n(Install tkinterdnd2 to enable drag-and-drop)"

    def _log(self, message: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", message + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.status_var.set("Transcription in progress...")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_var.set("Ready")

    def browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select media file",
            filetypes=[
                ("Media files", "*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.m4a *.flac"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_path_var.set(path)
            self._log(f"Selected file: {path}")

    def clear_file(self) -> None:
        self.file_path_var.set("")
        self.progress["value"] = 0
        self.progress_text_var.set("No active transcription")
        self._log("Cleared selected file.")

    def clear_youtube_url(self) -> None:
        self.youtube_url_var.set("")
        self._log("Cleared YouTube URL.")

    def _normalize_drop_path(self, raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1]
        return cleaned

    def _on_drop(self, event) -> None:
        if not event.data:
            return
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self.file_path_var.set(self._normalize_drop_path(paths[0]))
            self._log(f"Dropped file: {paths[0]}")

    def open_output_folder(self) -> None:
        raw = self.file_path_var.get().strip()
        if not raw:
            messagebox.showinfo(APP_TITLE, "Select a file first.")
            return
        p = Path(raw)
        folder = p.parent if p.exists() else Path.cwd()
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open folder:\n{exc}")

    def stop_transcription(self) -> None:
        self._stop_event.set()
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopping...")
        self._log("Stop requested — waiting for current segment to finish...")

    def start_transcription(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "A transcription is already running.")
            return

        raw_file = self.file_path_var.get().strip()
        raw_url = self.youtube_url_var.get().strip()

        if raw_file and raw_url:
            messagebox.showwarning(
                APP_TITLE,
                "Both a file and a YouTube URL are filled in.\nPlease clear one before starting.",
            )
            return

        if not raw_file and not raw_url:
            messagebox.showwarning(APP_TITLE, "Please select a media file or enter a YouTube URL first.")
            return

        use_youtube = bool(raw_url) and not raw_file

        if use_youtube and not YT_DLP_AVAILABLE:
            messagebox.showerror(APP_TITLE, "yt-dlp is not installed.\nRun: pip install yt-dlp")
            return

        if not use_youtube:
            input_path = Path(raw_file)
            if not input_path.exists():
                messagebox.showerror(APP_TITLE, f"File not found:\n{input_path}")
                return

        self._stop_event.clear()
        self.progress["value"] = 0
        self.progress_text_var.set("Starting...")
        self._set_busy(True)

        model_name = self.model_var.get().strip() or "large-v3"
        prefer_cuda = self.device_pref_var.get().strip().lower() == "cuda"

        self._log("=" * 72)

        logger = QueueLogger(self.queue, LOGGER)
        stop_event = self._stop_event

        if use_youtube:
            url = raw_url
            self._log(f"YouTube URL: {url}")
            self._log(f"Model: {model_name} | Device: {'CUDA' if prefer_cuda else 'CPU'}")

            def worker() -> None:
                downloaded_path: Path | None = None
                try:
                    downloaded_path = download_youtube_audio(url, logger, stop_event)
                    if stop_event.is_set():
                        logger.log("Download complete but stop was requested — skipping transcription.")
                        logger.done(False, "cancelled")
                        return
                    logger.log(f"Starting transcription: {downloaded_path.name}")
                    transcribe_file(
                        downloaded_path, model_name, prefer_cuda, logger, stop_event,
                        diarize=self.diarize_var.get(),
                        hf_token=self.hf_token_var.get().strip(),
                    )
                except Exception as exc:
                    if stop_event.is_set():
                        logger.log("Cancelled.")
                        logger.done(False, "cancelled")
                    else:
                        logger.log("Error:")
                        logger.log(str(exc))
                        logger.log(traceback.format_exc())
                        LOGGER.exception("Unhandled YouTube/transcription failure")
                        logger.done(False, f"{exc}\nLog file: {SESSION_LOG_PATH}")
                finally:
                    if downloaded_path is not None and downloaded_path.exists():
                        try:
                            downloaded_path.unlink()
                            logger.log(f"Cleaned up temporary file: {downloaded_path.name}")
                        except OSError as e:
                            logger.log(f"Warning: could not delete temp file: {e}")

        else:
            input_path = Path(raw_file)
            self._log(f"Starting transcription: {input_path}")
            self._log(f"Model: {model_name} | Device: {'CUDA' if prefer_cuda else 'CPU'}")

            def worker() -> None:  # type: ignore[misc]
                try:
                    transcribe_file(
                        input_path, model_name, prefer_cuda, logger, stop_event,
                        diarize=self.diarize_var.get(),
                        hf_token=self.hf_token_var.get().strip(),
                    )
                except Exception as exc:
                    logger.log("Error during transcription:")
                    logger.log(str(exc))
                    logger.log(traceback.format_exc())
                    LOGGER.exception("Unhandled transcription failure")
                    logger.done(False, f"{exc}\nLog file: {SESSION_LOG_PATH}")

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                item_type, payload = self.queue.get_nowait()
                if item_type == "log":
                    self._log(payload)
                elif item_type == "progress":
                    self._handle_progress(payload)
                elif item_type == "done":
                    self._handle_done(payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll_queue)

    def _handle_progress(self, payload: dict) -> None:
        progress = payload["progress"]
        processed = payload["processed_audio_sec"]
        total = payload["total_audio_sec"]
        elapsed = payload["elapsed_wall_sec"]
        eta = payload["eta_sec"]
        speed_x = payload["speed_x"]

        if total:
            self.progress["value"] = progress * 100
            speed_text = f"{speed_x:.2f}x" if speed_x else "--"
            self.progress_text_var.set(
                f"{progress * 100:6.2f}% | Audio {seconds_to_human(processed)} / {seconds_to_human(total)} | "
                f"Elapsed {seconds_to_human(elapsed)} | ETA {seconds_to_human(eta)} | Speed {speed_text}"
            )
        else:
            self.progress_text_var.set(
                f"Processed audio {seconds_to_human(processed)} | Elapsed {seconds_to_human(elapsed)}"
            )

    def _handle_done(self, payload: dict) -> None:
        success = payload["success"]
        message = payload["message"]
        self._set_busy(False)
        if success:
            self.progress["value"] = 100
            self.progress_text_var.set("Completed")
            self._log(message)
            self._log(f"Log saved to: {self.session_log_path}")
            messagebox.showinfo(APP_TITLE, f"{message}\nLog file:\n{self.session_log_path}")
        elif message == "cancelled":
            self.progress_text_var.set("Cancelled")
            self._log(f"Log saved to: {self.session_log_path}")
            messagebox.showinfo(APP_TITLE, "Transcription was stopped.")
        else:
            self.progress_text_var.set("Failed")
            self._log(f"Failed: {message}")
            self._log(f"Detailed log: {self.session_log_path}")
            messagebox.showerror(
                APP_TITLE,
                f"Transcription failed:\n{message}\nDetailed log:\n{self.session_log_path}",
            )

    # ------------------------------------------------------------------
    # Summarisation helpers
    # ------------------------------------------------------------------

    def browse_summary_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select transcript file",
            filetypes=[
                ("Transcript files", "*.txt *.srt"),
                ("Text files", "*.txt"),
                ("SRT files", "*.srt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.summary_file_var.set(path)

    def start_summarisation(self) -> None:
        raw = self.summary_file_var.get().strip()
        if not raw:
            messagebox.showwarning(APP_TITLE, "Please select a transcript file first.")
            return

        input_path = Path(raw)
        if not input_path.exists():
            messagebox.showerror(APP_TITLE, f"File not found:\n{input_path}")
            return

        llm_url = self.llm_url_var.get().strip()
        llm_model = self.llm_model_var.get().strip()
        if not llm_url or not llm_model:
            messagebox.showwarning(
                APP_TITLE, "Please configure LLM URL and model name in the Settings tab first."
            )
            return

        api_key = self.llm_api_key_var.get()
        mode = self.summary_mode_var.get()

        self.summarise_btn.configure(state="disabled")
        self.save_summary_btn.configure(state="disabled")
        self.summary_status_var.set("Contacting LLM, please wait…")
        self.summary_output.configure(state="normal")
        self.summary_output.delete("1.0", "end")
        self.summary_output.insert("end", "Sending transcript to LLM…")
        self.summary_output.configure(state="disabled")

        def worker() -> None:
            try:
                result = summarize_transcript(input_path, llm_url, api_key, llm_model, mode)
                self.root.after(0, lambda: self._on_summary_done(result, None))
            except Exception as exc:
                err = str(exc)
                self.root.after(0, lambda: self._on_summary_done(None, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_summary_done(self, result: str | None, error: str | None) -> None:
        self.summarise_btn.configure(state="normal")
        self.summary_output.configure(state="normal")
        self.summary_output.delete("1.0", "end")

        if error:
            self.summary_status_var.set("Summarisation failed.")
            self.summary_output.insert("end", f"Error:\n{error}")
            messagebox.showerror(APP_TITLE, f"Summarisation failed:\n{error}")
        else:
            self.summary_status_var.set("Summary generated successfully.")
            self.summary_output.insert("end", result)
            self.save_summary_btn.configure(state="normal")

        self.summary_output.configure(state="disabled")

    def save_summary(self) -> None:
        content = self.summary_output.get("1.0", "end").strip()
        if not content:
            return

        default_name = "summary.md"
        raw = self.summary_file_var.get().strip()
        if raw:
            default_name = Path(raw).stem + "_summary.md"

        save_path = filedialog.asksaveasfilename(
            title="Save summary",
            defaultextension=".md",
            initialfile=default_name,
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if save_path:
            Path(save_path).write_text(content, encoding="utf-8")
            messagebox.showinfo(APP_TITLE, f"Summary saved to:\n{save_path}")


# ------------------------------------------------------------------
# Entry point helpers
# ------------------------------------------------------------------

def _setup_dark_theme(root: tk.Tk) -> None:
    BG = "#1e1e1e"
    BG2 = "#2d2d2d"
    BG3 = "#3c3c3c"
    FG = "#e0e0e0"
    FG_DIM = "#aaaaaa"
    ACCENT = "#4a9eff"
    BORDER = "#505050"
    SELECT_BG = "#264f78"

    root.configure(bg=BG)

    # Style tk dropdown listboxes inside Combobox widgets
    root.option_add("*TCombobox*Listbox.background", BG2)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", SELECT_BG)
    root.option_add("*TCombobox*Listbox.selectForeground", FG)

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".",
                    background=BG, foreground=FG, fieldbackground=BG2,
                    bordercolor=BORDER, darkcolor=BG2, lightcolor=BG3,
                    troughcolor=BG2, focuscolor=ACCENT,
                    selectbackground=SELECT_BG, selectforeground=FG,
                    insertcolor=FG)

    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG)

    style.configure("TButton",
                    background=BG3, foreground=FG, bordercolor=BORDER,
                    focusthickness=0, padding=(8, 4))
    style.map("TButton",
              background=[("active", "#505050"), ("pressed", "#404040"), ("disabled", BG2)],
              foreground=[("disabled", "#666666")])

    style.configure("TEntry",
                    fieldbackground=BG2, foreground=FG, bordercolor=BORDER,
                    insertcolor=FG, selectbackground=SELECT_BG)
    style.map("TEntry",
              fieldbackground=[("disabled", BG2), ("readonly", BG2)],
              foreground=[("disabled", "#666666")])

    style.configure("TCombobox",
                    fieldbackground=BG2, background=BG3, foreground=FG,
                    bordercolor=BORDER, arrowcolor=FG,
                    selectbackground=SELECT_BG, selectforeground=FG)
    style.map("TCombobox",
              fieldbackground=[("readonly", BG2), ("disabled", BG2)],
              foreground=[("readonly", FG), ("disabled", "#666666")],
              selectbackground=[("readonly", BG2)],
              selectforeground=[("readonly", FG)],
              arrowcolor=[("disabled", "#666666")])

    style.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=(2, 5, 2, 0))
    style.configure("TNotebook.Tab",
                    background=BG2, foreground=FG_DIM, padding=(12, 6),
                    bordercolor=BORDER)
    style.map("TNotebook.Tab",
              background=[("selected", BG), ("active", BG3)],
              foreground=[("selected", FG), ("active", FG)])

    style.configure("TLabelframe", background=BG, bordercolor=BORDER)
    style.configure("TLabelframe.Label", background=BG, foreground=FG)

    style.configure("TScrollbar",
                    background=BG3, troughcolor=BG2, bordercolor=BG,
                    arrowcolor=FG_DIM, relief="flat")
    style.map("TScrollbar",
              background=[("active", "#606060"), ("pressed", "#707070")])

    style.configure("TProgressbar",
                    background=ACCENT, troughcolor=BG2, bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)

    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.map("TCheckbutton",
              background=[("active", BG)],
              indicatorcolor=[("selected", ACCENT), ("!selected", BG2)])

    style.configure("TSeparator", background=BORDER)


def build_root() -> tk.Tk:
    if DND_AVAILABLE:
        return TkinterDnD.Tk()
    return tk.Tk()


def main() -> None:
    # On first run on Windows, loading ctranslate2 (faster_whisper's backend) fires
    # a spurious SIGINT into the Python main thread, which closes the app.
    # Suppress SIGINT while the DLLs initialise, then restore normal Ctrl+C handling.
    _old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import faster_whisper  # noqa: F401 — pre-warms ctranslate2 DLL load
    except ImportError:
        pass
    signal.signal(signal.SIGINT, _old_sigint)

    try:
        root = build_root()
        _setup_dark_theme(root)
        TranscriptApp(root)
        root.mainloop()
    except Exception as exc:
        LOGGER.exception("Fatal application startup failure")
        try:
            messagebox.showerror(
                APP_TITLE,
                f"Application startup failed:\n{exc}\nLog file:\n{SESSION_LOG_PATH}",
            )
        except Exception:
            pass
        raise
