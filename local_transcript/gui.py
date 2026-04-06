import os
import queue
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

from local_transcript.constants import APP_TITLE, MODEL_OPTIONS
from local_transcript.logging_setup import LOGGER, SESSION_LOG_PATH, QueueLogger
from local_transcript.transcriber import transcribe_file
from local_transcript.utils import seconds_to_human


class TranscriptApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("980x720")
        self.root.minsize(860, 620)

        self.queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.session_log_path = SESSION_LOG_PATH

        self.file_path_var = tk.StringVar()
        self.model_var = tk.StringVar(value="large-v3")
        self.device_pref_var = tk.StringVar(value="cuda")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_text_var = tk.StringVar(value="No active transcription")

        self._build_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        header = ttk.Frame(self.root, padding=12)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text=APP_TITLE, font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(
            header,
            text="Drag and drop a media file, or browse manually. Generates TXT and SRT outputs.",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        file_frame = ttk.LabelFrame(self.root, text="Input", padding=12)
        file_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        file_frame.columnconfigure(0, weight=1)

        self.drop_zone = tk.Label(
            file_frame,
            text=self._drop_zone_text(),
            relief="groove",
            borderwidth=2,
            padx=16,
            pady=22,
            bg="#1e1e1e",
            fg="#f0f0f0",
            font=("Segoe UI", 11),
            justify="center",
        )
        self.drop_zone.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        if DND_AVAILABLE:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)

        path_entry = ttk.Entry(file_frame, textvariable=self.file_path_var)
        path_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        browse_btn = ttk.Button(file_frame, text="Browse MP4 / Media", command=self.browse_file)
        browse_btn.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        clear_btn = ttk.Button(file_frame, text="Clear", command=self.clear_file)
        clear_btn.grid(row=1, column=2, sticky="ew")

        controls = ttk.LabelFrame(self.root, text="Transcription Settings", padding=12)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Model:").grid(row=0, column=0, sticky="w")
        model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=MODEL_OPTIONS,
            state="readonly",
            width=18,
        )
        model_combo.grid(row=0, column=1, sticky="w", padx=(8, 16))

        ttk.Label(controls, text="Preferred device:").grid(row=0, column=2, sticky="w")
        device_combo = ttk.Combobox(
            controls,
            textvariable=self.device_pref_var,
            values=["cuda", "cpu"],
            state="readonly",
            width=12,
        )
        device_combo.grid(row=0, column=3, sticky="w", padx=(8, 0))

        button_row = ttk.Frame(controls)
        button_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        button_row.columnconfigure(2, weight=1)

        self.start_btn = ttk.Button(button_row, text="Start Transcription", command=self.start_transcription)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_btn = ttk.Button(button_row, text="Stop", command=self.stop_transcription, state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.open_output_btn = ttk.Button(button_row, text="Open Output Folder", command=self.open_output_folder)
        self.open_output_btn.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=12)
        progress_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(2, weight=1)

        self.progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew")

        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_text_var)
        self.progress_label.grid(row=1, column=0, sticky="w", pady=(8, 12))

        console_frame = ttk.Frame(progress_frame)
        console_frame.grid(row=2, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console = tk.Text(console_frame, wrap="word", height=18, bg="#0f0f0f", fg="#d8d8d8")
        self.console.grid(row=0, column=0, sticky="nsew")
        self.console.configure(state="disabled")

        console_scroll = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        console_scroll.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=console_scroll.set)

        status_bar = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        status_bar.grid(row=4, column=0, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self._log("Application initialized.")
        self._log(f"Session log file: {self.session_log_path}")
        LOGGER.info("Application initialized")
        LOGGER.info("Session log file: %s", self.session_log_path)
        if not DND_AVAILABLE:
            self._log("Drag-and-drop support is disabled. Install tkinterdnd2 to enable it: pip install tkinterdnd2")
        from local_transcript.transcriber import WhisperModel
        if WhisperModel is None:
            self._log("faster-whisper is not installed. Install it with: pip install faster-whisper")

    def _drop_zone_text(self) -> str:
        if DND_AVAILABLE:
            return "Drag and drop a media file here\n(or use Browse MP4 / Media below)"
        return "Browse MP4 / Media below\n(Install tkinterdnd2 to enable drag and drop)"

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
        file_path = filedialog.askopenfilename(
            title="Select media file",
            filetypes=[
                ("Media files", "*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.m4a *.flac"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.file_path_var.set(file_path)
            self._log(f"Selected file: {file_path}")

    def clear_file(self) -> None:
        self.file_path_var.set("")
        self.progress["value"] = 0
        self.progress_text_var.set("No active transcription")
        self._log("Cleared selected file.")

    def _normalize_drop_path(self, raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1]
        return cleaned

    def _on_drop(self, event) -> None:
        raw_data = event.data
        if not raw_data:
            return

        paths = self.root.tk.splitlist(raw_data)
        if not paths:
            return

        file_path = self._normalize_drop_path(paths[0])
        self.file_path_var.set(file_path)
        self._log(f"Dropped file: {file_path}")

    def open_output_folder(self) -> None:
        raw_path = self.file_path_var.get().strip()
        if not raw_path:
            messagebox.showinfo(APP_TITLE, "Select a file first.")
            return

        file_path = Path(raw_path)
        folder = file_path.parent if file_path.exists() else Path.cwd()
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Could not open folder:\n{e}")

    def stop_transcription(self) -> None:
        self._stop_event.set()
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopping...")
        self._log("Stop requested — waiting for current segment to finish...")

    def start_transcription(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "A transcription is already running.")
            return

        raw_path = self.file_path_var.get().strip()
        if not raw_path:
            messagebox.showwarning(APP_TITLE, "Please select a media file first.")
            return

        input_path = Path(raw_path)
        if not input_path.exists():
            messagebox.showerror(APP_TITLE, f"File not found:\n{input_path}")
            return

        self._stop_event.clear()
        self.progress["value"] = 0
        self.progress_text_var.set("Starting transcription...")
        self._set_busy(True)

        model_name = self.model_var.get().strip() or "large-v3"
        prefer_cuda = self.device_pref_var.get().strip().lower() == "cuda"

        self._log("=" * 72)
        self._log(f"Starting transcription for: {input_path}")
        self._log(f"Selected model: {model_name}")
        self._log(f"Preferred device: {'CUDA' if prefer_cuda else 'CPU'}")

        logger = QueueLogger(self.queue, LOGGER)
        stop_event = self._stop_event

        def worker() -> None:
            try:
                transcribe_file(input_path, model_name, prefer_cuda, logger, stop_event)
            except Exception as e:
                logger.log("Error during transcription:")
                logger.log(str(e))
                stack = traceback.format_exc()
                logger.log(stack)
                LOGGER.exception("Unhandled transcription failure")
                logger.done(False, f"{e}\nLog file: {SESSION_LOG_PATH}")

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
            self._log(f"Log file saved to: {self.session_log_path}")
            messagebox.showinfo(APP_TITLE, f"{message}\nLog file:\n{self.session_log_path}")
        elif message == "cancelled":
            self.progress_text_var.set("Cancelled")
            self._log(f"Log file saved to: {self.session_log_path}")
            messagebox.showinfo(APP_TITLE, "Transcription was stopped.")
        else:
            self.progress_text_var.set("Failed")
            self._log(f"Failed: {message}")
            self._log(f"Detailed log file: {self.session_log_path}")
            messagebox.showerror(
                APP_TITLE,
                f"Transcription failed:\n{message}\nDetailed log file:\n{self.session_log_path}",
            )


def build_root() -> tk.Tk:
    if DND_AVAILABLE:
        return TkinterDnD.Tk()
    return tk.Tk()


def main() -> None:
    try:
        root = build_root()
        try:
            style = ttk.Style(root)
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass
        app = TranscriptApp(root)
        root.mainloop()
    except Exception as e:
        LOGGER.exception("Fatal application startup failure")
        try:
            messagebox.showerror(
                APP_TITLE,
                f"Application startup failed:\n{e}\nLog file:\n{SESSION_LOG_PATH}",
            )
        except Exception:
            pass
        raise
