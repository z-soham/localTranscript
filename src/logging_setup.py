import logging
import queue
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.constants import LOG_DIR


def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"local_transcript_{timestamp}.log"

    logger = logging.getLogger("local_transcript_gui")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


LOGGER, SESSION_LOG_PATH = setup_logging()


class TranscriptionError(Exception):
    pass


class QueueLogger:
    def __init__(self, out_queue: queue.Queue, logger: logging.Logger):
        self.out_queue = out_queue
        self.logger = logger

    def log(self, message: str) -> None:
        self.logger.info(message)
        self.out_queue.put(("log", message))

    def progress(
        self,
        progress: float,
        processed_audio_sec: float,
        total_audio_sec: float | None,
        elapsed_wall_sec: float,
        eta_sec: float | None,
        speed_x: float | None,
    ) -> None:
        self.out_queue.put((
            "progress",
            {
                "progress": progress,
                "processed_audio_sec": processed_audio_sec,
                "total_audio_sec": total_audio_sec,
                "elapsed_wall_sec": elapsed_wall_sec,
                "eta_sec": eta_sec,
                "speed_x": speed_x,
            },
        ))

    def done(self, success: bool, message: str) -> None:
        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)
        self.out_queue.put(("done", {"success": success, "message": message}))
