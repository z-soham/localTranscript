"""Self-check for the batch plumbing: run with `python3 test_batch.py`.

Covers the two pieces that fail silently if they regress — _BatchLogger swallowing
per-file done(), and _transcribe_one's URL-vs-path routing + temp-file cleanup.
The queue/loop wiring above them is Tk-bound and verified by running the app.
"""

import queue
import threading
from pathlib import Path

import src.gui as gui
from src.logging_setup import LOGGER


def test_batch_logger_swallows_done():
    q = queue.Queue()
    log = gui._BatchLogger(q, LOGGER)

    log.done(True, "Transcription completed successfully.")
    assert log.last_success is True
    log.done(False, "boom")
    assert log.last_success is False

    kinds = [q.get_nowait()[0] for _ in range(q.qsize())]
    assert "done" not in kinds, f"per-file done() leaked to the GUI: {kinds}"
    assert kinds == ["log", "log"], kinds


def test_transcribe_one_routes_local_path(tmp: Path):
    seen = {}
    orig = gui.transcribe_file
    gui.transcribe_file = lambda path, **kw: seen.update(path=path, kw=kw)
    try:
        gui._transcribe_one(str(tmp), gui._BatchLogger(queue.Queue(), LOGGER),
                            threading.Event(), {"engine": "parakeet"})
    finally:
        gui.transcribe_file = orig

    assert seen["path"] == tmp, seen
    assert seen["kw"]["engine"] == "parakeet", seen


def test_transcribe_one_downloads_and_cleans_up(tmp: Path):
    orig_dl, orig_tf = gui.download_youtube_audio, gui.transcribe_file
    gui.download_youtube_audio = lambda url, logger, stop: tmp
    gui.transcribe_file = lambda path, **kw: None
    try:
        gui._transcribe_one("https://youtu.be/abc123", gui._BatchLogger(queue.Queue(), LOGGER),
                            threading.Event(), {})
    finally:
        gui.download_youtube_audio, gui.transcribe_file = orig_dl, orig_tf

    assert not tmp.exists(), "temp download was not unlinked"


def test_transcribe_one_cleans_up_on_failure(tmp: Path):
    orig_dl, orig_tf = gui.download_youtube_audio, gui.transcribe_file
    gui.download_youtube_audio = lambda url, logger, stop: tmp

    def boom(path, **kw):
        raise RuntimeError("transcription blew up")

    gui.transcribe_file = boom
    try:
        gui._transcribe_one("https://youtu.be/abc123", gui._BatchLogger(queue.Queue(), LOGGER),
                            threading.Event(), {})
    except RuntimeError:
        pass  # the batch loop catches this and moves to the next item
    else:
        raise AssertionError("exception should propagate to the batch loop")
    finally:
        gui.download_youtube_audio, gui.transcribe_file = orig_dl, orig_tf

    assert not tmp.exists(), "temp download survived a failed transcription"


if __name__ == "__main__":
    import tempfile

    test_batch_logger_swallows_done()
    for check in (test_transcribe_one_routes_local_path,
                  test_transcribe_one_downloads_and_cleans_up,
                  test_transcribe_one_cleans_up_on_failure):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "clip.mp3"
            f.write_bytes(b"")
            check(f)
    print("all batch checks passed")
