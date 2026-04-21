import faulthandler
import os
from pathlib import Path

# Unsafe workaround for duplicate OpenMP runtime issues seen on some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Write a native crash trace (segfault, illegal instruction, etc.) to a persistent file.
# This catches C-level crashes that bypass all Python exception handlers.
_crash_log_path = Path.home() / "LocalTranscriptLogs" / "crash.log"
_crash_log_path.parent.mkdir(parents=True, exist_ok=True)
_crash_log_file = open(_crash_log_path, "a")  # noqa: SIM115 — kept open for faulthandler lifetime
faulthandler.enable(file=_crash_log_file)

from src.gui import main

if __name__ == "__main__":
    main()
