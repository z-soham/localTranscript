import os

# Unsafe workaround for duplicate OpenMP runtime issues seen on some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from local_transcript.gui import main

if __name__ == "__main__":
    main()
