import os
import sys
from pathlib import Path

from src.logging_setup import QueueLogger

_CUDNN_BIN = [
    Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
    Path(sys.base_prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
]

_CUDA_LIB = [
    Path(sys.prefix) / "Library" / "bin",
    Path(sys.base_prefix) / "Library" / "bin",
]


def locate_cudnn_hint() -> str | None:
    if sys.platform != "win32":
        return None
    for candidate in _CUDNN_BIN:
        if (candidate / "cudnn_ops64_9.dll").exists():
            return str(candidate)
    return None


def preload_cuda_paths(logger: QueueLogger) -> None:
    if sys.platform != "win32":
        return
    cuda_path_entries = []
    for candidate in _CUDNN_BIN + _CUDA_LIB:
        if candidate.exists():
            cuda_path_entries.append(str(candidate))
            try:
                os.add_dll_directory(str(candidate))
            except (AttributeError, FileNotFoundError, OSError):
                pass

    if cuda_path_entries:
        existing_path = os.environ.get("PATH", "")
        prepend = os.pathsep.join(cuda_path_entries)
        os.environ["PATH"] = prepend + os.pathsep + existing_path if existing_path else prepend
        logger.log("Augmented DLL search path with likely CUDA/cuDNN directories.")
        for entry in cuda_path_entries:
            logger.log(f"  DLL path: {entry}")


def should_force_cpu_after_cuda_error(error_text: str) -> bool:
    cuda_error_markers = [
        "cudnn_ops64_9.dll",
        "cudnnCreateTensorDescriptor",
        "cuda",
        "cublas",
        "curand",
        "cudart",
        "ctranslate2",
        "invalid handle",
        "dll load failed",
    ]
    lowered = error_text.lower()
    return any(marker.lower() in lowered for marker in cuda_error_markers)
