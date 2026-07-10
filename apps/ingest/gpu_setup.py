"""NixOS GPU driver shim — import and call ``ensure_gpu_libs()`` BEFORE importing
torch / ultralytics / any CUDA-using library.

Why: the pip/uv torch wheel searches standard ``/usr/lib`` paths for ``libcuda.so``,
but on NixOS the driver lives at ``/run/opengl-driver/lib``. Without this, torch
silently reports ``cuda.is_available() == False`` even though the RTX 4050 and
``nvidia-smi`` work fine. We prepend the driver dir to ``LD_LIBRARY_PATH`` and
re-exec the process once so the loader picks it up.

No-op off NixOS (the driver path won't exist), so Windows/Fedora teammates get
GPU torch normally without this doing anything.
"""

import os
import sys

_DRIVER = "/run/opengl-driver/lib"


def ensure_gpu_libs() -> None:
    """Prepend the NixOS driver dir to LD_LIBRARY_PATH and re-exec once. No-op elsewhere."""
    if not os.path.exists(f"{_DRIVER}/libcuda.so"):
        return  # not NixOS (or no driver) → nothing to do
    if _DRIVER in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        return  # already set → avoid an infinite re-exec loop
    os.environ["LD_LIBRARY_PATH"] = f"{_DRIVER}:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable, *sys.argv])
