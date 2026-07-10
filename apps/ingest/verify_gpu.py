"""Phase-0 GPU verification: assert CUDA is really available and print the device.

A CPU fallback must be LOUD, not silent (see plan.md GPU section) — so this exits
non-zero if torch can't see the GPU. Run:  uv run python verify_gpu.py
"""

import sys

from gpu_setup import ensure_gpu_libs

ensure_gpu_libs()  # MUST run before importing torch

import torch  # noqa: E402


def main() -> int:
    print(f"torch                 {torch.__version__}")
    print(f"torch.version.cuda    {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"cuda.is_available()   {available}")

    if not available:
        print("\nFAIL: CUDA is not available — refusing to fall back to CPU silently.", file=sys.stderr)
        print("On NixOS check that /run/opengl-driver/lib/libcuda.so exists and that", file=sys.stderr)
        print("gpu_setup.ensure_gpu_libs() ran before torch was imported.", file=sys.stderr)
        return 1

    idx = torch.cuda.current_device()
    print(f"device                cuda:{idx}")
    print(f"device name           {torch.cuda.get_device_name(idx)}")
    props = torch.cuda.get_device_properties(idx)
    print(f"total VRAM            {props.total_memory / 1024**3:.2f} GiB")

    # Tiny fp16 matmul on-device to prove kernels actually run, not just enumerate.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    y = (x @ x).sum().item()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"fp16 matmul ok        sum={y:.1f}  peak={peak:.1f} MiB")
    print("\nOK: GPU is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
