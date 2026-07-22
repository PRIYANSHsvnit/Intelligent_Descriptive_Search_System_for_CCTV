"""Throwaway VRAM probe: how much VRAM does the LIVE tier's co-resident model set
actually take? Loads the india-profile detectors (vehicle YOLOv11-X @640 +
CrowdHuman YOLOv8n @1280) and SigLIP2 together — the exact way the pipeline loads
them — runs a warmup inference on each at its real imgsz, and reports peak VRAM.

reid is CPU-only (plain onnxruntime) so it is NOT in the GPU budget. The VLM is
offline-tier only, so it is deliberately excluded here.

Two numbers are reported per step:
  torch_alloc / torch_reserved : torch's OWN allocator (misses CUDA ctx, cudnn/
                                 cublas workspaces, ultralytics overhead).
  nvidia-smi_proc              : this process's TOTAL device memory — the real
                                 "does it fit in 6 GB" number.

Run from apps/ingest:  uv run python vram_probe.py
"""

import os
import sys

# GPU shim MUST run before torch/ultralytics import (re-execs once on NixOS).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_setup import ensure_gpu_libs  # noqa: E402

ensure_gpu_libs()

import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from constants import SIGLIP_MODEL  # noqa: E402
from pipeline.profiles import PROFILES  # noqa: E402


def smi_proc_mb() -> int | None:
    """This process's total VRAM per nvidia-smi (incl CUDA context). None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(os.getpid()):
            return int(parts[1])
    return 0  # our pid not listed yet = nothing allocated


def report(label: str) -> None:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    smi = smi_proc_mb()
    smi_s = f"{smi:5d}MB" if smi is not None else "  n/a "
    print(f"  {label:<34} torch_alloc={alloc:6.0f}MB  reserved={reserved:6.0f}MB  "
          f"nvidia-smi_proc={smi_s}")


def rand_img(h: int, w: int) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available after gpu_setup shim — aborting.", file=sys.stderr)
        sys.exit(1)

    dev = torch.device("cuda:0")
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={total_mb:.0f}MB\n")

    prof = PROFILES["india"]
    pdet = prof.person_detector

    # 0) CUDA context baseline (one tiny alloc forces context creation)
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report("0) CUDA context baseline")

    # 1) Vehicle detector: UVH-26 YOLOv11-X @ 640
    from ultralytics import YOLO
    veh = YOLO(prof.yolo_model)
    src = rand_img(1080, 1920)
    for _ in range(2):  # first pass includes lazy init/warmup
        veh.predict(src, imgsz=prof.yolo_imgsz, device=0, verbose=False)
    report(f"1) + vehicle YOLOv11-X @{prof.yolo_imgsz}")

    # 2) Person detector: CrowdHuman YOLOv8n @ 1280 (nano weights, big activations)
    per = YOLO(pdet.weights)
    for _ in range(2):
        per.predict(src, imgsz=pdet.imgsz, device=0, verbose=False)
    report(f"2) + person YOLOv8n @{pdet.imgsz}")

    # 3) SigLIP2 image tower (full AutoModel, fp16) — same load as embed_siglip.py
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to(dev).eval()
    batch = [Image.fromarray(rand_img(256, 128)) for _ in range(8)]  # crop-sized, batch 8
    with torch.no_grad():
        for _ in range(2):
            pv = processor(images=batch, return_tensors="pt").pixel_values.to(dev, torch.float16)
            model.get_image_features(pixel_values=pv)
    report("3) + SigLIP2 so400m (fp16, batch 8)")

    # Peak across the whole run
    print()
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    smi_final = smi_proc_mb()
    print(f"PEAK torch_alloc   = {peak_alloc:6.0f}MB")
    print(f"PEAK torch_reserved= {peak_reserved:6.0f}MB")
    if smi_final is not None:
        print(f"FINAL nvidia-smi   = {smi_final:6d}MB  (real per-process total incl CUDA ctx)")
        headroom = 6144 - smi_final
        print(f"\nvs 6144MB card: {'FITS' if headroom > 0 else 'OVER'} "
              f"({headroom:+d}MB headroom, before desktop/display overhead ~300-800MB)")


if __name__ == "__main__":
    main()
