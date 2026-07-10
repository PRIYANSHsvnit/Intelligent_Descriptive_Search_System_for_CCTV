"""Stage 6a: SigLIP2 semantic_vector (search), GPU, fp16.

Locked model google/siglip2-so400m-patch14-224 → 1152-dim. Runs the image encoder on
each of a tracklet's K crops, mean-pools, L2-normalizes. The SAME model's text encoder
embeds the user's query at search time (Phase 2).

Writes vec/semantic.npy (N,1152) aligned to tracklets.json order.
Call gpu_setup.ensure_gpu_libs() BEFORE importing (torch).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from . import paths
from .cfg import SEMANTIC_DIM, SIGLIP_MODEL

_BATCH = 64


def _load_rgb(rel_key: str) -> np.ndarray | None:
    img = cv2.imread(str(paths.OUTPUT_ROOT / rel_key))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(scene: str, cam: str, device: int = 0) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())

    # flatten (tracklet_index, crop) so we batch across the whole camera
    idx: list[int] = []
    imgs: list[np.ndarray] = []
    for i, t in enumerate(tracklets):
        for k in t["crop_refs"]:
            im = _load_rgb(k)
            if im is not None:
                idx.append(i)
                imgs.append(im)

    vecs = np.zeros((len(tracklets), SEMANTIC_DIM), dtype=np.float32)
    if not imgs:
        np.save(out / "vec" / "semantic.npy", vecs)
        return {"cam": cam, "tracklets": len(tracklets), "crops": 0}

    dev = f"cuda:{device}"
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to(dev).eval()

    sums = np.zeros((len(tracklets), SEMANTIC_DIM), dtype=np.float32)
    counts = np.zeros(len(tracklets), dtype=np.int32)
    with torch.no_grad():
        for b in range(0, len(imgs), _BATCH):
            batch = imgs[b:b + _BATCH]
            pv = processor(images=batch, return_tensors="pt").pixel_values.to(dev, torch.float16)
            feats = model.get_image_features(pixel_values=pv)
            if not isinstance(feats, torch.Tensor):  # transformers 5.x returns an output object
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy()
            for j, ti in enumerate(idx[b:b + _BATCH]):
                sums[ti] += feats[j]
                counts[ti] += 1

    (out / "vec").mkdir(exist_ok=True)
    for i in range(len(tracklets)):
        if counts[i]:
            v = sums[i] / counts[i]
            n = np.linalg.norm(v)
            vecs[i] = v / n if n > 0 else v
    np.save(out / "vec" / "semantic.npy", vecs)

    peak = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()
    return {"cam": cam, "tracklets": len(tracklets), "crops": len(imgs), "vram_peak_mb": round(peak, 1)}
