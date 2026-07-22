"""Stage 6a: SigLIP2 semantic_vector (search), GPU, fp16.

Locked model google/siglip2-so400m-patch14-224 → 1152-dim. Runs the image encoder on
each of a tracklet's K crops and preserves every normalized crop vector for multi-view
retrieval. The normalized mean remains as a compatibility fallback. The SAME model's
text encoder embeds the user's query at search time (Phase 2).

Writes vec/semantic.npy (N,1152) aligned to tracklets.json order.
Writes vec/semantic_crops.npz with vectors + tracklet/crop indices.
Call gpu_setup.ensure_gpu_libs() BEFORE importing (torch).
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from . import paths
from .cfg import SEMANTIC_DIM, SIGLIP_MODEL

_BATCH = int(os.environ.get("SIGLIP_BATCH", "64"))


def _load_rgb(rel_key: str) -> np.ndarray | None:
    img = cv2.imread(str(paths.OUTPUT_ROOT / rel_key))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(scene: str, cam: str, device: int = 0) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())

    # Flatten metadata only. Images are decoded one batch at a time; holding every crop
    # from a dense camera in RAM caused multi-GB host-memory spikes.
    refs: list[tuple[int, int, str]] = []
    for i, t in enumerate(tracklets):
        for ci, key in enumerate(t["crop_refs"]):
            refs.append((i, ci, key))

    (out / "vec").mkdir(exist_ok=True)
    vecs = np.zeros((len(tracklets), SEMANTIC_DIM), dtype=np.float32)
    if not refs:
        np.save(out / "vec" / "semantic.npy", vecs)
        np.savez(
            out / "vec" / "semantic_crops.npz",
            vectors=np.zeros((0, SEMANTIC_DIM), dtype=np.float32),
            tracklet_indices=np.zeros(0, dtype=np.int32),
            crop_indices=np.zeros(0, dtype=np.int16),
        )
        return {"cam": cam, "tracklets": len(tracklets), "crops": 0}

    dev = f"cuda:{device}"
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to(dev).eval()

    sums = np.zeros((len(tracklets), SEMANTIC_DIM), dtype=np.float32)
    counts = np.zeros(len(tracklets), dtype=np.int32)
    crop_vec_batches: list[np.ndarray] = []
    valid_idx: list[int] = []
    valid_crop_idx: list[int] = []
    with torch.no_grad():
        for b in range(0, len(refs), _BATCH):
            batch_meta = []
            batch = []
            for ti, ci, key in refs[b:b + _BATCH]:
                image = _load_rgb(key)
                if image is not None:
                    batch_meta.append((ti, ci))
                    batch.append(image)
            if not batch:
                continue
            pv = processor(images=batch, return_tensors="pt").pixel_values.to(dev, torch.float16)
            feats = model.get_image_features(pixel_values=pv)
            if not isinstance(feats, torch.Tensor):  # transformers 5.x returns an output object
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy()
            crop_vec_batches.append(feats)
            for j, (ti, ci) in enumerate(batch_meta):
                sums[ti] += feats[j]
                counts[ti] += 1
                valid_idx.append(ti)
                valid_crop_idx.append(ci)

    for i in range(len(tracklets)):
        if counts[i]:
            v = sums[i] / counts[i]
            n = np.linalg.norm(v)
            vecs[i] = v / n if n > 0 else v
    np.save(out / "vec" / "semantic.npy", vecs)
    crop_vecs = (np.concatenate(crop_vec_batches, axis=0) if crop_vec_batches
                 else np.zeros((0, SEMANTIC_DIM), dtype=np.float32))
    np.savez(
        out / "vec" / "semantic_crops.npz",
        vectors=crop_vecs,
        tracklet_indices=np.asarray(valid_idx, dtype=np.int32),
        crop_indices=np.asarray(valid_crop_idx, dtype=np.int16),
    )

    peak = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()
    return {"cam": cam, "tracklets": len(tracklets), "crops": len(valid_idx),
            "vram_peak_mb": round(peak, 1)}
