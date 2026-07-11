"""Stage 6b: VeRi re-ID appearance vector (tracing), CPU via plain onnxruntime.

Locked encoder: FastReID VeRi ResNet50-IBN (SBS), 2048-dim → reid_appearance. Runs on
CPU by design (offline batch pass over K crops only; plain onnxruntime has no CUDA deps,
sidestepping the NixOS CUDA discovery pain — see plan.md [6]).

Requires the exported model at models/veri_reid.onnx (see reid/export_onnx.py). If it's
missing this pass SKIPS (reid_appearance stays NULL); everything else still works.

Writes vec/reid_appearance.npy (N,2048) aligned to tracklets.json order.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from . import paths, profiles
from .cfg import REID_APPEARANCE_DIM, REID_INPUT_HW

_BATCH = 64


def model_path():
    """Encoder ONNX for the active profile (must output the locked 2048-d)."""
    return paths.INGEST_ROOT / "models" / profiles.active().reid_onnx


def available() -> bool:
    return model_path().exists()


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    # ONNX bakes in pixel_mean/std normalization → feed RAW 0-255 RGB, CHW.
    h, w = REID_INPUT_HW
    img = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img.transpose(2, 0, 1)  # CHW


def run(scene: str, cam: str) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    mp = model_path()
    if not mp.exists():
        return {"cam": cam, "skipped": f"{mp.name} missing", "tracklets": n}

    import onnxruntime as ort

    sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    idx: list[int] = []
    imgs: list[np.ndarray] = []
    for i, t in enumerate(tracklets):
        for k in t["crop_refs"]:
            im = cv2.imread(str(paths.OUTPUT_ROOT / k))
            if im is not None:
                idx.append(i)
                imgs.append(_preprocess(im))

    sums = np.zeros((n, REID_APPEARANCE_DIM), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    for b in range(0, len(imgs), _BATCH):
        batch = np.stack(imgs[b:b + _BATCH]).astype(np.float32)
        feats = sess.run([out_name], {in_name: batch})[0]
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        for j, ti in enumerate(idx[b:b + _BATCH]):
            sums[ti] += feats[j]
            counts[ti] += 1

    vecs = np.zeros((n, REID_APPEARANCE_DIM), dtype=np.float32)
    for i in range(n):
        if counts[i]:
            v = sums[i] / counts[i]
            vecs[i] = v / (np.linalg.norm(v) + 1e-12)
    (out / "vec").mkdir(exist_ok=True)
    np.save(out / "vec" / "reid_appearance.npy", vecs)
    return {"cam": cam, "tracklets": n, "crops": len(imgs)}
