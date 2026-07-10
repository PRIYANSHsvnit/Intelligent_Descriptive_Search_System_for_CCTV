"""Stage 5 (color): per-tracklet HSV color signature (56-dim) + a coarse color name.

Reads the K best crops written by detect_track, writes:
  - vec/reid_color.npy  (N,56) aligned to tracklets.json order
  - `color` field added to each tracklet in tracklets.json

The 56-dim signature (12x4 Hue x Sat + 8 Value) is the structured color signal the
appearance model binds loosely; it's fused with reid_appearance at match time and
also drives the (soft, never-hard-filter) `color` column.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from . import paths
from .cfg import COLOR_HS_BINS, COLOR_V_BINS, REID_COLOR_DIM, dominant_color_name


def _central(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    return crop_bgr[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]


def color_signature(crop_bgr: np.ndarray) -> np.ndarray:
    """56-dim Hellinger-normalized HSV signature for one crop."""
    region = _central(crop_bgr)
    if region.size == 0:
        region = crop_bgr
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, list(COLOR_HS_BINS), [0, 180, 0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [COLOR_V_BINS], [0, 256]).flatten()
    sig = np.concatenate([hs, v]).astype(np.float32)          # 48 + 8 = 56
    s = sig.sum()
    if s > 0:
        sig /= s                                              # sum-1
    sig = np.sqrt(sig)                                        # Hellinger
    n = np.linalg.norm(sig)
    if n > 0:
        sig /= n                                              # L2
    return sig


def _color_name(crop_bgr: np.ndarray) -> str:
    region = _central(crop_bgr)
    if region.size == 0:
        region = crop_bgr
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_med = float(np.median(hsv[:, 0]))
    s_med = float(np.median(hsv[:, 1]))
    v_med = float(np.median(hsv[:, 2]))
    return dominant_color_name(h_med, s_med, v_med)


def _load_crop(rel_key: str) -> np.ndarray | None:
    img = cv2.imread(str(paths.OUTPUT_ROOT / rel_key))
    return img


def run(scene: str, cam: str) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())

    sigs = np.zeros((len(tracklets), REID_COLOR_DIM), dtype=np.float32)
    named = 0
    for i, t in enumerate(tracklets):
        crops = [c for c in (_load_crop(k) for k in t["crop_refs"]) if c is not None]
        if not crops:
            t["color"] = None
            continue
        per_crop = np.stack([color_signature(c) for c in crops])
        mean_sig = per_crop.mean(axis=0)
        n = np.linalg.norm(mean_sig)
        if n > 0:
            mean_sig /= n
        sigs[i] = mean_sig
        t["color"] = _color_name(crops[0])                    # name from the best crop
        named += 1

    (out / "vec").mkdir(exist_ok=True)
    np.save(out / "vec" / "reid_color.npy", sigs)
    (out / "tracklets.json").write_text(json.dumps(tracklets, indent=2))
    return {"cam": cam, "tracklets": len(tracklets), "colored": named}
