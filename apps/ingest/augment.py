"""augment.py — synthetic "second camera" transforms for the re-ID robustness eval.

No multi-camera Surat footage was provided, so we manufacture a plausible *camera B* by
degrading camera A's pixels the way a genuinely different camera would: viewpoint
(perspective), sensor/ISP (white-balance + tone), optics (defocus blur), resolution
(downscale->upscale) and compression (JPEG). The SAME recipe is applied to crops (for the
metric, via embed_reid_dino.py --aug) and to full frames (for the visual, via render_camB.py),
so the rendered clip faithfully shows the degradation the embedded crops experienced.

Deterministic by design: the perspective is a FIXED homography (camera B has one fixed mounting
angle) and every other step is pointwise, so re-running yields bit-identical pixels — the A/B is
reproducible. A degraded copy shares camera A's background and pose, so any cross-view score it
yields is an UPPER BOUND on real cross-camera re-ID; report it as "synthetic cross-camera".

Pipeline order (scene -> camera): perspective -> white-balance -> tone -> blur -> resolution -> JPEG.
"""
from __future__ import annotations

import cv2
import numpy as np


def _hflip(img: np.ndarray, on) -> np.ndarray:
    """Horizontal mirror. The literal 'flip the clip' instruction; DINOv3 is trained with random
    horizontal flip, so its fingerprint is ~invariant to this by design (model card) — we run it to
    demonstrate that, not because it should change anything."""
    return cv2.flip(img, 1) if on else img


def _perspective(img: np.ndarray, s: float) -> np.ndarray:
    """Fixed rotate-in-depth homography (~s of the frame): a mild change of viewing angle.
    BORDER_REFLECT so the warp introduces no black wedges (those would be a giveaway cue)."""
    if s <= 0:
        return img
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx, dy = s * w, s * h * 0.5
    dst = np.float32([[dx, dy], [w - dx * 0.3, 0.0], [w - dx, h - dy], [dx * 0.3, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


def _white_balance(img: np.ndarray, rgain: float, bgain: float) -> np.ndarray:
    """Per-channel gain = a different sensor / auto white-balance (rgain>1,bgain<1 = warmer)."""
    if rgain == 1.0 and bgain == 1.0:
        return img
    out = img.astype(np.float32)
    out[..., 2] *= rgain          # BGR: red
    out[..., 0] *= bgain          # BGR: blue
    return np.clip(out, 0, 255).astype(np.uint8)


def _tone(img: np.ndarray, bright: float, contrast: float) -> np.ndarray:
    """Brightness offset + contrast about mid-grey = a different auto-exposure curve."""
    if bright == 0.0 and contrast == 1.0:
        return img
    out = (img.astype(np.float32) - 128.0) * contrast + 128.0 + bright
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian defocus. Kernel spans +-3 sigma (odd)."""
    if sigma <= 0:
        return img
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def _resolution(img: np.ndarray, factor: float) -> np.ndarray:
    """Downscale by `factor` (area) then upscale back (linear) = lost sensor resolution.
    Applied at the crop's native size *before* the encoder's 320px resize, so detail is
    genuinely destroyed rather than merely resampled."""
    if factor >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg(img: np.ndarray, quality: float) -> np.ndarray:
    """Lossy JPEG round-trip = the camera's video encoder (blocking + chroma loss)."""
    if quality >= 100 or quality <= 0:
        return img
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


# Combined "camera B" recipe for experiment A (one moderate, plausible different camera).
# Single-knob dicts for the experiment-B degradation sweeps live in KNOBS.
PRESETS: dict[str, dict] = {
    "camB": dict(perspective=0.08, rgain=1.15, bgain=0.90, bright=8.0, contrast=1.05,
                 blur=1.0, resolution=0.5, jpeg=40),
    # the literal "flip the clip" instruction, isolated:
    "hflip": dict(hflip=1),
    # camB + the horizontal flip on top (realistic second camera, flipped as instructed):
    "camBflip": dict(hflip=1, perspective=0.08, rgain=1.15, bgain=0.90, bright=8.0, contrast=1.05,
                     blur=1.0, resolution=0.5, jpeg=40),
    "identity": dict(),
}

# Baselines a single sweep holds fixed while it cranks one axis (experiment B).
KNOBS = ("perspective", "rgain", "bgain", "bright", "contrast", "blur", "resolution", "jpeg")


def apply(img: np.ndarray, spec: dict) -> np.ndarray:
    img = _hflip(img, spec.get("hflip", 0))
    img = _perspective(img, spec.get("perspective", 0.0))
    img = _white_balance(img, spec.get("rgain", 1.0), spec.get("bgain", 1.0))
    img = _tone(img, spec.get("bright", 0.0), spec.get("contrast", 1.0))
    img = _blur(img, spec.get("blur", 0.0))
    img = _resolution(img, spec.get("resolution", 1.0))
    img = _jpeg(img, spec.get("jpeg", 100))
    return img


def transform_for(name: str | None = None, **overrides):
    """Return (spec, fn). `name` selects a preset; `overrides` set/replace individual knobs
    (for experiment-B sweeps, e.g. transform_for("identity", jpeg=30))."""
    spec = dict(PRESETS.get(name, {})) if name else {}
    spec.update({k: v for k, v in overrides.items() if v is not None})
    return spec, (lambda im: apply(im, spec))
