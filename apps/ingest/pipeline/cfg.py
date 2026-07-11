"""Single config surface for the pipeline package: re-exports the Phase-0 locked
constants (apps/ingest/constants.py) and adds pipeline-specific parameters.

Importing this guarantees the top-level ``constants`` module is on sys.path
regardless of which entry point ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402  (re-export locked Phase-0 constants)
    REID_APPEARANCE_DIM,
    REID_COLOR_DIM,
    SEMANTIC_DIM,
    SIGLIP_MODEL,
    YOLO_CLASSES,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL,
)

__all__ = [
    "REID_APPEARANCE_DIM",
    "REID_COLOR_DIM",
    "SEMANTIC_DIM",
    "SIGLIP_MODEL",
    "YOLO_CLASSES",
    "YOLO_CONF",
    "YOLO_IMGSZ",
    "YOLO_MODEL",
    "K_CROPS",
    "COLOR_HS_BINS",
    "COLOR_V_BINS",
    "COLOR_VOCAB",
    "COLOR_PROMPT_TEMPLATES",
    "COLOR_PROMPT_NOUNS",
    "COLOR_MIN_MARGIN",
    "REID_INPUT_HW",
    "REID_MEAN",
    "REID_STD",
    "coco_subtype",
    "dominant_color_name",
]

# --- crop selection ----------------------------------------------------------
K_CROPS = 3

# --- HSV color signature (56-dim): 12x4 Hue x Sat (48) + 8 Value (8) ---------
COLOR_HS_BINS = (12, 4)
COLOR_V_BINS = 8

# --- SigLIP zero-shot color name (pipeline/color_siglip.py) -------------------
# Replaces the brittle HSV `dominant_color_name` for the display-only `color`
# column. Pragmatic vehicle palette (color is never a hard filter, so we favor
# common, visually-distinct names over exhaustive coverage).
COLOR_VOCAB = (
    "white", "black", "gray", "silver",
    "red", "blue", "green", "yellow", "orange", "brown",
)
# Prompt ensemble: every (template x noun) is embedded, then averaged per color.
COLOR_PROMPT_TEMPLATES = (
    "a photo of a {color} {noun}",
    "a {color} {noun}",
    "a {color}-colored {noun}",
)
COLOR_PROMPT_NOUNS = ("car", "vehicle", "truck")
# Min cosine margin between the top-2 colors to commit a label; below it the
# tracklet is left uncolored (None) rather than mislabeled. 0.0 = always label.
COLOR_MIN_MARGIN = 0.0

# --- VeRi FastReID preprocessing --------------------------------------------
# The exported ONNX BAKES IN normalization (pixel_mean/std are traced into the graph),
# so the pipeline feeds RAW 0-255 RGB resized to 256x256 (no external normalization).
# These constants (from the checkpoint's pixel_mean/std, 0-255 scale) are kept for
# reference / an alternate export that expects pre-normalized input.
REID_INPUT_HW = (256, 256)              # (H, W)
REID_MEAN = (123.675, 116.28, 103.53)   # 0-255 scale, RGB
REID_STD = (58.395, 57.12, 57.375)

# COCO id → our subtype. entity_type = 'person' iff class 0 else 'vehicle'.
_COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def coco_subtype(cls_ids) -> tuple[str, str]:
    """Majority-vote a track's class ids → (subtype, entity_type)."""
    from collections import Counter

    top = Counter(int(c) for c in cls_ids).most_common(1)[0][0]
    subtype = _COCO.get(top, "car")
    entity_type = "person" if top == 0 else "vehicle"
    return subtype, entity_type


# Coarse human color names from a dominant HSV bin (for the `color` column /
# soft signal only — never a hard filter). H in [0,180] (OpenCV), S,V in [0,255].
def dominant_color_name(h: float, s: float, v: float) -> str:
    if v < 40:
        return "black"
    if s < 40:
        return "white" if v > 200 else ("silver" if v > 120 else "gray")
    if h < 10 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 33:
        return "yellow"
    if h < 78:
        return "green"
    if h < 100:
        return "cyan"
    if h < 131:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"
