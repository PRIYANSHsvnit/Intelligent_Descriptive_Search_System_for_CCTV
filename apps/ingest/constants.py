"""Locked model + vector-dimension constants for the whole pipeline.

These are frozen in Phase 0 and the Postgres schema hardcodes the dims. Switching
any of these later means re-embedding every vector, so DO NOT change them.
"""

# --- Semantic search vector (SigLIP2), runs on GPU ---------------------------
# Locked to the so400m variant (over base/768) because search is the headline
# demo and so400m retrieves noticeably better; it fits the 6 GB RTX 4050 since it
# is the only model resident during the SigLIP pass, and the +50% index size is
# trivial at a few-thousand rows.
# NOTE: so400m has no patch16-224 variant on HF; patch14-224 is the smallest so400m
# at 224px input and is still 1152-dim, so the locked schema dim is unaffected.
SIGLIP_MODEL = "google/siglip2-so400m-patch14-224"
SEMANTIC_DIM = 1152

# --- Re-ID appearance vector (VeRi-trained FastReID ResNet50-IBN), runs on CPU ---
# Dim is fixed by the encoder (BNNeck 2048-d feature).
REID_APPEARANCE_DIM = 2048

# --- HSV color signature, fused with appearance at match time ----------------
# 12x4 Hue x Saturation (48) + 8 Value bins = 56.
REID_COLOR_DIM = 56

# --- YOLO detector ------------------------------------------------------------
YOLO_MODEL = "yolo11m.pt"
YOLO_IMGSZ = 1280
YOLO_CONF = 0.3
# COCO class ids we keep: person, bicycle, car, motorcycle, bus, truck
YOLO_CLASSES = [0, 1, 2, 3, 5, 7]
