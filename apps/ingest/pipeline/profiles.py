"""Domain profiles: the small set of knobs that differ between deployment domains.

Everything expensive in the pipeline — BoT-SORT tracking, SigLIP embed, SigLIP color,
the matcher, the DB, the search API, the UI, the evaluator, and the LOCKED vector dims
(SigLIP 1152 / re-ID 2048 / color 56) — is domain-agnostic and SHARED. Only three
things actually differ between CityFlow (US highway footage) and Indian urban CCTV:

  * detector    : YOLO weights + inference params + which class ids to keep
  * class_map   : detector class id -> (subtype, entity_type)   [majority-voted per track]
  * reid_onnx   : the re-ID appearance encoder (MUST output the locked 2048-d)
  * footage_dir : where this domain's source videos live under footage_data/

Select with the INGEST_PROFILE env var (default "cityflow") or run_ingest --profile.
CityFlow is the default so existing runs and the S01 eval baseline are unchanged; it is
also the regression bench (the only footage with ground truth / an IDF1 number).

Both profiles must emit the same locked dims, so the DB / matcher / search / UI never
fork — only detect + embed_reid select a knob here.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# constants.py (Phase-0 locked dims + the CityFlow detector defaults) is the parent dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import (  # noqa: E402
    YOLO_CLASSES as _CF_YOLO_CLASSES,
    YOLO_CONF as _CF_YOLO_CONF,
    YOLO_IMGSZ as _CF_YOLO_IMGSZ,
    YOLO_MODEL as _CF_YOLO_MODEL,
)


@dataclass(frozen=True)
class Profile:
    name: str
    yolo_model: str                              # weights path/name for ultralytics.YOLO(...)
    yolo_imgsz: int
    yolo_conf: float
    yolo_classes: tuple[int, ...] | None         # None = keep every class the model emits
    tracker: str
    class_map: dict[int, tuple[str, str]]        # class id -> (subtype, entity_type)
    default_subtype: str                         # for class ids not in class_map
    reid_onnx: str                               # filename under models/ (must be 2048-d)
    footage_dir: str                             # subdir under footage_data/ holding scenes


# CityFlow: stock YOLO11m on COCO ids (person/bicycle/car/motorcycle/bus/truck). This map
# reproduces the old cfg.coco_subtype exactly (id 0 is the only 'person' entity).
_CITYFLOW_CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("person", "person"),
    1: ("bicycle", "vehicle"),
    2: ("car", "vehicle"),
    3: ("motorcycle", "vehicle"),
    5: ("bus", "vehicle"),
    7: ("truck", "vehicle"),
}

# India: UVH-26 / VehicleNet 14-class taxonomy (all 'vehicle' entities). The ids below are
# the published class order; CONFIRM against the downloaded model's names before the first
# India run, and swap reid_onnx once an India-tuned encoder exists (still 2048-d).
_INDIA_CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("hatchback", "vehicle"),
    1: ("sedan", "vehicle"),
    2: ("suv", "vehicle"),
    3: ("muv", "vehicle"),
    4: ("bus", "vehicle"),
    5: ("truck", "vehicle"),
    6: ("three_wheeler", "vehicle"),   # auto-rickshaw
    7: ("two_wheeler", "vehicle"),
    8: ("lcv", "vehicle"),
    9: ("mini_bus", "vehicle"),
    10: ("tempo_traveller", "vehicle"),
    11: ("bicycle", "vehicle"),
    12: ("van", "vehicle"),
    13: ("other", "vehicle"),
}


PROFILES: dict[str, Profile] = {
    "cityflow": Profile(
        name="cityflow",
        yolo_model=_CF_YOLO_MODEL,
        yolo_imgsz=_CF_YOLO_IMGSZ,
        yolo_conf=_CF_YOLO_CONF,
        yolo_classes=tuple(_CF_YOLO_CLASSES),
        tracker="botsort.yaml",
        class_map=_CITYFLOW_CLASS_MAP,
        default_subtype="car",
        reid_onnx="veri_reid.onnx",
        footage_dir="train",
    ),
    # PLACEHOLDER — not runnable until the UVH-26 model is downloaded to models/ and the
    # class ids are confirmed. Kept here so the seam exists and CityFlow stays the default.
    "india": Profile(
        name="india",
        yolo_model="models/vehiclenet_y26m.pt",   # UVH-26 VehicleNet (YOLOv11); download first
        yolo_imgsz=_CF_YOLO_IMGSZ,
        yolo_conf=_CF_YOLO_CONF,
        yolo_classes=None,                          # keep all 14 India classes
        tracker="botsort.yaml",
        class_map=_INDIA_CLASS_MAP,
        default_subtype="other",
        reid_onnx="veri_reid.onnx",                 # until an India-tuned encoder is trained
        footage_dir="india",
    ),
}


_ACTIVE: Profile | None = None


def active() -> Profile:
    """The selected profile (INGEST_PROFILE env, default 'cityflow'). Resolved once."""
    global _ACTIVE
    if _ACTIVE is None:
        name = os.environ.get("INGEST_PROFILE", "cityflow")
        if name not in PROFILES:
            raise KeyError(f"unknown INGEST_PROFILE {name!r}; have {list(PROFILES)}")
        _ACTIVE = PROFILES[name]
    return _ACTIVE


def use(name: str) -> Profile:
    """Explicitly select the active profile (entry points call this before doing work)."""
    global _ACTIVE
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; have {list(PROFILES)}")
    _ACTIVE = PROFILES[name]
    os.environ["INGEST_PROFILE"] = name          # so subprocesses / late imports agree
    return _ACTIVE


def subtype_vote(cls_ids) -> tuple[str, str]:
    """Majority-vote a track's class ids -> (subtype, entity_type) via the active profile."""
    p = active()
    top = Counter(int(c) for c in cls_ids).most_common(1)[0][0]
    return p.class_map.get(top, (p.default_subtype, "vehicle"))
