"""Domain profiles: the small set of knobs that differ between deployment domains.

Everything expensive in the pipeline — BoT-SORT tracking, SigLIP embed, SigLIP color,
the matcher, the DB, the search API, the UI, the evaluator, and the LOCKED vector dims
(SigLIP 1152 / re-ID 2048 / color 56) — is domain-agnostic and SHARED. Only three
things actually differ between CityFlow (US highway footage) and Indian urban CCTV:

  * detector    : YOLO weights + inference params + which class ids to keep
  * class_map   : detector class id -> (subtype, entity_type)   [majority-voted per track]
  * reid_onnx   : the re-ID appearance encoder (MUST output the locked 2048-d)
  * footage_dir : where this domain's source videos live under footage_data/

Select with the INGEST_PROFILE env var (default "india") or run_ingest --profile.
On this branch INDIA is the default — we work on and demo the real Surat (SUR01) footage.
CityFlow is retained as the dormant REGRESSION BENCH: the only footage with ground truth
(the S01 IDF1 / re-ID rank-1/5 numbers), run it explicitly with --profile cityflow
(evaluate_reid.py still defaults to cityflow so scored eval is unchanged). Nothing CityFlow
loads or runs unless it is explicitly selected.

Both profiles must emit the same locked dims, so the DB / matcher / search / UI never
fork — only detect + embed_reid select a knob here.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # apps/ingest/pipeline/profiles.py -> repo root

# constants.py (Phase-0 locked dims + the CityFlow detector defaults) is the parent dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import (  # noqa: E402
    YOLO_CLASSES as _CF_YOLO_CLASSES,
    YOLO_CONF as _CF_YOLO_CONF,
    YOLO_IMGSZ as _CF_YOLO_IMGSZ,
    YOLO_MODEL as _CF_YOLO_MODEL,
)


@dataclass(frozen=True)
class PersonDetector:
    """A second, COCO-trained detector run alongside a vehicle-only primary to add
    'person' entities (the india UVH-26 model has no person class). Its boxes are
    tracked in a separate id namespace and can override person-confusable vehicle
    boxes (a pedestrian mislabelled as a two-wheeler) at the merge step."""
    weights: str                                 # stock COCO YOLO weights (has a 'person' class)
    class_id: int                                # COCO person = 0
    imgsz: int
    conf: float
    suppress_vehicle_classes: tuple[int, ...]    # vehicle class ids a person box may override (2-wheeler/bicycle)


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
    person_detector: PersonDetector | None = None  # optional 2nd detector for 'person' entities


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

# India: UVH-26 / VehicleNet-Y26x 14-class taxonomy (all 'vehicle' entities). Ids below are
# VERIFIED against the checkpoint's model.names (best.pt). Swap reid_onnx once an India-tuned
# encoder exists (still 2048-d).
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
    # Needs India footage under footage_data/india/ and (eventually) an India-tuned reid_onnx;
    # the detector below is the official IISc model, wired and verified to load.
    "india": Profile(
        name="india",
        # Official IISc AIM UVH-26-MV YOLOv11-X (arXiv 2511.02563, Apache-2.0 repo / AGPL-3.0
        # underlying YOLO). Fine-tuned on the UVH-26 Bengaluru-CCTV dataset, 14 Indian classes.
        # Absolute path — ingest runs from apps/ingest but weights live at repo-root models/.
        # (A third-party YOLO26x fine-tune, Perception365/VehicleNet-Y26x, is also in models/;
        # we use the official one for provenance/citability.)
        yolo_model=str(_REPO_ROOT / "models" / "iisc-aim" / "UVH-26" / "weights"
                       / "YOLOv11-X" / "UVH-26-MV-YOLOv11-X.pt"),
        yolo_imgsz=640,                             # trained at 640; bump if small-object recall needs it
        yolo_conf=_CF_YOLO_CONF,
        yolo_classes=None,                          # keep all 14 India classes
        tracker="botsort.yaml",
        class_map=_INDIA_CLASS_MAP,
        default_subtype="other",
        reid_onnx="veri_reid.onnx",                 # until an India-tuned encoder is trained
        footage_dir="india",
        # UVH-26 is vehicle-only; pair it with a CrowdHuman-trained person detector for people.
        # Model chosen by A/B on SUR01 (Jul 2026): CrowdHuman YOLOv8n (yakhyo/yolov8-crowdhuman,
        # native ultralytics, class {0:'person'}) BEAT stock YOLO11-X-COCO *and* a proprietary
        # YOLOv8-X person model on Surat dense-crowd footage — training-domain match (packed,
        # occluded crowds) dominates model size. imgsz 1280 (native-res recovers small/distant
        # peds; far-crowd misses are a resolution wall only tiling/SAHI fixes), conf 0.25 (distant
        # dets are low-conf, tracker prunes flicker). Suppress two-wheeler(7)/bicycle(11) boxes a
        # same-sized person box explains (pedestrian FPs; real riders kept, bike box >1.9x rider).
        person_detector=PersonDetector(
            weights=str(_REPO_ROOT / "models" / "crowdhuman" / "yolov8n_crowdhuman.pt"),
            class_id=0,
            imgsz=1280,
            conf=0.25,
            suppress_vehicle_classes=(7, 11),
        ),
    ),
}


_ACTIVE: Profile | None = None


def active() -> Profile:
    """The selected profile (INGEST_PROFILE env, default 'india'). Resolved once."""
    global _ACTIVE
    if _ACTIVE is None:
        name = os.environ.get("INGEST_PROFILE", "india")
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
