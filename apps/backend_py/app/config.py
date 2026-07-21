"""Backend configuration. Model id MUST match the ingest lock (apps/ingest/constants.py)
so query text vectors live in the same space as the stored image vectors."""

import functools
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load apps/backend_py/.env (GROQ_API_KEY etc.) before anything reads os.environ.
# config.py is imported first by every module, so this runs early. does not override
# vars already set in the shell.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# LOCKED — identical to apps/ingest/constants.py SIGLIP_MODEL. Do not diverge.
SIGLIP_MODEL = "google/siglip2-so400m-patch14-224"
SEMANTIC_DIM = 1152

# repo root: apps/backend_py/app/config.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
# where the ingest pipeline wrote crops + per-camera MP4s (relative keys resolve here)
OUTPUT_ROOT = Path(os.environ.get("INGEST_OUTPUT_ROOT", REPO_ROOT / "apps" / "ingest" / "output"))

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cctv:cctv@localhost:5432/cctv")

# Scene location maps (cam_loc/*.png). S03/S04/S05 share one map (S0345.png).
CAM_LOC_DIR = REPO_ROOT / "footage_data" / "cam_loc"


def scene_map_file(scene: str) -> Path:
    name = "S0345" if scene in ("S03", "S04", "S05") else scene
    return CAM_LOC_DIR / f"{name}.png"

# Human-readable camera names for the UI (cam_labels/<scene>.json: camera_id -> label).
# camera_id stays the functional key (media URLs, map positions); this is display-only,
# so a missing file/entry harmlessly falls back to the raw id.
CAM_LABEL_DIR = REPO_ROOT / "footage_data" / "cam_labels"


@functools.lru_cache(maxsize=None)
def _scene_labels(scene: str) -> dict[str, str]:
    f = CAM_LABEL_DIR / f"{scene}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())


def camera_label(scene: str, camera_id: str) -> str:
    return _scene_labels(scene).get(camera_id, camera_id)

# Per-camera clock offsets (cam_timestamp/<scene>.txt: "<cam> <seconds>"). Stored
# tracklet timestamps are scene-clock (offset included); the video file starts at the
# offset, so player seeks need ts - offset. Missing file -> 0 (ts already video-local).
CAM_TS_DIR = REPO_ROOT / "footage_data" / "cam_timestamp"


@functools.lru_cache(maxsize=None)
def _scene_offsets(scene: str) -> dict[str, float]:
    f = CAM_TS_DIR / f"{scene}.txt"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = float(parts[1])
    return out


def camera_offset(scene: str, camera_id: str) -> float:
    return _scene_offsets(scene).get(camera_id, 0.0)

# SigLIP text encoder device: CPU is plenty for one short query per search.
DEVICE = os.environ.get("SEARCH_DEVICE", "cpu")
