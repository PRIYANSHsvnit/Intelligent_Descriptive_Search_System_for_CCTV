"""Backend configuration. Model id MUST match the ingest lock (apps/ingest/constants.py)
so query text vectors live in the same space as the stored image vectors."""

import os
from pathlib import Path

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

# SigLIP text encoder device: CPU is plenty for one short query per search.
DEVICE = os.environ.get("SEARCH_DEVICE", "cpu")
