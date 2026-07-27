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

# Forensic exports are kept outside ingest output: ingest artifacts are evidence inputs,
# while packages are derived case outputs. The private Ed25519 key is generated lazily,
# mode 0600, and must be replaced with managed key material in a real deployment.
FORENSIC_EXPORT_ROOT = Path(os.environ.get(
    "FORENSIC_EXPORT_ROOT", REPO_ROOT / "apps" / "backend_py" / "forensic_exports"))
FORENSIC_KEY_DIR = Path(os.environ.get(
    "FORENSIC_KEY_DIR", REPO_ROOT / "apps" / "backend_py" / ".forensic_keys"))
FORENSIC_CLIP_PADDING_S = float(os.environ.get("FORENSIC_CLIP_PADDING_S", "2.0"))

# Model inventory recorded in every export. Scene-specific detector selection mirrors the
# ingest profiles; environment overrides support a deployment with different checkpoints.
SIGLIP_REVISION = os.environ.get("SIGLIP_REVISION", "main (unpinned prototype)")
INDIA_VEHICLE_DETECTOR = os.environ.get(
    "INDIA_VEHICLE_DETECTOR", "IISc UVH-26 YOLOv11-X")
INDIA_PERSON_DETECTOR = os.environ.get(
    "INDIA_PERSON_DETECTOR", "CrowdHuman YOLOv8n")
CITYFLOW_DETECTOR = os.environ.get("CITYFLOW_DETECTOR", "Ultralytics YOLO11m COCO")
TRACKER_MODEL = os.environ.get("TRACKER_MODEL", "BoT-SORT")


def forensic_model_inventory(scene: str) -> dict:
    detectors = (
        {"vehicle": INDIA_VEHICLE_DETECTOR, "person": INDIA_PERSON_DETECTOR}
        if scene.upper().startswith("SUR") or "surat" in scene.lower()
        else {"vehicle_and_person": CITYFLOW_DETECTOR}
    )
    return {
        "semantic_encoder": {
            "id": SIGLIP_MODEL,
            "revision": SIGLIP_REVISION,
            "embedding_dimension": SEMANTIC_DIM,
        },
        "detectors": detectors,
        "tracker": TRACKER_MODEL,
        "attribute_vlm": {"enabled": False, "role": "not used for retrieval"},
    }

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

# Wall-clock rendering for forensic reports. Stored tracklet timestamps are scene-clock
# seconds whose time-of-day is the real camera clock, but whose date component comes from
# the ingest constant SCENE_BASE_WALL (2024-01-01) — a placeholder, not footage metadata.
# The report therefore prints a configured recording date and says so explicitly rather
# than presenting the ingest placeholder as a recorded fact.
RECORDING_TIMEZONE = os.environ.get("RECORDING_TIMEZONE", "IST (UTC+05:30)")
# ISO date (YYYY-MM-DD) per scene: "SUR01=2025-11-14,S01=2024-01-01". Scenes absent here
# fall back to the ingest base date, which keeps existing exports reproducible.
_RECORDING_DATES = dict(
    part.split("=", 1)
    for part in os.environ.get("RECORDING_DATES", "").split(",")
    if "=" in part
)
INGEST_BASE_DATE = os.environ.get("INGEST_BASE_DATE", "2024-01-01")


def recording_date(scene: str) -> tuple[str, bool]:
    """(ISO date for `scene`, whether it was explicitly configured)."""
    configured = _RECORDING_DATES.get(scene)
    return (configured, True) if configured else (INGEST_BASE_DATE, False)

# SigLIP text encoder device: CPU is plenty for one short query per search.
DEVICE = os.environ.get("SEARCH_DEVICE", "cpu")

# HNSW recall knob. pgvector's hnsw.ef_search defaults to 40 candidates — fine at a
# few thousand rows, but at SUR01's ~25k tracklets it drops true neighbors (manifests
# as "the obviously-matching clip is missing"). We raise it per query in engine._connect().
# 200 is a safe recall/latency trade at this scale (still single-digit ms); set 0 to disable.
HNSW_EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", "200"))

# Multi-view retrieval. ANN finds a broad crop pool; exact cosine scoring and
# per-tracklet aggregation happen in Python so the final ordering is deterministic.
CROP_CANDIDATES_PER_PROMPT = int(os.environ.get("CROP_CANDIDATES_PER_PROMPT", "750"))
CROP_AGGREGATION = os.environ.get("CROP_AGGREGATION", "top2_mean")
COMPOSITION_MODE = os.environ.get("COMPOSITION_MODE", "weighted")
EXACT_CROP_SCAN = os.environ.get("EXACT_CROP_SCAN", "0").lower() in ("1", "true", "yes")

# Conservative presentation-only fragment grouping. Simultaneous objects are never
# merged; nearby non-overlapping fragments need very high semantic similarity.
DEDUP_MAX_GAP_S = float(os.environ.get("DEDUP_MAX_GAP_S", "2.0"))
DEDUP_MIN_SIM = float(os.environ.get("DEDUP_MIN_SIM", "0.94"))
