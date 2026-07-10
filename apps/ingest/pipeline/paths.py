"""Filesystem layout, camera discovery, and the global scene clock.

Output layout (per camera), all keys stored in the DB are RELATIVE to OUTPUT_ROOT:
    output/<scene>/<cam>/
        crops/t<tid>_<k>.jpg      K best thumbnails, k=0 is best
        detections.npy            (M,8) float32: frame,tid,x1,y1,x2,y2,conf,cls  (for the evaluator)
        tracklets.json            per-tracklet metadata (grows as passes run)
        vec/semantic.npy          (N,1152) aligned to tracklets.json order
        vec/reid_color.npy        (N,56)
        vec/reid_appearance.npy   (N,2048)
        media/<cam>.mp4           web-playable transcode (video_ref)
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

# apps/ingest/
INGEST_ROOT = Path(__file__).resolve().parent.parent
# repo root
REPO_ROOT = INGEST_ROOT.parent.parent
FOOTAGE_ROOT = REPO_ROOT / "footage_data" / "train"
CAM_TIMESTAMP_DIR = REPO_ROOT / "footage_data" / "cam_timestamp"
OUTPUT_ROOT = INGEST_ROOT / "output"

# Synthetic display clock only (CityFlow has no real wall time); ts_*_s is the real filter.
SCENE_BASE_WALL = _dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=_dt.timezone.utc)

DEFAULT_FPS = 10.0


def scene_dir(scene: str) -> Path:
    return FOOTAGE_ROOT / scene


def cam_video(scene: str, cam: str) -> Path:
    return FOOTAGE_ROOT / scene / cam / "vdo.avi"


def cam_gt(scene: str, cam: str) -> Path:
    return FOOTAGE_ROOT / scene / cam / "gt" / "gt.txt"


def cam_out(scene: str, cam: str) -> Path:
    return OUTPUT_ROOT / scene / cam


def list_cams(scene: str) -> list[str]:
    d = scene_dir(scene)
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "vdo.avi").exists())


def load_cam_offsets(scene: str) -> dict[str, float]:
    """cam_timestamp/<scene>.txt → {cam: offset_seconds}. Never default to 0 (S04 reaches 176s)."""
    f = CAM_TIMESTAMP_DIR / f"{scene}.txt"
    offsets: dict[str, float] = {}
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cam, off = line.split()
        offsets[cam] = float(off)
    return offsets


def wall_time(ts_s: float) -> _dt.datetime:
    return SCENE_BASE_WALL + _dt.timedelta(seconds=ts_s)


def rel_key(path: Path) -> str:
    """Path under OUTPUT_ROOT → relative key stored in the DB (never laptop-absolute)."""
    return str(path.relative_to(OUTPUT_ROOT))
