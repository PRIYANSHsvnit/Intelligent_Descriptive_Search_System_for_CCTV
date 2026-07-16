"""Per-tracklet bounding-box timeline for the player overlay.

Reads the ingest pipeline's per-camera detections.npy (rows of
frame,tid,x1,y1,x2,y2,conf,cls — see apps/ingest/pipeline/paths.py) and returns
one tracklet's boxes keyed by video-local seconds, so the UI can draw a live
box over the <video> while it plays. Coordinates are source-video pixels; the
served MP4 keeps source resolution (pipeline/media.py never rescales), so the
frontend maps them onto the element via videoWidth/videoHeight.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess

import numpy as np

from . import config

# person rows carry this sentinel class id (apps/ingest/pipeline/detect_track.py):
# vehicle and person trackers have separate id namespaces ('t' / 'p' prefixes),
# so a tid alone is ambiguous — the cls column disambiguates.
PERSON_CLS = 100

_ID_RE = re.compile(r"^(?P<scene>[^_]+)_(?P<cam>[^_]+)_(?P<prefix>[tp])(?P<tid>\d+)$")


@functools.lru_cache(maxsize=32)
def _video_fps(scene: str, cam: str) -> float:
    """r_frame_rate of the served MP4 — same clock the ingest used for ts_*
    (cv2 CAP_PROP_FPS of the source; media copy/remux/transcode keep it)."""
    path = config.OUTPUT_ROOT / scene / cam / "media" / f"{cam}.mp4"
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        num, den = json.loads(out)["streams"][0]["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        if fps > 0:
            return fps
    except Exception:
        pass
    return 10.0  # matches ingest's DEFAULT_FPS fallback


@functools.lru_cache(maxsize=8)
def _detections(scene: str, cam: str) -> np.ndarray | None:
    path = config.OUTPUT_ROOT / scene / cam / "detections.npy"
    if not path.exists():
        return None
    return np.load(str(path))


@functools.lru_cache(maxsize=512)
def tracklet_boxes(tracklet_id: str) -> dict | None:
    m = _ID_RE.match(tracklet_id)
    if not m:
        return None
    scene, cam = m["scene"], m["cam"]
    det = _detections(scene, cam)
    if det is None or det.size == 0:
        return None
    is_person = det[:, 7] == PERSON_CLS
    mask = (det[:, 1] == int(m["tid"])) & (is_person if m["prefix"] == "p" else ~is_person)
    rows = det[mask]
    if rows.size == 0:
        return None
    rows = rows[np.argsort(rows[:, 0])]
    fps = _video_fps(scene, cam)
    return {
        "tracklet_id": tracklet_id,
        "fps": fps,
        # [video-local seconds, x1, y1, x2, y2] in source pixels; frame_idx is
        # 1-based, so frame 1 sits at t=0
        "boxes": [
            [round(float((r[0] - 1) / fps), 3),
             round(float(r[2]), 1), round(float(r[3]), 1),
             round(float(r[4]), 1), round(float(r[5]), 1)]
            for r in rows
        ],
    }
