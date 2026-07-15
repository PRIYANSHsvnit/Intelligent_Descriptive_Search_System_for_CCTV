"""Stage 7: upsert per-camera tracklet rows into Postgres.

Reads tracklets.json + vec/*.npy for a camera and INSERTs one row per tracklet
(ON CONFLICT DO UPDATE). Drops tracks with num_detections < 2 (detector noise).
reid_appearance is optional — stored NULL until the re-ID pass backfills it.
"""

from __future__ import annotations

import json

import numpy as np
from psycopg.types.json import Jsonb

from . import paths
from .db import connect

_COLS = [
    "tracklet_id", "scene", "camera_id", "entity_type", "subtype", "color",
    "frame_start", "frame_end", "ts_start_s", "ts_end_s", "wall_start", "wall_end",
    "num_detections", "avg_conf", "crop_refs", "video_ref",
    "semantic_vector", "reid_appearance", "reid_color", "person_attrs",
]
_UPDATE = ", ".join(f"{c}=EXCLUDED.{c}" for c in _COLS if c != "tracklet_id")
_SQL = (
    f"INSERT INTO tracklets ({', '.join(_COLS)}) "
    f"VALUES ({', '.join(['%s'] * len(_COLS))}) "
    f"ON CONFLICT (tracklet_id) DO UPDATE SET {_UPDATE}"
)


def _load_vecs(out, name, n, dim):
    p = out / "vec" / f"{name}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.shape != (n, dim):
        raise ValueError(f"{p} has shape {arr.shape}, expected {(n, dim)}")
    return arr


def run(scene: str, cam: str, min_detections: int = 2) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    semantic = _load_vecs(out, "semantic", n, 1152)
    reid_color = _load_vecs(out, "reid_color", n, 56)
    reid_app = _load_vecs(out, "reid_appearance", n, 2048)

    video_path = out / "media" / f"{cam}.mp4"
    video_ref = paths.rel_key(video_path) if video_path.exists() else None

    rows = []
    dropped = 0
    for i, t in enumerate(tracklets):
        if t["num_detections"] < min_detections:
            dropped += 1
            continue
        rows.append((
            t["tracklet_id"], t["scene"], t["camera_id"], t["entity_type"], t["subtype"], t.get("color"),
            t["frame_start"], t["frame_end"], t["ts_start_s"], t["ts_end_s"], t["wall_start"], t["wall_end"],
            t["num_detections"], t["avg_conf"], t["crop_refs"], video_ref,
            semantic[i] if semantic is not None else None,
            reid_app[i] if reid_app is not None else None,
            reid_color[i] if reid_color is not None else None,
            Jsonb(t["person_attrs"]) if t.get("person_attrs") else None,
        ))

    with connect() as conn, conn.cursor() as cur:
        cur.executemany(_SQL, rows)
        conn.commit()

    return {"cam": cam, "inserted": len(rows), "dropped_lt_min": dropped, "video_ref": video_ref}
