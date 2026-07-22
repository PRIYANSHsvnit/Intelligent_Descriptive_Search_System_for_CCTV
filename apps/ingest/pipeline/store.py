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
    "plate_text", "plate_conf", "plate_raw",
]
_UPDATE = ", ".join(f"{c}=EXCLUDED.{c}" for c in _COLS if c != "tracklet_id")
_SQL = (
    f"INSERT INTO tracklets ({', '.join(_COLS)}) "
    f"VALUES ({', '.join(['%s'] * len(_COLS))}) "
    f"ON CONFLICT (tracklet_id) DO UPDATE SET {_UPDATE}"
)
_CROP_SQL = """
    INSERT INTO tracklet_crops
      (tracklet_id, crop_index, frame_no, crop_ref, quality, semantic_vector)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (tracklet_id, crop_index) DO UPDATE SET
      frame_no=EXCLUDED.frame_no,
      crop_ref=EXCLUDED.crop_ref,
      quality=EXCLUDED.quality,
      semantic_vector=EXCLUDED.semantic_vector
"""


def _load_vecs(out, name, n, dim):
    p = out / "vec" / f"{name}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.shape != (n, dim):
        raise ValueError(f"{p} has shape {arr.shape}, expected {(n, dim)}")
    return arr


def _load_crop_vecs(out, tracklets):
    """Load crop-aligned SigLIP vectors; legacy cameras without the artifact return None."""
    p = out / "vec" / "semantic_crops.npz"
    if not p.exists():
        return None
    with np.load(p) as data:
        vectors = data["vectors"].astype(np.float32)
        tracklet_indices = data["tracklet_indices"].astype(np.int64)
        crop_indices = data["crop_indices"].astype(np.int64)
    if vectors.ndim != 2 or vectors.shape[1] != 1152:
        raise ValueError(f"{p} vectors have shape {vectors.shape}, expected (*, 1152)")
    if not (len(vectors) == len(tracklet_indices) == len(crop_indices)):
        raise ValueError(f"{p} arrays are not aligned")

    rows = []
    for vec, ti, ci in zip(vectors, tracklet_indices, crop_indices):
        if ti < 0 or ti >= len(tracklets):
            raise ValueError(f"{p} has invalid tracklet index {ti}")
        t = tracklets[int(ti)]
        refs = t.get("crop_refs") or []
        if ci < 0 or ci >= len(refs):
            raise ValueError(f"{p} has invalid crop index {ci} for {t['tracklet_id']}")
        meta = t.get("crop_meta") or []
        cm = meta[int(ci)] if ci < len(meta) else {}
        rows.append((
            t["tracklet_id"], int(ci), cm.get("frame_no"), refs[int(ci)],
            cm.get("quality"), vec,
        ))
    return rows


def run(scene: str, cam: str, min_detections: int = 2) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    semantic = _load_vecs(out, "semantic", n, 1152)
    reid_color = _load_vecs(out, "reid_color", n, 56)
    reid_app = _load_vecs(out, "reid_appearance", n, 2048)
    crop_rows = _load_crop_vecs(out, tracklets)

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
            t.get("plate_text"), t.get("plate_conf"), t.get("plate_raw"),
        ))

    eligible_ids = {r[0] for r in rows}
    if crop_rows is not None:
        crop_rows = [r for r in crop_rows if r[0] in eligible_ids]

    with connect() as conn, conn.cursor() as cur:
        cur.executemany(_SQL, rows)
        if crop_rows is not None:
            affected = sorted(eligible_ids)
            if affected:
                cur.execute("DELETE FROM tracklet_crops WHERE tracklet_id = ANY(%s)", (affected,))
            if crop_rows:
                cur.executemany(_CROP_SQL, crop_rows)
        conn.commit()

    return {"cam": cam, "inserted": len(rows),
            "crop_vectors": len(crop_rows) if crop_rows is not None else 0,
            "dropped_lt_min": dropped, "video_ref": video_ref}
