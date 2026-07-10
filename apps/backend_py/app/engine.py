"""Search engine: SigLIP text encoder (CPU) + pgvector query with hybrid filters,
dedup, and fail-open behavior. See plan.md Path B and the API contract."""

from __future__ import annotations

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoProcessor

from . import config

_processor = None
_model = None


def load_model() -> None:
    """Load the SigLIP text/image model once (we only call the text tower)."""
    global _processor, _model
    if _model is not None:
        return
    _processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)
    _model = AutoModel.from_pretrained(config.SIGLIP_MODEL).to(config.DEVICE).eval()


def encode_text(query: str) -> np.ndarray:
    """Text → L2-normalized 1152-d vector in the same space as stored image vectors."""
    inp = _processor(
        text=[query], padding="max_length", max_length=64, return_tensors="pt"
    ).to(config.DEVICE)
    with torch.no_grad():
        feats = _model.get_text_features(**inp)
        if not isinstance(feats, torch.Tensor):  # transformers 5.x output object
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats[0].float().cpu().numpy()


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn


_SELECT = """
    SELECT tracklet_id, scene, camera_id, subtype, color, entity_type,
           ts_start_s, ts_end_s, crop_refs, video_ref, global_id,
           1 - (semantic_vector <=> %(q)s) AS score
    FROM tracklets
    WHERE {where}
    ORDER BY semantic_vector <=> %(q)s
    LIMIT %(lim)s
"""


def _run_query(qvec, entity_type, scene, t0, t1, limit) -> list[dict]:
    where = ["semantic_vector IS NOT NULL"]
    params = {"q": qvec, "lim": limit}
    if entity_type:
        where.append("entity_type = %(etype)s")
        params["etype"] = entity_type
    if scene:
        where.append("scene = %(scene)s")
        params["scene"] = scene
    if t0 is not None and t1 is not None:  # tracklet overlaps [t0, t1]
        where.append("ts_start_s <= %(t1)s AND ts_end_s >= %(t0)s")
        params["t0"], params["t1"] = t0, t1
    sql = _SELECT.format(where=" AND ".join(where))
    with _connect() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _overlaps(a: dict, b: dict) -> bool:
    return a["ts_start_s"] <= b["ts_end_s"] and b["ts_start_s"] <= a["ts_end_s"]


def _dedup(rows: list[dict]) -> list[dict]:
    """Collapse near-duplicates so one physical object appears once: same global_id,
    or same-camera same-subtype fragments overlapping in time. Keeps best score
    (rows arrive score-desc)."""
    kept: list[dict] = []
    seen_gid: set[int] = set()
    for r in rows:
        gid = r.get("global_id")
        if gid is not None:
            if gid in seen_gid:
                continue
            seen_gid.add(gid)
            kept.append(r)
            continue
        dup = any(
            k.get("global_id") is None
            and k["scene"] == r["scene"]
            and k["camera_id"] == r["camera_id"]
            and k["subtype"] == r["subtype"]
            and _overlaps(k, r)
            for k in kept
        )
        if not dup:
            kept.append(r)
    return kept


def _crop_url(row: dict) -> str | None:
    refs = row.get("crop_refs")
    return f"/files/{refs[0]}" if refs else None


def search(query, entity_type, scene, t0, t1, limit):
    qvec = encode_text(query)
    fetch = max(limit * 4, 40)  # over-fetch so dedup still returns ~limit
    rows = _run_query(qvec, entity_type, scene, t0, t1, fetch)

    fell_back = False
    if not rows and (entity_type or (t0 is not None)):
        # fail open: a starving filter (type/time) — drop it, keep the reliable scene
        rows = _run_query(qvec, None, scene, None, None, fetch)
        fell_back = True

    rows = _dedup(rows)[:limit]
    results = [
        {
            "tracklet_id": r["tracklet_id"],
            "scene": r["scene"],
            "camera_id": r["camera_id"],
            "subtype": r["subtype"],
            "color": r["color"],
            "ts_start_s": round(r["ts_start_s"], 2),
            "ts_end_s": round(r["ts_end_s"], 2),
            "score": round(float(r["score"]), 4),
            "crop_url": _crop_url(r),
            "video_url": f"/media/{r['scene']}/{r['camera_id']}" if r["video_ref"] else None,
            "global_id": r["global_id"],
        }
        for r in rows
    ]
    return {"results": results, "fell_back": fell_back}


def trace(scene: str, global_id: int):
    sql = """
        SELECT tracklet_id, camera_id, ts_start_s, ts_end_s, ground_x, ground_y,
               crop_refs, video_ref
        FROM tracklets WHERE scene = %s AND global_id = %s
        ORDER BY ts_start_s
    """
    with _connect() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, (scene, global_id))
        rows = cur.fetchall()
    hops = [
        {
            "tracklet_id": r["tracklet_id"],
            "camera_id": r["camera_id"],
            "ts_start_s": round(r["ts_start_s"], 2),
            "ts_end_s": round(r["ts_end_s"], 2),
            "ground_x": r["ground_x"],
            "ground_y": r["ground_y"],
            "crop_url": f"/files/{r['crop_refs'][0]}" if r["crop_refs"] else None,
            "video_url": f"/media/{scene}/{r['camera_id']}" if r["video_ref"] else None,
        }
        for r in rows
    ]
    return {"scene": scene, "global_id": global_id, "hops": hops}
