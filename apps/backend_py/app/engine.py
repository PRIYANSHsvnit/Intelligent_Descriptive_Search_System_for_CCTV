"""Search engine: SigLIP text encoder (CPU) + pgvector query with hybrid filters,
dedup, and fail-open behavior. See plan.md Path B and the API contract."""

from __future__ import annotations

import re

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoProcessor

from . import config, query_rewrite

_processor = None
_model = None


def load_model() -> None:
    """Load the SigLIP text/image model once (text tower for q=, image tower for
    reference-image search — both encode into the same 1152-d space)."""
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


def encode_image(img) -> np.ndarray:
    """PIL image → L2-normalized 1152-d vector, same space as stored crop vectors.
    Expects a tight crop of the object (a full frame embeds 'street scene')."""
    inp = _processor(images=[img.convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        feats = _model.get_image_features(pixel_values=inp.pixel_values.to(config.DEVICE))
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
           ts_start_s, ts_end_s, crop_refs, video_ref, global_id, person_attrs,
           plate_text, plate_conf,
           1 - (semantic_vector <=> %(q)s) AS score
    FROM tracklets
    WHERE {where}
    ORDER BY semantic_vector <=> %(q)s
    LIMIT %(lim)s
"""


def _run_query(qvec, entity_type, scene, t0, t1, limit, camera_id=None, color=None) -> list[dict]:
    where = ["semantic_vector IS NOT NULL"]
    params = {"q": qvec, "lim": limit}
    if entity_type:
        where.append("entity_type = %(etype)s")
        params["etype"] = entity_type
    if scene:
        where.append("scene = %(scene)s")
        params["scene"] = scene
    if camera_id:
        where.append("camera_id = %(cam)s")
        params["cam"] = camera_id
    if color:
        where.append("color = %(color)s")
        params["color"] = color
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
    seen_gid: set[tuple[str, int]] = set()  # gids are scene-namespaced
    for r in rows:
        gid = r.get("global_id")
        if gid is not None:
            key = (r["scene"], gid)
            if key in seen_gid:
                continue
            seen_gid.add(key)
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


def _video_window(scene: str, cam: str, ts0: float, ts1: float) -> dict:
    """Stored timestamps are scene-clock; the per-camera video file starts at the
    camera's offset — subtract it so the player can seek."""
    off = config.camera_offset(scene, cam)
    return {
        "video_start_s": round(max(0.0, ts0 - off), 2),
        "video_end_s": round(max(0.0, ts1 - off), 2),
    }


def _attr_tags(pa: dict | None) -> list[str]:
    """Compact human-readable chips from the VLM/SigLIP person_attrs JSONB.
    Skips 'unknown'/'none'/'no' so only committed attributes show."""
    if not pa:
        return []

    def name(key: str) -> str | None:
        v = pa.get(key)
        n = v.get("name") if isinstance(v, dict) else None
        return n if n and n not in ("unknown", "none", "no") else None

    tags: list[str] = []
    for key in ("apparent_gender", "age"):
        if n := name(key):
            tags.append(n)
    if n := name("headwear"):
        tags.append(n)
    if name("backpack") == "yes":
        tags.append("backpack")
    if n := name("upper_color"):
        tags.append(f"{n} top")
    if n := name("lower_color"):
        tags.append(f"{n} bottom")
    return tags


_PLATE_SELECT = """
    SELECT tracklet_id, scene, camera_id, subtype, color, entity_type,
           ts_start_s, ts_end_s, crop_refs, video_ref, global_id, person_attrs,
           plate_text, plate_conf, plate_raw,
           COALESCE(plate_text = %(p)s, false) AS exact,
           COALESCE(plate_text ILIKE %(pat)s, false)
             OR COALESCE(plate_raw ILIKE %(pat)s, false) AS substr,
           GREATEST(COALESCE(similarity(plate_text, %(p)s), 0),
                    COALESCE(similarity(plate_raw, %(p)s), 0)) AS sim,
           COUNT(*) OVER (PARTITION BY scene, COALESCE(global_id::text, tracklet_id))
             AS sightings
    FROM tracklets
    WHERE (plate_text IS NOT NULL OR plate_raw IS NOT NULL) {extra}
      AND (plate_text = %(p)s
           OR plate_text ILIKE %(pat)s OR plate_raw ILIKE %(pat)s
           OR similarity(plate_text, %(p)s) > %(minsim)s
           OR similarity(plate_raw, %(p)s) > %(minsim)s)
    ORDER BY exact DESC, substr DESC, (plate_text IS NOT NULL) DESC, sim DESC,
             (ts_end_s - ts_start_s) DESC
    LIMIT %(lim)s
"""


def search_plate(plate_query, scene, limit, min_sim=0.30, camera_id=None):
    """Layered plate lookup, tiered so retrieval never over-claims:
    exact on the validated plate first, then substring on both tiers (partial
    queries like '1139'), then trigram similarity (catches raw reads such as
    'JO5AY1139' for the query 'GJ05AY1139'). Results carry matched_on so the UI
    can present fuzzy hits as candidates to verify, not identifications."""
    p = re.sub(r"[^A-Z0-9]", "", plate_query.upper())
    if not p:
        return {"results": [], "fell_back": False}
    params = {"p": p, "pat": f"%{p}%", "minsim": min_sim, "lim": limit}
    extra = ""
    if scene:
        extra += " AND scene = %(scene)s"
        params["scene"] = scene
    if camera_id:
        extra += " AND camera_id = %(cam)s"
        params["cam"] = camera_id
    with _connect() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(_PLATE_SELECT.format(extra=extra), params)
        rows = cur.fetchall()
    # plate-stitched tracklets share a global_id — one card per physical vehicle
    # (rows arrive best-tier-first, so the kept row is the strongest match)
    seen: set[tuple[str, int]] = set()
    deduped = []
    for r in rows:
        gid = r.get("global_id")
        if gid is not None:
            if (r["scene"], gid) in seen:
                continue
            seen.add((r["scene"], gid))
        deduped.append(r)
    rows = deduped
    results = [
        {
            "tracklet_id": r["tracklet_id"],
            "scene": r["scene"],
            "camera_id": r["camera_id"],
            "camera_label": config.camera_label(r["scene"], r["camera_id"]),
            "subtype": r["subtype"],
            "color": r["color"],
            "ts_start_s": round(r["ts_start_s"], 2),
            "ts_end_s": round(r["ts_end_s"], 2),
            **_video_window(r["scene"], r["camera_id"], r["ts_start_s"], r["ts_end_s"]),
            "score": round(float(r["sim"]), 4),
            "crop_url": _crop_url(r),
            "video_url": f"/media/{r['scene']}/{r['camera_id']}" if r["video_ref"] else None,
            "global_id": r["global_id"],
            "attrs": _attr_tags(r.get("person_attrs")),
            "plate": r["plate_text"],
            "plate_conf": r["plate_conf"],
            "sightings": r["sightings"],
            "matched_on": ("exact" if r["exact"] else
                           "partial" if r["substr"] else "fuzzy"),
        }
        for r in rows
    ]
    return {"results": results, "fell_back": False}


def _shape_result(r: dict) -> dict:
    return {
        "tracklet_id": r["tracklet_id"],
        "scene": r["scene"],
        "camera_id": r["camera_id"],
        "camera_label": config.camera_label(r["scene"], r["camera_id"]),
        "subtype": r["subtype"],
        "color": r["color"],
        "ts_start_s": round(r["ts_start_s"], 2),
        "ts_end_s": round(r["ts_end_s"], 2),
        **_video_window(r["scene"], r["camera_id"], r["ts_start_s"], r["ts_end_s"]),
        "score": round(float(r["score"]), 4),
        "crop_url": _crop_url(r),
        "video_url": f"/media/{r['scene']}/{r['camera_id']}" if r["video_ref"] else None,
        "global_id": r["global_id"],
        "attrs": _attr_tags(r.get("person_attrs")),
        "plate": r.get("plate_text"),
        "plate_conf": r.get("plate_conf"),
    }


def _search_with_vec(qvec, entity_type, scene, t0, t1, limit, camera_id=None, color=None):
    """Shared tail for text and image queries: pgvector search + fail-open + dedup."""
    fetch = max(limit * 4, 40)  # over-fetch so dedup still returns ~limit
    rows = _run_query(qvec, entity_type, scene, t0, t1, fetch, camera_id, color)

    fell_back = False
    if not rows and (entity_type or color or (t0 is not None)):
        # fail open: a starving soft filter (type/colour/time) — drop those, but keep the
        # intentional scene + camera location the user picked
        rows = _run_query(qvec, None, scene, None, None, fetch, camera_id, None)
        fell_back = True

    rows = _dedup(rows)[:limit]
    return {"results": [_shape_result(r) for r in rows], "fell_back": fell_back}


def search(query, entity_type, scene, t0, t1, limit, camera_id=None, color=None):
    # translate (Gujarati/Hindi/…→English) + caption-template the raw query so it
    # lands in SigLIP's caption-trained space; fail-open to a static wrapper.
    rewritten = query_rewrite.rewrite(query)
    out = _search_with_vec(encode_text(rewritten), entity_type, scene, t0, t1, limit,
                           camera_id, color)
    # debug/demo transparency — drop these two fields before shipping.
    out["original_query"] = query
    out["rewritten_query"] = rewritten
    return out


def search_image(img, entity_type, scene, t0, t1, limit, camera_id=None, color=None):
    """Reference-image search: SigLIP image tower → same vector space as text.
    NOTE image↔image scores run much higher than text↔image (modality gap) —
    ranking is comparable, absolute scores are not."""
    return _search_with_vec(encode_image(img), entity_type, scene, t0, t1, limit,
                            camera_id, color)


# "more like this": the stored vectors ARE valid query vectors, so similarity search
# by an existing tracklet needs no encoder at all. semantic = looks-alike (SigLIP);
# reid = same-instance appearance (2048-d re-ID embedding, what it's trained for).
_SIMILAR_COLS = {"semantic": "semantic_vector", "reid": "reid_appearance"}


def search_similar(tracklet_id: str, vec: str, limit: int):
    src_sql = """
        SELECT semantic_vector AS sem, reid_appearance AS reid,
               scene, global_id, entity_type
        FROM tracklets WHERE tracklet_id = %s
    """
    with _connect() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(src_sql, (tracklet_id,))
        src = cur.fetchone()
        if src is None:
            return None
        col = _SIMILAR_COLS[vec]
        qv = src["reid"] if vec == "reid" else src["sem"]
        fell_back = False
        if qv is None and vec == "reid":
            # fail open: no re-ID vector stored (e.g. SUR01) — use visual similarity
            qv, col, fell_back = src["sem"], "semantic_vector", True
        if qv is None:
            return {"results": [], "fell_back": fell_back}

        # same entity_type only: person and vehicle re-ID vectors come from different
        # encoders (and cross-type "similar" is meaningless for semantic too). Exclude
        # the source object itself — its own sightings are /trace's job, not search's.
        where = [f"{col} IS NOT NULL", "entity_type = %(etype)s",
                 "tracklet_id <> %(tid)s"]
        params = {
            "q": qv,  # pgvector Vector round-trips as-is
            "etype": src["entity_type"],
            "tid": tracklet_id,
            "lim": max(limit * 4, 40),
        }
        if src["global_id"] is not None:
            where.append("NOT (scene = %(sscene)s AND global_id = %(sgid)s)")
            params["sscene"], params["sgid"] = src["scene"], src["global_id"]
        sql = f"""
            SELECT tracklet_id, scene, camera_id, subtype, color, entity_type,
                   ts_start_s, ts_end_s, crop_refs, video_ref, global_id, person_attrs,
                   plate_text, plate_conf,
                   1 - ({col} <=> %(q)s) AS score
            FROM tracklets
            WHERE {' AND '.join(where)}
            ORDER BY {col} <=> %(q)s
            LIMIT %(lim)s
        """
        cur.execute(sql, params)
        rows = cur.fetchall()

    rows = _dedup(rows)[:limit]
    return {"results": [_shape_result(r) for r in rows], "fell_back": fell_back}


def scene_cameras(scene: str) -> dict:
    """Cameras in a scene with their label, indexed count, and time-of-day coverage —
    feeds the frontend location picker (each SUR01 camera is its own location today)."""
    sql = """
        SELECT camera_id, COUNT(*) AS n,
               MIN(ts_start_s) AS t0, MAX(ts_end_s) AS t1
        FROM tracklets WHERE scene = %s AND semantic_vector IS NOT NULL
        GROUP BY camera_id ORDER BY camera_id
    """
    with _connect() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, (scene,))
        rows = cur.fetchall()
    cams = [
        {
            "camera_id": r["camera_id"],
            "camera_label": config.camera_label(scene, r["camera_id"]),
            "count": r["n"],
            "ts_start_s": round(r["t0"], 2),
            "ts_end_s": round(r["t1"], 2),
        }
        for r in rows
    ]
    return {"scene": scene, "cameras": cams, "total": sum(c["count"] for c in cams)}


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
            "camera_label": config.camera_label(scene, r["camera_id"]),
            "ts_start_s": round(r["ts_start_s"], 2),
            "ts_end_s": round(r["ts_end_s"], 2),
            **_video_window(scene, r["camera_id"], r["ts_start_s"], r["ts_end_s"]),
            "ground_x": r["ground_x"],
            "ground_y": r["ground_y"],
            "crop_url": f"/files/{r['crop_refs'][0]}" if r["crop_refs"] else None,
            "video_url": f"/media/{scene}/{r['camera_id']}" if r["video_ref"] else None,
        }
        for r in rows
    ]
    return {"scene": scene, "global_id": global_id, "hops": hops}
