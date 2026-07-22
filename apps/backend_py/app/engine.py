"""Multi-view SigLIP retrieval with exact reranking and authoritative filters."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoProcessor

from . import config, query_components, query_rewrite

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


def encode_texts(queries: list[str]) -> np.ndarray:
    """Batch text captions into normalized vectors; composition costs one forward pass."""
    inp = _processor(
        text=queries, padding="max_length", max_length=64, return_tensors="pt"
    ).to(config.DEVICE)
    with torch.no_grad():
        feats = _model.get_text_features(**inp)
        if not isinstance(feats, torch.Tensor):  # transformers 5.x output object
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.float().cpu().numpy()


def encode_text(query: str) -> np.ndarray:
    """Compatibility wrapper for one caption."""
    return encode_texts([query])[0]


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
    # lift HNSW recall above pgvector's default ef_search=40, which drops true neighbors
    # at 25k+ rows. Session-scoped SET on this fresh connection; no-op for non-vector queries.
    if config.HNSW_EF_SEARCH:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {int(config.HNSW_EF_SEARCH)}")
    return conn


_RESULT_COLS = """
    t.tracklet_id, t.scene, t.camera_id, t.subtype, t.color, t.entity_type,
    t.ts_start_s, t.ts_end_s, t.crop_refs, t.video_ref, t.global_id,
    t.person_attrs, t.plate_text, t.plate_conf, t.semantic_vector
"""


def _filters(entity_type, scene, t0, t1, camera_id=None, color=None, alias="t"):
    """Build authoritative SQL filters. None are silently removed later."""
    where: list[str] = []
    params: dict = {}
    p = f"{alias}." if alias else ""
    if entity_type:
        where.append(f"{p}entity_type = %(etype)s")
        params["etype"] = entity_type
    if scene:
        where.append(f"{p}scene = %(scene)s")
        params["scene"] = scene
    if camera_id:
        where.append(f"{p}camera_id = %(cam)s")
        params["cam"] = camera_id
    if color:
        where.append(f"{p}color = %(color)s")
        params["color"] = color
    if t0 is not None:
        where.append(f"{p}ts_end_s >= %(t0)s")
        params["t0"] = t0
    if t1 is not None:
        where.append(f"{p}ts_start_s <= %(t1)s")
        params["t1"] = t1
    return where, params


def _to_np(value) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return value.to_numpy().astype(np.float32)
    return np.asarray(list(value), dtype=np.float32)


def _aggregate(values: np.ndarray, mode: str) -> float:
    if not len(values):
        return float("-inf")
    ordered = np.sort(values)[::-1]
    if mode == "max":
        return float(ordered[0])
    if mode == "best_single":
        return float(values[0])
    if mode == "mean":
        return float(values.mean())
    if mode == "top2_mean":
        return float(ordered[:2].mean())
    raise ValueError(f"unsupported crop aggregation: {mode}")


def _percentiles(values: np.ndarray) -> np.ndarray:
    """Stable [0,1] rank normalization; prompt modalities need not share a scale."""
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / float(len(values) - 1)


def _mean_query(qvec, entity_type, scene, t0, t1, limit, camera_id=None, color=None):
    where, params = _filters(entity_type, scene, t0, t1, camera_id, color)
    where.append("t.semantic_vector IS NOT NULL")
    params.update({"q": qvec, "lim": limit})
    sql = f"""
        SELECT {_RESULT_COLS}, 1 - (t.semantic_vector <=> %(q)s) AS score
        FROM tracklets t
        WHERE {' AND '.join(where)}
        ORDER BY t.semantic_vector <=> %(q)s
        LIMIT %(lim)s
    """
    with _connect() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    for row in rows:
        row["dedup_vector"] = _to_np(row["semantic_vector"])
    return rows


def _multi_crop_query(qvecs, entity_type, scene, t0, t1, limit, camera_id=None,
                      color=None, aggregation=None, composition=None):
    """ANN candidate union -> exact crop scoring -> prompt fusion."""
    aggregation = aggregation or config.CROP_AGGREGATION
    composition = composition or config.COMPOSITION_MODE
    if aggregation not in {"mean", "best_single", "max", "top2_mean"}:
        raise ValueError(f"unsupported crop aggregation: {aggregation}")
    if composition not in {"weighted", "soft_and"}:
        raise ValueError(f"unsupported composition mode: {composition}")

    where, base_params = _filters(entity_type, scene, t0, t1, camera_id, color)
    candidate_ids: set[str] = set()
    with _connect() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SELECT to_regclass('tracklet_crops') AS name")
        if cur.fetchone()["name"] is None:
            return None
        if config.EXACT_CROP_SCAN:
            sql = f"""
                SELECT DISTINCT c.tracklet_id
                FROM tracklet_crops c JOIN tracklets t ON t.tracklet_id=c.tracklet_id
                {('WHERE ' + ' AND '.join(where)) if where else ''}
            """
            cur.execute(sql, base_params)
            candidate_ids.update(r["tracklet_id"] for r in cur.fetchall())
        else:
            crop_where = ["c.semantic_vector IS NOT NULL", *where]
            sql = f"""
                SELECT c.tracklet_id
                FROM tracklet_crops c JOIN tracklets t ON t.tracklet_id=c.tracklet_id
                WHERE {' AND '.join(crop_where)}
                ORDER BY c.semantic_vector <=> %(q)s
                LIMIT %(candidate_limit)s
            """
            for qvec in qvecs:
                params = {**base_params, "q": qvec,
                          "candidate_limit": config.CROP_CANDIDATES_PER_PROMPT}
                cur.execute(sql, params)
                candidate_ids.update(r["tracklet_id"] for r in cur.fetchall())
        if not candidate_ids:
            return None

        cur.execute(
            f"""
                SELECT {_RESULT_COLS}, c.crop_index, c.crop_ref,
                       c.semantic_vector AS crop_vector
                FROM tracklets t JOIN tracklet_crops c ON c.tracklet_id=t.tracklet_id
                WHERE t.tracklet_id = ANY(%s)
                ORDER BY t.tracklet_id, c.crop_index
            """,
            (sorted(candidate_ids),),
        )
        crop_rows = cur.fetchall()

    grouped: dict[str, dict] = {}
    for row in crop_rows:
        tid = row["tracklet_id"]
        group = grouped.setdefault(tid, {"row": dict(row), "vectors": [], "refs": []})
        group["vectors"].append(_to_np(row["crop_vector"]))
        group["refs"].append(row["crop_ref"])
    if not grouped:
        return None

    ids = list(grouped)
    per_prompt = np.zeros((len(qvecs), len(ids)), dtype=np.float32)
    best_prompt_indices = np.zeros((len(qvecs), len(ids)), dtype=np.int32)
    for j, tid in enumerate(ids):
        vectors = np.stack(grouped[tid]["vectors"])
        mean = vectors.mean(axis=0)
        grouped[tid]["dedup_vector"] = mean / (np.linalg.norm(mean) + 1e-12)
        for pi, qvec in enumerate(qvecs):
            sims = vectors @ qvec
            per_prompt[pi, j] = _aggregate(sims, aggregation)
            best_prompt_indices[pi, j] = int(np.argmax(sims))

    if len(qvecs) == 1:
        final = per_prompt[0]
    else:
        normed = np.stack([_percentiles(scores) for scores in per_prompt])
        components = normed[1:]
        if composition == "soft_and":
            safe = np.clip(components, 1e-4, 1.0)
            harmonic = len(safe) / np.sum(1.0 / safe, axis=0)
            final = 0.5 * normed[0] + 0.5 * harmonic
        else:
            final = 0.5 * normed[0] + 0.5 * components.mean(axis=0)

    rows = []
    for j, tid in enumerate(ids):
        group = grouped[tid]
        row = group["row"]
        row["score"] = float(final[j])
        row["prompt_scores"] = [float(per_prompt[pi, j]) for pi in range(len(qvecs))]
        row["prompt_crop_refs"] = [
            group["refs"][best_prompt_indices[pi, j]] for pi in range(len(qvecs))
        ]
        row["matched_crop_ref"] = row["prompt_crop_refs"][0]
        row["dedup_vector"] = group["dedup_vector"]
        rows.append(row)
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:max(limit * 8, 80)]


def _overlaps(a: dict, b: dict) -> bool:
    return a["ts_start_s"] <= b["ts_end_s"] and b["ts_start_s"] <= a["ts_end_s"]


def _time_gap(a: dict, b: dict) -> float | None:
    if _overlaps(a, b):
        return None
    if a["ts_end_s"] < b["ts_start_s"]:
        return b["ts_start_s"] - a["ts_end_s"]
    return a["ts_start_s"] - b["ts_end_s"]


def _dedup(rows: list[dict]) -> list[dict]:
    """Conservatively group fragments for display without mutating evidence rows."""
    kept: list[dict] = []
    for r in rows:
        r["fragment_ids"] = [r["tracklet_id"]]
        r["sightings"] = 1
        gid = r.get("global_id")
        match = None
        for k in kept:
            if gid is not None and k.get("global_id") == gid and k["scene"] == r["scene"]:
                match = k
                break
            if (k["scene"] != r["scene"] or k["camera_id"] != r["camera_id"]
                    or k["entity_type"] != r["entity_type"] or k["subtype"] != r["subtype"]):
                continue
            gap = _time_gap(k, r)
            if gap is None:  # simultaneous look-alikes are distinct objects
                continue
            same_plate = (r.get("plate_text") and r.get("plate_text") == k.get("plate_text"))
            if same_plate and gap <= 30.0:
                match = k
                break
            if gap > config.DEDUP_MAX_GAP_S:
                continue
            av, bv = k.get("dedup_vector"), r.get("dedup_vector")
            if av is not None and bv is not None and float(np.dot(av, bv)) >= config.DEDUP_MIN_SIM:
                match = k
                break
        if match is None:
            kept.append(r)
        else:
            match["fragment_ids"].append(r["tracklet_id"])
            match["sightings"] += 1
    return kept


def _crop_url(row: dict) -> str | None:
    if row.get("matched_crop_ref"):
        return f"/files/{row['matched_crop_ref']}"
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
        "entity_type": r["entity_type"],
        "subtype": r["subtype"],
        "color": r["color"],
        "ts_start_s": round(r["ts_start_s"], 2),
        "ts_end_s": round(r["ts_end_s"], 2),
        **_video_window(r["scene"], r["camera_id"], r["ts_start_s"], r["ts_end_s"]),
        "score": round(float(r["score"]), 4),
        "crop_url": _crop_url(r),
        "matched_crop_ref": r.get("matched_crop_ref"),
        "video_url": f"/media/{r['scene']}/{r['camera_id']}" if r["video_ref"] else None,
        "global_id": r["global_id"],
        "attrs": _attr_tags(r.get("person_attrs")),
        "plate": r.get("plate_text"),
        "plate_conf": r.get("plate_conf"),
        "sightings": r.get("sightings", 1),
        "fragment_ids": r.get("fragment_ids", [r["tracklet_id"]]),
        "prompt_scores": r.get("prompt_scores"),
        "prompt_crop_refs": r.get("prompt_crop_refs"),
    }


def _filter_suggestions(entity_type, t0, t1, color):
    suggestions = []
    if t0 is not None or t1 is not None:
        suggestions.append({"action": "expand_time", "label": "Expand the time window"})
    if color:
        suggestions.append({"action": "remove_color", "label": "Remove the colour filter"})
    if entity_type:
        suggestions.append({"action": "all_entity_types", "label": "Search all entity types"})
    return suggestions


def _search_with_vectors(qvecs, entity_type, scene, t0, t1, limit, camera_id=None,
                         color=None, aggregation=None, composition=None):
    """Shared strict-filter tail for text/image queries with a mean-vector fallback."""
    fetch = max(limit * 8, 80)
    chosen_aggregation = aggregation or config.CROP_AGGREGATION
    if chosen_aggregation == "legacy_mean":
        rows = _mean_query(qvecs[0], entity_type, scene, t0, t1, fetch, camera_id, color)
        search_mode = "legacy_mean"
    else:
        rows = _multi_crop_query(
            qvecs, entity_type, scene, t0, t1, fetch, camera_id, color,
            aggregation=chosen_aggregation, composition=composition,
        )
        search_mode = "multi_crop"
    if rows is None:
        # Compatibility for live/legacy scenes not yet backfilled. This retains every
        # explicit filter; it is not the former unsafe filter-relaxation fallback.
        rows = _mean_query(qvecs[0], entity_type, scene, t0, t1, fetch, camera_id, color)
        search_mode = "mean_fallback"
    rows = _dedup(rows)[:limit]
    return {
        "results": [_shape_result(r) for r in rows],
        "fell_back": False,
        "search_mode": search_mode,
        "aggregation": chosen_aggregation,
        "composition": composition or config.COMPOSITION_MODE,
        "suggestions": _filter_suggestions(entity_type, t0, t1, color) if not rows else [],
        "applied_filters": {
            "type": entity_type, "scene": scene, "camera_id": camera_id,
            "color": color, "t0": t0, "t1": t1,
        },
    }


def search(query, entity_type, scene, t0, t1, limit, camera_id=None, color=None,
           aggregation=None, compositional=True, composition=None):
    # Translate first, then use a deterministic visible-vocabulary decomposition.
    rewritten = query_rewrite.rewrite(query)
    plan = query_components.build_query_plan(rewritten, entity_type)
    if aggregation == "legacy_mean" and not compositional:
        captions = [rewritten]
    else:
        captions = [plan.full_caption]
    if compositional and captions[0] == plan.full_caption:
        captions.extend(plan.components)
    qvecs = list(encode_texts(captions))
    out = _search_with_vectors(
        qvecs, entity_type, scene, t0, t1, limit, camera_id, color,
        aggregation=aggregation, composition=composition,
    )
    out["original_query"] = query
    out["rewritten_query"] = rewritten
    out["search_captions"] = captions
    out["entity_hint"] = plan.entity_hint
    out["searched_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for result in out["results"]:
        scores = result.pop("prompt_scores", None)
        crop_refs = result.pop("prompt_crop_refs", None) or []
        if scores:
            result["component_scores"] = [
                {
                    "caption": caption,
                    "score": round(float(score), 6),
                    "kind": "overall" if index == 0 else "component",
                    "supporting_crop_ref": crop_refs[index] if index < len(crop_refs) else None,
                    "supporting_crop_url": (
                        f"/files/{crop_refs[index]}" if index < len(crop_refs) else None
                    ),
                }
                for index, (caption, score) in enumerate(zip(captions, scores))
            ]
    # These labels are deliberately relative to this result set, not calibrated model
    # probabilities. They make the explanation readable without calling similarity a
    # confidence percentage.
    for component_index in range(len(captions)):
        available = [
            row["component_scores"][component_index]["score"]
            for row in out["results"]
            if len(row.get("component_scores", [])) > component_index
        ]
        if not available:
            continue
        low, high = np.percentile(available, [35, 70]) if len(available) > 2 else (
            min(available), max(available)
        )
        for row in out["results"]:
            if len(row.get("component_scores", [])) <= component_index:
                continue
            component = row["component_scores"][component_index]
            value = component["score"]
            component["match_strength"] = (
                "strong" if value >= high else "moderate" if value >= low else "weak"
            )
    return out


def search_image(img, entity_type, scene, t0, t1, limit, camera_id=None, color=None,
                 aggregation=None):
    """Reference-image search: SigLIP image tower → same vector space as text.
    NOTE image↔image scores run much higher than text↔image (modality gap) —
    ranking is comparable, absolute scores are not."""
    out = _search_with_vectors(
        [encode_image(img)], entity_type, scene, t0, t1, limit, camera_id, color,
        aggregation=aggregation,
    )
    out["searched_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for result in out["results"]:
        result.pop("prompt_scores", None)
    return out


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
