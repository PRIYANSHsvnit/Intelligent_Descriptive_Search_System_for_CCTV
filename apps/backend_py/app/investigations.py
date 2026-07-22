"""Case-board state and its append-only investigation audit ledger."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from . import config


class InvestigationError(ValueError):
    pass


def _clean(value: Any, field: str, maximum: int = 120) -> str:
    out = str(value or "").strip()
    if not out:
        raise InvestigationError(f"{field} is required")
    if len(out) > maximum:
        raise InvestigationError(f"{field} exceeds {maximum} characters")
    return out


def _case_id(value: Any) -> str:
    out = _clean(value, "case_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", out):
        raise InvestigationError("case_id may contain only letters, numbers, dot, underscore, and dash")
    return out


def ensure_case(case_id: str, officer: str, title: str | None = None) -> dict[str, Any]:
    case_id = _case_id(case_id)
    officer = _clean(officer, "officer")
    title = str(title or "").strip()[:240] or None
    created = False
    with psycopg.connect(config.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT * FROM cases WHERE case_id = %s", (case_id,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """INSERT INTO cases (case_id, title, officer)
                       VALUES (%s, %s, %s) RETURNING *""",
                    (case_id, title, officer),
                )
                row = cur.fetchone()
                created = True
            elif title and title != row["title"]:
                cur.execute(
                    """UPDATE cases SET title=%s, updated_at=NOW()
                       WHERE case_id=%s RETURNING *""",
                    (title, case_id),
                )
                row = cur.fetchone()
    if created:
        record_event(case_id, officer, "case_created", metadata={"title": title})
    return _json_row(row)


def record_event(
    case_id: str | None,
    officer: str,
    event_type: str,
    *,
    original_query: str | None = None,
    normalized_query: str | None = None,
    filters: dict[str, Any] | None = None,
    search_mode: str | None = None,
    returned_results: list[dict[str, Any]] | None = None,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    officer = _clean(officer, "officer")
    event_id = str(uuid.uuid4())
    model_versions = config.forensic_model_inventory(
        str((filters or {}).get("scene") or "unknown")
    )
    sql = """
        INSERT INTO search_events (
          event_id, case_id, officer, event_type, original_query, normalized_query,
          filters, search_mode, model_versions, returned_results, latency_ms, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with psycopg.connect(config.DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                event_id, case_id, officer, event_type, original_query, normalized_query,
                Jsonb(filters or {}), search_mode, Jsonb(model_versions),
                Jsonb(returned_results or []), latency_ms, Jsonb(metadata or {}),
            ))
    return event_id


def record_search(
    output: dict[str, Any], *, case_id: str, officer: str, query: str,
    filters: dict[str, Any], search_mode: str, latency_ms: float,
) -> str:
    ensure_case(case_id, officer)
    returned = [
        {
            "rank": rank,
            "tracklet_id": row.get("tracklet_id"),
            "score": row.get("score"),
            "component_scores": row.get("component_scores", []),
        }
        for rank, row in enumerate(output.get("results", []), start=1)
    ]
    return record_event(
        case_id, officer, "search", original_query=query,
        normalized_query=output.get("rewritten_query"), filters=filters,
        search_mode=search_mode, returned_results=returned, latency_ms=latency_ms,
        metadata={
            "aggregation": output.get("aggregation"),
            "composition": output.get("composition"),
            "search_captions": output.get("search_captions", []),
            "result_count": len(returned),
        },
    )


def set_case_item(
    case_id: str, tracklet_id: str, officer: str, status: str,
    note: str | None, snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    ensure_case(case_id, officer)
    tracklet_id = _clean(tracklet_id, "tracklet_id", 200)
    if status not in {"pinned", "excluded"}:
        raise InvestigationError("status must be pinned or excluded")
    clean_note = str(note or "").strip()[:2000] or None
    with psycopg.connect(config.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT status, note FROM case_items WHERE case_id=%s AND tracklet_id=%s",
                        (case_id, tracklet_id))
            previous = cur.fetchone()
            cur.execute(
                """INSERT INTO case_items (
                     case_id, tracklet_id, status, note, result_snapshot
                   ) VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (case_id, tracklet_id) DO UPDATE SET
                     status=EXCLUDED.status, note=EXCLUDED.note,
                     result_snapshot=EXCLUDED.result_snapshot, updated_at=NOW()
                   RETURNING *""",
                (case_id, tracklet_id, status, clean_note, Jsonb(snapshot or {})),
            )
            row = cur.fetchone()
    event_type = "result_pinned" if status == "pinned" else "result_excluded"
    if previous and previous["status"] == status and previous["note"] != clean_note:
        event_type = "note_updated"
    record_event(
        case_id, officer, event_type,
        metadata={
            "tracklet_id": tracklet_id, "status": status, "note": clean_note,
            "previous_status": previous["status"] if previous else None,
        },
    )
    return _json_row(row)


def get_case(case_id: str) -> dict[str, Any] | None:
    with psycopg.connect(config.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT * FROM cases WHERE case_id=%s", (case_id,))
            case = cur.fetchone()
            if case is None:
                return None
            cur.execute(
                """SELECT ci.*, t.scene, t.camera_id, t.entity_type, t.subtype, t.color,
                          t.ts_start_s, t.ts_end_s, t.crop_refs, t.video_ref, t.plate_text
                   FROM case_items ci JOIN tracklets t ON t.tracklet_id=ci.tracklet_id
                   WHERE ci.case_id=%s
                   ORDER BY CASE ci.status WHEN 'pinned' THEN 0 ELSE 1 END,
                            t.ts_start_s, ci.created_at""",
                (case_id,),
            )
            items = cur.fetchall()
    return {"case": _json_row(case), "items": [_shape_item(row) for row in items]}


def get_timeline(case_id: str) -> list[dict[str, Any]]:
    with psycopg.connect(config.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """SELECT event_id, case_id, officer, event_type, occurred_at,
                          original_query, normalized_query, filters, search_mode,
                          returned_results, latency_ms, metadata
                   FROM search_events WHERE case_id=%s ORDER BY occurred_at, event_id""",
                (case_id,),
            )
            return [_json_row(row) for row in cur.fetchall()]


def pinned_items(case_id: str) -> list[dict[str, Any]]:
    board = get_case(case_id)
    return [] if not board else [row for row in board["items"] if row["status"] == "pinned"]


def _shape_item(row: dict[str, Any]) -> dict[str, Any]:
    refs = row.get("crop_refs") or []
    snapshot = row.get("result_snapshot") or {}
    return {
        **_json_row(row),
        "camera_label": config.camera_label(row["scene"], row["camera_id"]),
        "crop_url": snapshot.get("crop_url") or (f"/files/{refs[0]}" if refs else None),
        "video_url": f"/media/{row['scene']}/{row['camera_id']}" if row.get("video_ref") else None,
        "plate": row.get("plate_text"),
    }


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
              if isinstance(value, datetime) else value)
        for key, value in dict(row).items()
    }
