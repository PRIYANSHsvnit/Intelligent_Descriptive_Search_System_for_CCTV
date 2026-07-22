"""Incremental DB writer for the live tier — one row per tracklet, committed as it
finalizes (the "durably committed" unit). Reuses the frozen store stage's exact column
list + upsert SQL so live rows are schema-identical to batch rows (the frontend can't tell
them apart). No schema change: provisional/provenance is a later layer; live rows go under
their own ``scene`` and are isolated from the demo build by that alone.
"""

from __future__ import annotations

from typing import Any

from pipeline.db import connect
from pipeline.store import _COLS, _CROP_SQL, _SQL

_PLATE_SQL = ("UPDATE tracklets SET plate_text=%s, plate_conf=%s, plate_raw=%s "
              "WHERE tracklet_id=%s")


class DBWriter:
    """One psycopg connection. NOT shared across threads — the engine and the OCR pool
    each hold their own instance (see ocr_worker)."""

    def __init__(self) -> None:
        self.conn = connect()  # registers the pgvector adapter (np.ndarray -> vector)

    def insert(self, row: dict[str, Any], crop_rows: list[tuple] | None = None) -> None:
        # The live tier skips the batch `media` stage, so nothing sets video_ref. The
        # per-camera MP4 lives at the same rel key the media stage would emit; fill it
        # here so the backend returns a playable video_url (it only checks truthiness).
        if not row.get("video_ref"):
            row["video_ref"] = f"{row['scene']}/{row['camera_id']}/media/{row['camera_id']}.mp4"
        vals = tuple(row.get(c) for c in _COLS)  # _COLS order == _SQL placeholder order
        with self.conn.cursor() as cur:
            cur.execute(_SQL, vals)
            if crop_rows:
                cur.execute("DELETE FROM tracklet_crops WHERE tracklet_id=%s",
                            (row["tracklet_id"],))
                cur.executemany(_CROP_SQL, crop_rows)
        self.conn.commit()

    def update_plate(self, tracklet_id: str, text, conf, raw) -> None:
        with self.conn.cursor() as cur:
            cur.execute(_PLATE_SQL, (text, conf, raw, tracklet_id))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
