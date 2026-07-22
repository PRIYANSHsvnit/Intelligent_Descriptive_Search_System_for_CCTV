-- CCTV descriptive-search schema. One row per tracklet.
-- Vector dims are LOCKED in Phase 0 (see apps/ingest/constants.py) — never change:
--   semantic_vector = 1152  (SigLIP2 so400m-patch16-224)
--   reid_appearance = 2048  (VeRi FastReID ResNet50-IBN, BNNeck feature)
--   reid_color      = 56    (HSV signature)
-- Every not-yet-computed capability is a NULLABLE column, so adding people/plates/
-- global_id later is UPDATE/append work, never a migration-and-rebuild.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy plate search (trigram GIN indexes)

CREATE TABLE IF NOT EXISTS tracklets (
  tracklet_id     TEXT PRIMARY KEY,           -- e.g. 'S01_c001_t7'
  scene           TEXT NOT NULL,              -- 'S01'
  camera_id       TEXT NOT NULL,              -- 'c001'
  -- what it is
  entity_type     TEXT NOT NULL,              -- 'vehicle' | 'person'
  subtype         TEXT,                       -- 'car','truck','bus','motorcycle','bicycle','person'
  color           TEXT,                       -- dominant color name (Phase 1); RANKING signal only, never a hard filter
  color_conf      REAL,                       -- SigLIP zero-shot confidence for `color`; lets ranking demote unsure colors
  -- when (global scene clock)
  frame_start     INT,  frame_end   INT,
  ts_start_s      REAL, ts_end_s    REAL,     -- seconds from scene start
  wall_start      TIMESTAMPTZ, wall_end TIMESTAMPTZ,   -- SYNTHETIC display clock only (not real time); filter on ts_*_s
  -- evidence
  num_detections  INT,  avg_conf    REAL,
  crop_refs       TEXT[],                     -- RELATIVE keys to the K crops (grid thumbnails)
  video_ref       TEXT,                       -- RELATIVE key to per-camera web-playable H.264/MP4; UI seeks ts_start/ts_end
  -- ground position (Phase 3): H^-1 of box bottom-center -> GPS -> local meters
  ground_x        REAL, ground_y    REAL,
  -- vectors  (dims LOCKED — see header)
  semantic_vector vector(1152),              -- SigLIP so400m-patch16-224 (search)   Phase 1
  reid_appearance vector(2048),              -- VeRi re-ID appearance (tracing)      Phase 1
  reid_color      vector(56),                -- HSV color signature; fused at match time  Phase 1
  -- cross-camera + later work (nullable = additive)
  global_id       INT,                        -- Phase 3; scene-namespaced (don't collide S01#42 with S02#42)
  -- plates, two tiers (see pipeline/plate.py): plate_text = validated ASSERTION
  -- (constraint stack + voting, NULL = abstain); plate_raw = best raw OCR read,
  -- RECALL tier — never displayed as fact, only searched fuzzily.
  plate_text      TEXT, plate_conf REAL,
  plate_raw       TEXT,
  person_attrs    JSONB                       -- Phase 5 (outfit colors, hat, bag...)
);

-- Dimension tables: where cameras physically are. Replaces footage_data/cam_labels/*.json
-- as the source of truth for display names once populated (backend can fall back to the
-- JSON until then). Camera ids are scene-scoped ('c001' exists in both S01 and SUR01),
-- so the camera key is (scene, camera_id) — same pair tracklets already carries. No FK
-- from tracklets: ingest must never fail because a camera row wasn't registered yet.
CREATE TABLE IF NOT EXISTS locations (
  location_id  TEXT PRIMARY KEY,             -- e.g. 'SUR01_athwa_gate'
  name         TEXT NOT NULL,                -- "Athwa Gate Junction"
  geo_lat      DOUBLE PRECISION,             -- junction point for the trace-map UI
  geo_long     DOUBLE PRECISION,
  site_group   TEXT                          -- optional zone/precinct grouping
);

CREATE TABLE IF NOT EXISTS cameras (
  scene        TEXT NOT NULL,                -- 'SUR01'
  camera_id    TEXT NOT NULL,                -- 'c001'
  location_id  TEXT REFERENCES locations(location_id),
  label        TEXT,                         -- human-readable: "Athwa Gate, facing north"
  is_active    BOOLEAN DEFAULT TRUE,
  PRIMARY KEY (scene, camera_id)
);

-- Vector index on the SEARCH vector only (cosine HNSW). Optional at a few-thousand
-- rows (flat scan is fine) — created here for when the table grows.
-- No index on reid_* — cross-camera matching is an offline batch job (which also
-- frees reid_appearance from pgvector's 2000-dim index cap).
-- NOTE recall: pgvector's hnsw.ef_search defaults to 40 candidates, which drops true
-- neighbors once the table reaches ~25k rows. The backend raises it per query
-- (SET hnsw.ef_search, default 200 — see backend config.HNSW_EF_SEARCH), so recall is
-- controlled at query time rather than by the index build here.
CREATE INDEX IF NOT EXISTS tracklets_semantic_hnsw
  ON tracklets USING hnsw (semantic_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS tracklets_filter
  ON tracklets (scene, camera_id, entity_type, color);

-- One semantic vector per retained view. `tracklets.semantic_vector` remains as a
-- compatibility/fallback mean, while descriptive retrieval ranks each tracklet from
-- its individual crop scores (normally the mean of its best two query matches).
CREATE TABLE IF NOT EXISTS tracklet_crops (
  tracklet_id      TEXT NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
  crop_index       SMALLINT NOT NULL,
  frame_no         INT,
  crop_ref         TEXT NOT NULL,
  quality          REAL,
  semantic_vector  vector(1152) NOT NULL,
  PRIMARY KEY (tracklet_id, crop_index)
);

CREATE INDEX IF NOT EXISTS tracklet_crops_tracklet
  ON tracklet_crops (tracklet_id);
CREATE INDEX IF NOT EXISTS tracklet_crops_semantic_hnsw
  ON tracklet_crops USING hnsw (semantic_vector vector_cosine_ops);

-- Immutable receipt for every generated forensic package. The ZIP remains portable;
-- this row is the server-side chain-of-custody anchor used to identify its signer and
-- the manifest originally emitted by this deployment.
CREATE TABLE IF NOT EXISTS forensic_exports (
  export_id         UUID PRIMARY KEY,
  case_id           TEXT NOT NULL,
  officer           TEXT NOT NULL,
  tracklet_id       TEXT NOT NULL REFERENCES tracklets(tracklet_id),
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  package_path      TEXT NOT NULL,
  manifest_sha256   TEXT NOT NULL,
  source_sha256     TEXT NOT NULL,
  signing_key_id    TEXT NOT NULL,
  metadata          JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS forensic_exports_case_created
  ON forensic_exports (case_id, created_at DESC);

CREATE OR REPLACE FUNCTION reject_forensic_export_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'forensic export receipts are append-only';
END;
$$;

DROP TRIGGER IF EXISTS forensic_exports_append_only ON forensic_exports;
CREATE TRIGGER forensic_exports_append_only
BEFORE UPDATE OR DELETE ON forensic_exports
FOR EACH ROW EXECUTE FUNCTION reject_forensic_export_mutation();

-- Additive column for DBs created before the plate stage (CREATE TABLE IF NOT
-- EXISTS won't add it); harmless no-op on fresh DBs.
ALTER TABLE tracklets ADD COLUMN IF NOT EXISTS plate_raw TEXT;

-- Trigram indexes for the layered plate search (exact -> substring -> similarity).
CREATE INDEX IF NOT EXISTS tracklets_plate_trgm
  ON tracklets USING gin (plate_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS tracklets_plate_raw_trgm
  ON tracklets USING gin (plate_raw gin_trgm_ops);
