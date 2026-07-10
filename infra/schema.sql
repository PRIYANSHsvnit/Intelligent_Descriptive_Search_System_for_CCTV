-- CCTV descriptive-search schema. One row per tracklet.
-- Vector dims are LOCKED in Phase 0 (see apps/ingest/constants.py) — never change:
--   semantic_vector = 1152  (SigLIP2 so400m-patch16-224)
--   reid_appearance = 2048  (VeRi FastReID ResNet50-IBN, BNNeck feature)
--   reid_color      = 56    (HSV signature)
-- Every not-yet-computed capability is a NULLABLE column, so adding people/plates/
-- global_id later is UPDATE/append work, never a migration-and-rebuild.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tracklets (
  tracklet_id     TEXT PRIMARY KEY,           -- e.g. 'S01_c001_t7'
  scene           TEXT NOT NULL,              -- 'S01'
  camera_id       TEXT NOT NULL,              -- 'c001'
  -- what it is
  entity_type     TEXT NOT NULL,              -- 'vehicle' | 'person'
  subtype         TEXT,                       -- 'car','truck','bus','motorcycle','bicycle','person'
  color           TEXT,                       -- dominant color name (Phase 1); RANKING signal only, never a hard filter
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
  plate_text      TEXT, plate_conf REAL,      -- Phase 5 (bonus)
  person_attrs    JSONB                       -- Phase 5 (outfit colors, hat, bag...)
);

-- Vector index on the SEARCH vector only (cosine HNSW). Optional at a few-thousand
-- rows (flat scan is fine) — created here for when the table grows.
-- No index on reid_* — cross-camera matching is an offline batch job (which also
-- frees reid_appearance from pgvector's 2000-dim index cap).
CREATE INDEX IF NOT EXISTS tracklets_semantic_hnsw
  ON tracklets USING hnsw (semantic_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS tracklets_filter
  ON tracklets (scene, camera_id, entity_type, color);
