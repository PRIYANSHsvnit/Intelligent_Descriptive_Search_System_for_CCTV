# Problem Statement Requirements — Implementation Status

Tracking sheet against the Surat Smart City problem statement
("Intelligent Descriptive Search System for CCTV").
✅ = implemented & verified · ⚠️ = partially implemented · ❌ = not yet implemented

## Key Objectives

- ✅ Natural-language descriptive search overlay for existing CCTV storage
- ✅ Retrieve timestamped frames/clips matching attribute or text queries across cameras
- ✅ Reduce manual footage-review effort (search over 25k+ indexed entities on SUR01)
- ❌ Forensically sound, exportable outputs for case documentation

## I. Ingestion & Indexing

- ✅ **I.a** Index recorded footage from multiple cameras and timeframes
  — stage-major pipeline over SUR01 (5 cams) with per-camera time offsets and
  wall-clock timestamps; 22,064 person + 3,285 vehicle tracklets stored
- ✅ **I.b** Object, person, and vehicle detection with attribute extraction
  — UVH-26 YOLOv11-X (14 Indian vehicle classes) + CrowdHuman person detector;
  colour via SigLIP zero-shot; clothing/gender/age via Qwen3-VL `vlm_attrs`
- ✅ **I.c** Searchable metadata index over detected entities
  — Postgres + pgvector (HNSW on 1152-d semantic vectors), btree filter index,
  trigram indexes for plates

## II. Descriptive Search

- ✅ **II.a** Natural-language and tag-based queries
  — `/search?q=` through SigLIP text tower ("red hatchback", "man in yellow
  t-shirt" work); VLM attribute chips shown on results
- ⚠️ **II.b** Filter by camera/location, time window, and attribute
  — scene + entity-type filters work; **no per-camera filter**; time filter is
  scene-clock seconds only (**no wall-clock "8 PM–10 PM" filter**, though
  `wall_start`/`wall_end` are stored for every row); frontend exposes only
  scene + type
- ✅ **II.c** Search by reference image (person/vehicle re-identification)
  — `POST /search/image` (SigLIP image tower) + `GET /search/similar`
  (semantic | reid; reid falls open to semantic on SUR01 — no re-ID vectors stored)

## III. Results & Review

- ✅ **III.a** Timestamped frames/clips with camera ID and location
  — results carry camera label, real time-of-day, crop thumbnail; player seeks
  and loops the object's time window
- ⚠️ **III.b** Cross-camera tracking of the same target
  — full matcher + trace UI on the CityFlow bench (IDF1 0.417); on SUR01 only
  plate-based stitching (12 global-id groups) — cameras are non-overlapping
- ❌ **III.c** Export of matching clips, annotated images, and reports

## IV. Forensic Output

- ❌ **IV.a** Chain-of-custody metadata and integrity hashing on exports
- ❌ **IV.b** Search-history and audit logging

## Bonus Points

- ✅ Multi-language query support (Gujarati, Hindi)
  — Groq `llama-3.1-8b-instant` translate + caption-rewrite in front of SigLIP
  (`query_rewrite.py`); verified end-to-end on SUR01 — Hindi "पीली शर्ट में आदमी"
  and English "man in yellow shirt" rewrite to the same caption and return the
  same ranked results; fail-open to a static `"a photo of {q}"` template
- ✅ License-plate integration
  — two-tier plate OCR (`plate_text`/`plate_raw`) + layered exact/partial/fuzzy
  `/search?plate=`; plate-based tracklet stitching
- ❌ Face-recognition integration ("where permitted" — deliberately skipped)
- ❌ Live-feed snapshot scanning (pipeline is offline/batch)
- ❌ Lightweight models for edge/field deployment (pipeline is GPU-first by design)

## Deliverables

- ✅ Working prototype/demo on sample CCTV footage — end-to-end on real Surat
  footage (SUR01)
- ⚠️ Descriptive-search dashboard with worked queries — dashboard live; the
  statement's own worked queries need the II.b time/camera filters to be typeable
- ❌ Sample forensic export with metadata
- ❌ Documentation (models, indexing, integrity controls) — deferred to the end

## Some More things to Fix
- 1. HNSW recall — likely a real, free win. schema.sql builds the index with defaults, and the backend never sets hnsw.ef_search, so every query runs at pgvector's default of 40 candidates. That was fine at 1,233 CityFlow rows, but a full SUR01 ingest is ~24k+ tracklets — at that scale ef_search=40 will genuinely drop true neighbors, and it manifests exactly as "the search missed a clip that's obviously there." Honestly, at 30k rows × 1152 dims you can afford to skip the ANN index entirely and brute-force scan (tens of ms) for guaranteed 100% recall, or just SET LOCAL hnsw.ef_search = 200 per query. Your memory already flags ef_search as an open item — this is it coming due.

- K_CROPS = 3 + mean-pooling — the semantic-vector knob. Each tracklet's search vector is the mean of only its 3 "best" crops, where best = area × Laplacian sharpness (detect_track.py:54). Two things to consider:
    Raising K to ~5–7 adds viewpoint diversity to the mean cheaply (siglip pass is batched; cost is linear and offline). The area×sharpness metric also biases toward the closest-to-camera frame, which for fast vehicles is often the motion-blurred or most-occluded one — worth eyeballing what it picks.
    The bigger structural option: mean-pooling blurs away single-view attributes. A query like "text on the back of a truck" or "bag on left shoulder" matches one crop strongly and the mean weakly. Storing per-crop vectors and taking max-similarity per tracklet at query time is the highest-ceiling retrieval change available — but it's a schema/search rework (N rows per tracklet), so I'd file it, not rush it.
    
- ✅ Prompt-template the text query. DONE via the same `query_rewrite.py` Groq hook:
  every query is rewritten to caption style ("man red shirt" → "a man in a red
  shirt") before SigLIP encoding, with a static `"a photo of {q}"` wrapper as the
  fail-open fallback (so the fixed-template win holds even without Groq). Negations
  ("without helmet") are preserved verbatim by the prompt.
