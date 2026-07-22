# Problem Statement Requirements — Implementation Status

Tracking sheet against the Surat Smart City problem statement
("Intelligent Descriptive Search System for CCTV").
✅ = implemented & verified · ⚠️ = partially implemented · ❌ = not yet implemented

## Key Objectives

- ✅ Natural-language descriptive search overlay for existing CCTV storage
- ✅ Retrieve timestamped frames/clips matching attribute or text queries across cameras
- ✅ Reduce manual footage-review effort (search over 25k+ indexed entities on SUR01)
- ✅ Forensically sound, exportable outputs for case documentation
  — signed portable ZIP with source/derived artifact roles, report, manifest, hashes,
  server-pinned verification, and an append-only database receipt

## I. Ingestion & Indexing

- ✅ **I.a** Index recorded footage from multiple cameras and timeframes
  — stage-major pipeline over SUR01 (5 cams) with per-camera time offsets and
  wall-clock timestamps; 22,064 person + 3,285 vehicle tracklets stored
- ✅ **I.b** Object, person, and vehicle detection with attribute extraction
  — UVH-26 YOLOv11-X (14 Indian vehicle classes) + CrowdHuman person detector;
  vehicle colour via SigLIP zero-shot; person clothing/accessories are retrieved from
  individual multi-view SigLIP vectors. Qwen3-VL is retained only as an optional ablation
  and is disabled in default ingest.
- ✅ **I.c** Searchable metadata index over detected entities
  — Postgres + pgvector (HNSW on 1152-d semantic vectors), btree filter index,
  trigram indexes for plates

## II. Descriptive Search

- ✅ **II.a** Natural-language and tag-based queries
  — `/search?q=` through SigLIP text tower with deterministic component captions,
  per-crop matching, and exact reranking
- ✅ **II.b** Filter by camera/location, time window, and attribute
  — **per-camera/location filter** (typeable Location combobox → `camera_id`,
  fed by new `/scene-cameras/{scene}`) and a **wall-clock time-of-day** picker
  (from/to → seconds-since-midnight `t0/t1`, per-camera coverage presets) now live
  in the frontend console; `/search` gained `camera_id` (+ `color`) params.
  Person clothing/accessories remain recall-first semantic ranking signals; explicit
  camera/time/entity filters are authoritative and never silently relaxed. Vehicle
  colour remains a vehicle-only structured filter.
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
- ✅ **III.c** Export of matching clips, annotated images, and reports
  — result player generates a case ZIP containing indexed source media, a padded selected
  clip, full-frame box annotation, and a PDF summary

## IV. Forensic Output

- ✅ **IV.a** Chain-of-custody metadata and integrity hashing on exports
  — `manifest.json` records case/officer/query/filter/camera/time/model/retrieval/crop
  provenance and artifact roles; `SHA256SUMS` covers all evidence and metadata artifacts;
  Ed25519 signs the checksum file; UI/CLI verification pins the deployment public key;
  `forensic_exports` keeps an append-only server receipt
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
- ⚠️ Lightweight models for edge/field deployment
  — default ingest excludes the 4B VLM and measured SigLIP peak allocation is ~2.63 GiB;
  live-tier throughput still varies by camera density and needs the stride-2 quality bench

## Deliverables

- ✅ Working prototype/demo on sample CCTV footage — end-to-end on real Surat
  footage (SUR01)
- ✅ Descriptive-search dashboard with worked queries — forensic-console dashboard
  live (unified plate/description search, Location + Time filters, results grid,
  clip player w/ box overlay); II.b filters are now typeable
- ✅ Sample forensic export with metadata — generated from SUR01 and verified end-to-end
- ✅ Documentation (models, indexing, integrity controls)
  — model/index/retrieval design plus forensic trust model, package layout, key handling,
  verification, and tamper-demo procedure are documented

## Some More things to Fix
- ✅ 1. HNSW recall — DONE. `backend config.HNSW_EF_SEARCH` (default 200, env-overridable)
  is applied per query via `SET hnsw.ef_search` in `engine._connect()`, lifting recall
  above pgvector's default 40 candidates (which drops true neighbors at SUR01's ~25k rows).
  `schema.sql` documents that recall is controlled at query time. Set 0 to disable.

- ✅ 2. Multi-view crop retrieval — DONE. `tracklet_crops` stores each retained SigLIP
  vector; HNSW retrieves a broad crop pool and exact reranking supports legacy mean,
  max, and best-two aggregation. SUR01 has 73,976 backfilled crop vectors. New detections
  select five crops from quality winners plus temporal samples and support configurable
  person-box expansion. Human relevance labelling remains the accuracy release gate.
    
- ✅ Prompt-template the text query. DONE via the same `query_rewrite.py` Groq hook:
  every query is rewritten to caption style ("man red shirt" → "a man in a red
  shirt") before SigLIP encoding, with a static `"a photo of {q}"` wrapper as the
  fail-open fallback (so the fixed-template win holds even without Groq). Negations
  ("without helmet") are preserved verbatim by the prompt.
