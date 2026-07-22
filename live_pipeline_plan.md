use tensorRT and let person detector run at every two frames (if that helps with speed)

---

- Commit the DB transaction last — after the crop files are written.
- Name crops deterministically (clip/tracklet-scoped) so a redo overwrites rather than duplicates — idempotent by name.
- GC orphan crops — any crop file with no DB row (from a rolled-back clip) is garbage; sweep them periodically.

---

An NVR is a Network Video Recorder — the box (or software) that sits on the CCTV network, receives the video streams from all the IP cameras, and records them to disk. It's the modern, IP-camera equivalent of the old analog DVR. In your setup, this is the "police's storage" I kept referring to — the thing that holds the raw footage.

"NVR file" = how that recorder actually stores the video on disk. NVRs don't save one giant continuous file per camera — they chunk each camera's stream into fixed segments, typically:

- a new file every N minutes (commonly 5, 10, 15, or 60 min), or sometimes size-based (e.g., a new file every 512 MB / 1 GB),
- named with camera ID + timestamp, e.g. c004_20260722_143000.mp4 (camera c004, 22 Jul 2026, 14:30:00),
- often in H.264/H.265 .mp4 or .mkv, sometimes a vendor-proprietary container.

---

# Session log — live tier made functional + perf-tuned (2026-07-22)

Goal that day: the `surat-live` scene wasn't showing in the frontend, and the live sweep ran
below realtime. Everything below is what we found and changed, so we don't re-derive it from
memory next time.

## 1. Why `surat-live` didn't show in the frontend
Data was fine (the live run *had* stored rows) — the **frontend scene picker was hardcoded**
to only `SUR01` and `S01` (`apps/frontend/app/page.tsx`, the `<select>` ~line 227). Added a
`surat-live` option. The `scene` state + all fetch calls were already wired correctly.

## 2. Live vs batch artifact parity (what "full parity" actually means)
The backend reads **only three per-camera artifacts**: `detections.npy` (bbox overlay via
`/tracklets/{id}/boxes`), `media/<cam>.mp4` (playback, gated on the DB `video_ref` column),
and `crops/` (static thumbnails). Everything else the batch pipeline writes is scaffolding
with **no live consumer**:
- `tracklets.json` — only the multi-pass batch stages read/rewrite it (it's the inter-stage
  hand-off). Live is single-pass and inserts straight to DB, so it needs no such file.
- `vec/*.npy` — only `evaluate_reid.py` + `pipeline/matcher.py` (both re-ID, excluded from live).

So "replicate everything except re-ID + VLM" reduces to **crops + `detections.npy` + media**.
Gaps we closed:
- **`detections.npy`**: live tier discarded per-frame boxes (`_SINK` cleared each frame). Now
  `live/engine.py` accumulates boxes and dumps `detections.npy` in the exact batch layout
  `(M,8) float32: frame,tid,x1,y1,x2,y2,conf,cls` (person rows carry `PERSON_CLS=100`).
  Verified the backend `boxes.tracklet_boxes()` reads real timelines from it. **Only helps
  runs made AFTER this change** — pre-existing rows have no boxes and can't be reconstructed.
- **`video_ref`**: live never set it → search returned `video_url: null`. `live/writer.py` now
  auto-fills `video_ref = "<scene>/<cam>/media/<cam>.mp4"` at insert (backend only checks
  truthiness). Confirmed auto-populated on all rows on the next run.
- **media MP4s**: the live tier does NOT transcode (no `media` stage). For `surat-live` the
  footage is the *same source AVIs* as `SUR01`, so we **symlink SUR01's already-transcoded
  MP4s** into `output/surat-live/<cam>/media/<cam>.mp4`. Offsets are identical
  (`cam_timestamp/{SUR01,surat-live}.txt` match) so player seeking stays aligned. If live
  footage were genuinely new, we'd need to wire the `media` stage into `run_live`.
- **cam labels**: created `footage_data/cam_labels/surat-live.json` (c001/c003/c004/c005; c002
  dropped as too dense). NOTE: `config._scene_labels` is `@lru_cache`'d → backend must restart
  (or `--reload` touch) to pick up a newly-added labels file.

Live rows have **no re-ID / global_id** (stage excluded) → the cross-camera trace map won't
populate for live scenes. By design.

## 3. OCR stays on CPU — decision + why (NOT a NixOS limitation)
`rapidocr_onnxruntime` + `open_image_models` are CPU ONNX. GPU OCR would need `onnxruntime-gpu`
which is **CUDA-12**, clashing with this env's **CUDA-13 torch** (`2.13.0+cu130`, bundled cu13
cublas/cudnn) — and both run in the *same process*, risking the torch GPU path. Even if
installed, the CUDA EP would fail to load (no cu12 libs) and **silently fall back to CPU**.
On **Windows** the clean path would be the **DirectML EP** (`onnxruntime-directml`, no CUDA) —
so GPU OCR is far more feasible there than on this NixOS box.
But the deciding factor is compute, not the install: profiling showed the **GPU is starved,
not saturated** (see §4), so OCR belongs on the otherwise-idle CPU. The CPU pressure people
worried about was **GMC, not OCR** (§4). Verdict: keep OCR on CPU.

## 4. Performance investigation (the big one)
Profiled a capped live run with a background `nvidia-smi` sampler. Added lightweight profiling
to `live/engine.py` (accumulates ultralytics `.speed` = preprocess/inference/postprocess, and
times SigLIP embed + crop-write/insert; prints a `[profile]` breakdown per cam — kept, it's
near-free).

**Finding A — the GPU was idle, the bottleneck was CPU.** GMC was on: median GPU util 27%,
power 37 W (never near the ~90–115 W ceiling), cool — i.e. idle-waiting, not thermal/clock
throttled. The `.track()` remainder (tracker + glue, CPU) was **60% of wall**.

**Fix A — disable GMC for fixed cameras.** Stock `botsort.yaml` sets
`gmc_method: sparseOptFlow` (ORB + optical-flow *camera-motion* compensation, per frame, per
tracker, on CPU) — pointless for bolted-down CCTV. Created
`apps/ingest/pipeline/trackers/botsort_static.yaml` (identical but `gmc_method: none`).
Measured on 800 frames of c004: **0.56× → 0.96× realtime**, tracker+glue 42.4s → 10.3s.
Wired a `LIVE_TRACKER` env override in `engine.py`; `run_live.py` now defaults `--tracker` to
the static yaml. (This applies to the **batch** pipeline too — same stock tracker — but flip
batch only after checking IDF1 on the CityFlow bench, since it has ground truth.)

**Finding B — thread oversubscription re-starved the GPU.** With GMC off but OCR on, the box
thrashed: **load-avg 25 on 12 cores**, GPU back to 19%. Cause: no thread caps set → every lib
(OpenCV decode, ONNX Runtime × 2 OCR workers, Torch/BLAS) each spawned ~12 threads. RapidOCR's
`config.yaml` defaults `intra_op_num_threads: -1` (all cores) for its det/cls/rec sessions.

**Fix B — cap the thread pools.** In `run_live.py`: `LIVE_CPU_THREADS` (default 4) → sets
`OMP/OPENBLAS/MKL/NUMEXPR` env *before imports* + `cv2.setNumThreads` + `torch.set_num_threads`.
In `live/ocr_worker.py`: build the detector/OCR ourselves with capped ONNX sessions
(`_capped_models(intra_threads=2)`) instead of `plate._models()` — plate.py stays frozen.
GOTCHA: RapidOCR's `update_global_to_module` overwrites the per-section (Det/Cls/Rec) thread
values from the **Global** section, so the working knob is the **non-prefixed**
`intra_op_num_threads=` kwarg (NOT `det_intra_op_num_threads=`, which gets clobbered). The
detector (`open_image_models`) takes a real `sess_options=`. Result: load-avg 25 → 7.7.

**After both fixes the GPU is the honest bottleneck** — detector inference is now 51–58% of
wall. Remaining sub-realtime is real GPU compute: two heavy detectors per frame (UVH-26
**YOLOv11-X @ imgsz 1280** + CrowdHuman person) + SigLIP embedding every tracklet.

## 5. Full-fps baseline (2026-07-22, 4 cams, GMC-off, threads capped, OCR on CPU)
Command: `uv run python run_live.py --scene surat-live --cams c001 c003 c004 c005`

| cam | tracklets | realtime | detect inf % | notes |
|----|----|----|----|----|
| c001 | 2,006 | 0.76× | 58% | light |
| c003 | 5,157 | 0.50× | 52% | crowd (Gopi Talav) |
| c004 | 1,813 | 0.68× | 52% | light |
| c005 | 5,945 | 0.41× | 51% | crowd (densest kept cam) |

TOTAL: **1,205 s footage in 2,176 s = 0.55× realtime**; OCR drain **+386.6 s** (2,575 vehicle
tracklets, 73 validated plates); end-to-end **2,575 s (~43 min)**.

**Per-cam speed is driven by scene density** — 2.5× more tracklets ≈ 2.5× more SigLIP embed +
2× more per-detection glue, both competing on the one GPU. c005/c003 (crowds) are slowest.

**OCR drain tail** (386 s) is partly self-inflicted: capped 2 workers can't keep pace with
2,575 vehicles during the sweep, so a backlog drains after the GPU work stops. It's
**non-blocking in true live** (plate UPDATEs are eventually-consistent; `run_live` only *waits*
on drain to print clean totals). To shrink it: `--ocr-workers 4` (CPU headroom exists now).

## 6. Knobs / commands
- `run_live.py --tracker <yaml>` — default `pipeline/trackers/botsort_static.yaml` (GMC off).
  Pass `botsort.yaml` to restore GMC.
- `LIVE_CPU_THREADS=N` (default 4) — process-wide CPU thread budget.
- `LIVE_TRACKER=<yaml>` — env override read by `LiveEngine.run`.
- `--stride 2` — process every other frame; ~halves detection+embed load → ≈2× realtime_x
  (tracking survives; a vehicle barely moves in 1/20 s). Free lever to clear realtime.
- `--ocr-workers N` (default 2), `--no-ocr` to measure pure GPU path.
- Backend must be **down** (or not hit `/search`) during a live run, or it pulls SigLIP
  (~2.15 GB) onto the GPU and contends with the live models (~2.7 GB).

## 7. Open items / next levers (ranked)
1. `--stride 2` — immediate, free path to ≥realtime.
2. `--ocr-workers 4` — cut the OCR drain tail.
3. **TensorRT / FP16** on the detectors + SigLIP — ~1.5–2× GPU throughput; the structural fix
   for full-fps realtime (VRAM fits; it's now a speed lever, not a fit lever).
4. Lower detector `imgsz` 1280→960 or a smaller YOLO variant — cheaper detection, small
   accuracy cost.
5. Validate GMC-off on CityFlow IDF1 before flipping the **batch** tracker default.
6. Wire the `media` stage into `run_live` so live is self-contained (only matters when live
   footage differs from a batch scene we can symlink from).

## 8. Files touched this session (all uncommitted as of 2026-07-22)
- `apps/frontend/app/page.tsx` — surat-live picker option
- `apps/ingest/pipeline/trackers/botsort_static.yaml` — NEW, GMC-off tracker
- `apps/ingest/run_live.py` — `--tracker` default, `LIVE_CPU_THREADS`, cv2/torch caps
- `apps/ingest/live/engine.py` — `detections.npy` persistence, `LIVE_TRACKER`, profiling
- `apps/ingest/live/writer.py` — auto-fill `video_ref`
- `apps/ingest/live/ocr_worker.py` — capped ONNX OCR sessions (`_capped_models`)
- `footage_data/cam_labels/surat-live.json` — NEW
- `output/surat-live/<cam>/media/<cam>.mp4` — symlinks to SUR01 transcodes (not in git)


