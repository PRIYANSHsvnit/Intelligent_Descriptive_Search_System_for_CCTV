# Plan: Intelligent Descriptive Search System for CCTV

> This document is self-contained: every model name, parameter, threshold, formula, and
> data shape needed to build the system is written here. You do not need any prior code to
> execute it.

## Context — what we're building and why

**The product.** Point it at hours of CCTV from many cameras. A user *describes* an object in plain words — "white pickup truck", later "man in a red shirt and black cap" — and instantly gets every matching clip, **plus** a reconstruction of that object's path as it moves from camera to camera across the area.

**The dataset (a big advantage).** `footage_data/` is **CityFlow** — 6 scenes (S01–S06), **65 camera feeds** (46 unique cameras — S05 reuses S03/S04 IDs; quote "46 cameras" in the pitch), videos `vdo.avi` at **1920×1080, 10 fps**. Length varies a lot: S01–S03/S06 ~2–3 min (~2000 frames), but **S05 is 5–7 min (~3400–4300 frames)** and **S04 is very short (21–71 s)** — so per-camera frame counts differ, and S04's short clips + big time offsets mean its cameras barely overlap in time (the trajectory cue is S01-shaped, not universal). Crucially each camera folder also ships:
- `gt/gt.txt` — **ground truth**, MOTChallenge CSV: `frame, global_id, x, y, w, h, conf, -1, -1, -1`. **Column 2 is a global vehicle ID that is consistent across cameras** — i.e. the correct answer to "same car?". This lets us *measure* accuracy instead of guessing.
- `calibration.txt` — a **3×3 homography matrix** (stored **ground→image**; **invert it (`H⁻¹`) to go image→ground/GPS** — see Detail) + a reprojection error, letting us place objects on a real-world map.
- `roi.jpg` — a mask of the usable region (road area) for that camera.
- `det/`, `mtsc/`, `segm/` — precomputed detector/tracker/segmentation baselines we can ignore or use as references.

Camera location maps live in `footage_data/cam_loc/` (`S01.png`, `S02.png`, `S0345.png`, `S06.png`). The full 16 GB `cityflow.zip` **also contains** per-camera time offsets (`cam_timestamp/`, confirmed present) and CityFlow's 2019 scorer `eval/eval.py` (we compute IDF1 with a maintained library instead — see Verification — and keep `eval.py` only as an optional cross-check). Extract `cam_timestamp/` in Phase 0.

**Locked decisions.**
1. Demo = **descriptive search + cross-camera tracing**, **vehicles first**, people/plates added later with zero rework.
2. Search = **hybrid**: free-text semantic **and** structured filters (type, color, time).
3. Storage = **Postgres + pgvector** — local Docker now, a hosted Postgres (Supabase/Neon, both support pgvector) later so teammates connect over the internet with the *same* code and schema.
4. **GPU-first everywhere** (RTX 4050, 6 GB) — see the GPU section for the required NixOS driver fix.
5. **Two vectors per object**: a **semantic** vector (SigLIP) for text search and a **re-ID** vector (a VeRi-trained vehicle re-ID model + color) for tracing. Rationale in "The big idea" and the Re-ID section.

*(A prior throwaway spike validated the detect→track→crop flow and the appearance+color re-ID idea on S01. Its numbers and lessons are folded in below as our chosen parameters; the code itself is not a dependency.)*

---

## Quick glossary (plain-English)

- **Tracklet** — one object's journey through a *single* camera: every frame that camera saw it, stitched into one item ("the silver car in c001 from 20:00:04 to 20:00:19"). It is the basic unit we save, describe, and search. One real car passing 3 cameras = 3 tracklets; tracing later links them.
- **Detection** — one box around one object in one frame. Many detections over time → one tracklet.
- **Embedding / vector** — a fixed-length list of numbers describing something. Similar things get similar lists, so "are these alike?" becomes "are their lists close?" — fast math a database does in milliseconds.
- **Semantic vector** — the SigLIP embedding. Lives in the *same number-space as text*, so a typed sentence can be matched to images. Used for **search**.
- **Re-ID vector** — a VeRi-trained vehicle re-ID model's appearance features + a color signature. Tuned to tell *this exact object* from a look-alike. Used for **cross-camera tracing**.
- **global_id** — one ID shared by all tracklets that are the *same* physical object across cameras. Assigning these = the tracing feature.
- **Homography** — the 3×3 matrix relating a camera's flat image to the real-world ground plane (GPS), so we can reason about where an object physically is. CityFlow stores it **ground→image**, so we invert it (`H⁻¹`) to place an image point on the map.

---

## The big idea in one paragraph

We run every video **once, ahead of time** (not live) through a pipeline that finds each object, follows it within a single camera (a **tracklet**), and describes it with: (a) a **semantic vector** (SigLIP) that lives in the same space as text, (b) a **re-ID vector** (a VeRi-trained vehicle re-ID model's appearance fused with a color signature) tuned to distinguish look-alikes, and (c) plain attributes (type, color, time, later plate/outfit). All of that goes into Postgres. **Search** is then math: turn the user's sentence into a semantic vector, ask Postgres for the nearest tracklets (optionally filtered by color/type/time), return their clips. **Tracing** uses the re-ID vectors: tracklets from different cameras that match get one `global_id`, so a path is just "all tracklets with this global_id, ordered by time," drawn on the camera map.

Why two vectors: SigLIP is trained to match images to *captions*, so it nails "a white SUV" (great for search) but binds *identity* loosely (a white SUV and a similar white SUV look close to it). A vehicle re-ID model — trained specifically on "the same car seen by different cameras" — plus explicit color is tuned for fine appearance differences (great for "same exact car"). Using each where it's strong is the accuracy play. The schema carries both from day one.

---

## Architecture — two paths

### Path A — Offline Ingest (build the searchable index), run once per camera

```
 vdo.avi (1920x1080 @ 10fps)
   │
   ▼
[1 DECODE]  read frames at FULL 10 fps — do NOT subsample before        GPU-decode optional
            tracking (that fragments tracks; we subsample at [4])
   │
   ▼
[2 DETECT]  YOLO11 (m/l/x, not nano) at imgsz=1280 → boxes for           ★ GPU, half-precision
            classes {person, bicycle, car, motorcycle, bus, truck},
            confidence ≥ 0.3.  Optionally drop boxes outside roi.jpg.
            [DETECTOR IS PROFILE-SWAPPED — see "Models at a glance".
             india runs TWO detectors: UVH-26 YOLOv11-X (vehicles) +
             CrowdHuman YOLOv8n (people), merged frame-aligned.]
   │
   ▼
[3 TRACK]   BoT-SORT (bundled with Ultralytics) → links each object's    ★ GPU
            boxes across FULL-fps frames into ONE tracklet (persist);
            full fps keeps motion small so IDs don't fragment
   │
   ▼
[4 CROP]    per tracklet keep the K=3 best thumbnails (this is where we
            SUBSAMPLE — pick best crops, don't embed every frame),
            scored by area × sharpness — big + in-focus win
   │
   ▼
[5 ATTRIBUTES]  per tracklet:
              • type/subtype ← majority-vote of YOLO class over the track
              • color        ← name via SigLIP zero-shot; HSV 56-d kept as match feature (§ Detail)
              • time         ← frame → seconds → global scene clock (§ Detail)
              • plate_text   ← LATER (OCR, vehicles, bonus)
              • person_attrs ← LATER (VLM: gender/age/backpack/headwear + outfit colors)
   │
   ▼
[6 EMBED]   two vectors per tracklet (mean-pooled over its K crops),
            run as TWO disk-separated passes (see VRAM note):
              • semantic_vector ← SigLIP image encoder      (search)   ★ GPU
              • reid_vector     ← VeRi re-ID model + color  (tracing)  CPU (onnxruntime)
   │
   ▼
[7 STORE]   INSERT one row per tracklet into Postgres (pgvector),
            drop tracks with < 2 detections (detector noise)
```

**Per-camera media prep (once per camera, for playback — NOT per tracklet).** The UI shows footage by loading a camera's video and **seeking** to a tracklet's `ts_start`/`ts_end` (no per-clip cutting). Browsers play a short list of web formats (**H.264/MP4**, also VP9/AV1 in WebM) but **not** CityFlow's `msmpeg4v2` AVI, so give each camera one web-playable video (H.264/MP4 is the safe default) by **probing the source and doing the minimum** — never a blanket "transcode":
- already H.264/MP4 + faststart (likely for production MP4s) → **use as-is** (zero cost)
- MP4/H.264 without faststart → **remux**: `ffmpeg -i in -c copy -movflags +faststart cam.mp4` (near-instant, no re-encode)
- incompatible container/codec — CityFlow's **AVI/MS-MPEG4**, or **HEVC/H.265** (patchy browser support) → **transcode**: `ffmpeg -i in -c:v libx264 -movflags +faststart -g 30 cam.mp4`

The frontend seeks to the object (set `currentTime = ts_start`, pause on `timeupdate` at `ts_end`); **crops are the grid thumbnails, the seek is the playback.** Store the playable path as a per-camera `video_ref`, served with HTTP range requests. **Source format is a production unknown** — probe, don't assume (see "Target domain").

#### Step 6 explained (the linchpin — the two embeddings)

After step 4 we have ~3 clean thumbnails of one object. Step 6 turns them into numbers, because a computer compares numbers instantly but can't compare raw pictures. We compute **two** number-lists per tracklet, each for a different job:

**semantic_vector — SigLIP, for text search.**
- SigLIP takes an image and returns a fixed list of numbers (768 for `siglip-base`, 1152 for the larger `so400m`). It was pre-trained on billions of image+caption pairs so that **images and the words that describe them land in the same space**.
- That shared space is the whole basis of search-by-description: type "white SUV" → SigLIP's *text* encoder makes a vector → the database finds tracklets whose *image* vectors are nearest → those are the matches. No tags or keywords needed.
- We run SigLIP on each of the 3 crops and **average** the results into one steady vector (a single blurry angle can't skew it).

**reid_vector — a VeRi-trained vehicle re-ID model + color, for tracing.**
- **Appearance:** a vehicle re-ID model *trained on the exact task "same car, different camera"* (on the VeRi dataset) turns each crop into an appearance descriptor that's good at separating look-alikes. Averaged over crops, L2-normalized. (A generic model like DINOv2 can stand in but is far weaker at vehicle identity — kept only as an ablation.)
- **Color:** an explicit HSV color signature (56 numbers, see Detail) because appearance models bind color loosely. Averaged over crops.
- These two are **fused** into the re-ID vector: `L2normalize(concat[appearance, w·color])`, with a weight `w` we tune. This fusion is what turns weak "kind of looks similar" into reliable "same object."

**Why two different models (the analogy).** It comes down to what each was trained to do. SigLIP learned from billions of **image + caption** pairs, so it understands language and put text and images in one shared space — it's a **librarian**: describe what you want in words and it hands you matching pictures ("find me a white pickup"), but ask it to tell two white pickups apart and it just shrugs "both are white pickups." The **re-ID model** was trained on the exact task *"is this the same specific vehicle another camera saw?"* (thousands of cars shot by many traffic cameras), so it's a **detective**: it can't take a text query, but put two photos side by side and it's excellent at "same exact vehicle, or two different ones?" — it keys on the dent, the trim, the wheel pattern. Our two features need exactly these two talents: **search = words→pictures = the librarian (SigLIP)**; **tracing = is-this-the-same-one = the detective (the re-ID model + color)**. They're two specialists, not competitors. (SigLIP *can* do re-ID too, just far less accurately — its features group by category, not identity — which is why we spend the second vector.)

**We never train these models ourselves.** Both are pre-trained — SigLIP on image+captions, the re-ID model on VeRi vehicles — and we download and run them as-is (like a calculator). When new footage arrives, we run these *same* frozen models on the new clips and `INSERT` the new vectors alongside the old — purely additive, no rebuild, no retrain. The only thing that would force recomputing every vector is *switching to a different model* (its numbers wouldn't be comparable) — so we lock the two models now and keep them.

### Cross-camera Re-ID (offline, after a whole scene is ingested)

> **"Tracking" vs "Re-ID" — two different jobs, don't conflate them.**
> - **Within ONE camera = tracking (BoT-SORT, stage 3).** Links a car's boxes frame-to-frame into one tracklet as it crosses a single camera's view. The re-ID model is *not* involved here.
> - **Across DIFFERENT cameras = re-ID / tracing (the re-ID model + color, this section).** Different cameras share no IDs (c001 calls it "track 7", c002 calls it "track 3"). **The VeRi-trained re-ID model fused with the color signature (the `reid_vector`) is the engine that says "same car" and assigns one `global_id`** — this is the "follow a car across the city" feature. (The re-ID model is the main clue; color is the tie-breaker; the space-time trajectory check is the filter.)
> ```
> c001: BoT-SORT → S01_c001_t7 ─┐
> c002: BoT-SORT → S01_c002_t3 ─┼─► re-ID model+color say "same" → global_id = 42
> c005: BoT-SORT → S01_c005_t9 ─┘     → trace = these 3, ordered by time
> ```

**The problem.** Tracking gives each object a tracklet, but IDs are *local*: c001 calls a car "track 7", c002 calls the same car "track 3" — nothing links them.
```
BEFORE:  c001:track7  c002:track3  c005:track9   (all the same silver sedan)
AFTER:   all three → global_id = 42
```
Then tracing is one query: `... WHERE global_id = 42 ORDER BY time`.

**Why it's hard.** Same car looks different across cameras (viewpoint, lighting, distance) and different cars can look identical (two white Corollas). So we fuse **independent clues — two always-on, one optional**:

1. **Appearance similarity (main clue)** — cosine similarity between two tracklets' **reid_vectors** (the re-ID model's appearance part). Robust to viewpoint, captures shape/type/detail.
2. **Color agreement (tie-breaker)** — the color part of the reid_vector. Appearance models rank a white and a dark SUV as similar; the explicit color signal breaks that tie. (Adding color is the single biggest cheap win — in an early spike it lifted rank-1 from ~7% *appearance-only* to ~20% *appearance+color*; that ~20% is the "generic baseline" referenced elsewhere.)
3. **Space-time trajectory consistency (OPTIONAL booster — only where cameras are calibrated + time-synced)** — a *positive* cue: do the two tracklets trace a consistent path through space and time? After projecting both to the shared GPS frame (High #1: `H⁻¹`), at **overlapping timestamps** the same car's positions should be **close** (~15–30 m of slack); at **non-overlapping** times the implied speed should be **plausible** (≤ ~40 m/s). Note this is **NOT an "exit A → entrance B" veto**: S01's five cameras *overlap* (the same car is visible in several at once), so spatial *agreement at the same instant* is one of the **strongest** cues — not an impossibility to reject. **A bonus signal, not a foundation** — a deployment without calibration/sync (e.g. uncalibrated street CCTV) switches it off and leans on the always-on clues. See "Target domain: Indian roads vs CityFlow" for why we never let it be load-bearing.

**Combined score** = appearance+color cosine (the reid vectors already blend both); where geometry exists, also apply the space-time trajectory-consistency check.

**Pairwise → groups (with guardrails, NOT naïve chaining).** The dangerous way is **connected components**: "if A~B and B~C then A, B, C are one car." A *single* wrong link chains look-alikes into one giant blob, and nothing ever un-merges — on a busy scene of white sedans it melts the whole intersection into one `global_id`. Instead we group with rules that can say **no**:
- **Cannot-link — always-on, universal:** two tracklets that **overlap in time in the same camera** can never be the same object (one car can't be two simultaneous boxes). *Optional, where geometry exists:* also forbid physically impossible pairs (far apart at the same instant).
- **Mutual nearest neighbor:** only link A–B if each is the *other's* best match.
- **One-to-one per camera pair (Hungarian):** each tracklet matches ≤ 1 tracklet in any other camera — structurally bounds how large a group can grow.
- **Constrained agglomerative clustering:** merge highest-similarity pairs first, but **reject any merge that violates a cannot-link** within the resulting cluster; stop below threshold.

Same inputs as connected components, but a single bad link stays local instead of cascading. Each tracklet ends up in a cross-camera group or solo.

**Proving it (our headline metric).** `gt.txt` holds the true global IDs, so we score our predictions with the metrics defined in Verification. For reference, an early spike with a *generic* appearance model managed only **~20% rank-1 / ~35% rank-5** — which is exactly why the shipped re-ID encoder is a **VeRi-trained vehicle re-ID model** (see [6]), not a generic one. A VeRi-trained encoder (see [6]) plus space-time gating and better color/ROI masking should give a **large lift** over that generic baseline — but 97%-on-VeRi ≠ 97%-on-CityFlow (viewpoint/resolution domain gap), so **measure it with `evaluate_reid.py`, don't promise a number.**

**Generalizes to people** — same machinery on person tracklets, swapping the *vehicle* re-ID model for a *person* re-ID model (+ outfit colors + the same space-time trajectory check). Caveat: this dataset has no person ground truth, so person tracing can't be *scored* here (see Risks).

### Path B — Online Search (the live demo)

```
 "white pickup truck"  +  optional filters {type:vehicle, time:T+…}
        │                              │        (color is NOT a hard filter)
        ▼                              │
 SigLIP TEXT encoder → q-vector        │
        │                              ▼
        └──►  Postgres:  SELECT ... WHERE type/time match      (hard = safe filters only)
                         ORDER BY semantic_vector <=> :q        (SigLIP carries "white")
                         [+ optional soft color-agreement boost]   LIMIT 20
        │
        ▼
 ranked tracklets → thumbnails + clips
        │
        └── click a result → look up its global_id
                             → all tracklets with that id, ordered by time
                             → draw the PATH on the scene map (cam_loc/*.png) + a timeline
```

**Color is a ranking signal, never a hard filter.** Most traffic is white/black/gray/silver — achromatic colors where our HSV label is unreliable (shadow makes white read gray, sun makes silver read white). A hard `WHERE color='white'` would then *silently exclude* a white car mislabeled "silver" and give the user no hint anything's missing — a precision/recall trap that makes the search look broken with no error to debug. So: let **SigLIP carry the color word** in the free-text ranking (it does this fine on 100px+ crops); if you keep a structured color control, make it a **soft boost or a top-2 match**, not an equality filter; and **fail open** — if any filter starves the results, auto-fall-back to the unfiltered semantic ranking rather than showing an empty grid. (Neat asymmetry: the same wrong color barely hurts *re-ID*, since both tracklets of one car under the same light share the same wrong label — so color is dangerous as a search *filter* but fine as a re-ID *tie-breaker*.)

**Dedup the results.** One physical object can still surface multiple times (fragments of one car, or several tracklets sharing a `global_id`). Always **collapse near-duplicates** in the search response — same `global_id`, or same-camera fragments of one object — so each object appears once. This is the must-have safety net for the demo even if the per-camera fragment-merge pass misses one.

---

## Tech stack & why

| Layer | Choice | Why |
|---|---|---|
| Ingest CV | Python + `uv`, PyTorch (**GPU**), Ultralytics **YOLO11** + **BoT-SORT** | one library does detect+track; GPU-ready |
| Semantic model | **SigLIP** (`transformers` or `open_clip`) | image+text in one space → search-by-description; works for cars **and** people from one model |
| Re-ID model | **VeRi-trained vehicle re-ID** (FastReID weights via ONNX, run on **CPU**; see [6]); DINOv2 = ablation only | trained on "same car, different camera" — the accuracy engine for tracing |
| Color | OpenCV **HSV histogram** (56-d match feature) + **SigLIP zero-shot** name | HSV gives the structured match signal; SigLIP *names* the color (the HSV name was ~70% wrong on white/gray/silver) |
| Plate (later) | **PaddleOCR / EasyOCR** | read plates; nullable bonus |
| Store | **Postgres 16 + pgvector** (Docker → Supabase/Neon) | metadata + both vectors + filters in one SQL query; internet-accessible for the team |
| Backend | **FastAPI** (`apps/backend_py`) | already scaffolded; add real endpoints |
| Frontend | **Next.js 16** (`apps/frontend`) | already scaffolded; add search UI + map |

---

## Models at a glance

The authoritative *what runs where* for the pipeline — exact IDs, device, and output dim. (The "Tech stack" table above gives the *why*; this one is the quick lookup. Stage numbers map to Path A.)

| Stage | Model / engine | Exact ID / file | Device | Output (dim) |
|---|---|---|---|---|
| [2] detect | YOLO — **profile-swapped** (india = 2 detectors) | **cityflow:** `yolo11m.pt` — one COCO model for vehicles **and** people (imgsz 1280, conf 0.3, ids 0/1/2/3/5/7). **india:** vehicles `UVH-26-MV-YOLOv11-X.pt` (imgsz 640, all 14 Indian classes) **+** people `crowdhuman/yolov8n_crowdhuman.pt` (CrowdHuman, class {0:'person'}, imgsz 1280, conf 0.25) | GPU fp16 | boxes + class |
| [3] track | **BoT-SORT** | `botsort.yaml` (bundled with Ultralytics) | GPU | track ids |
| [5] color signature | HSV histogram (no ML) | 12×4 H×S + 8 V | CPU / OpenCV | `reid_color` (56) |
| [6a] semantic | **SigLIP2** image encoder | `google/siglip2-so400m-patch14-224` | GPU fp16 | `semantic_vector` (1152) |
| [6b] color name | **SigLIP2** text encoder (same model, zero-shot) | `google/siglip2-so400m-patch14-224` | GPU | `color` name (10-color vocab) |
| [6c] appearance | **FastReID VeRi ResNet50-IBN (SBS)** — **profile-swappable** | `veri_sbs_R50-ibn.pth` → `veri_reid.onnx` (256×256 RGB) | **CPU** onnxruntime | `reid_appearance` (2048) |
| [6e] person attributes / VLM (persons only) | **Qwen3-VL-4B-Instruct** (UD-Q6_K_XL GGUF) via **llama.cpp llama-server** — one multi-image request per tracklet (all K crops), JSON answer with "unknown" allowed, see note | `models/QWEN VLM/Qwen3-VL-4B-Instruct-UD-Q6_K_XL.gguf` + `mmproj-F16.gguf`; server `models/llama-cpp-cuda` (nix CUDA build; **`-c 2048` required on 6 GB**) | GPU (~4.7 GB, freed after the pass) | `person_attrs`: apparent_gender/age/backpack/headwear **+ upper/lower_color** (JSONB `{name}`; color vocab constrained in the prompt, off-vocab abstains). Sole source — the former [6d] region-split SigLIP outfit-color stage was **removed Jul 2026**, see the person-attributes note |
| search (Path B) | **SigLIP2** text encoder (same model) | `google/siglip2-so400m-patch14-224` | **CPU** | query vector (1152) |

**Notes that matter (reasoning, not repetition):**
- **SigLIP2 `so400m-patch14-224` is used in three places** — image embed (index), color naming, and query-time text embed. It's **locked** to 1152-d and must be byte-identical between ingest and the backend (`apps/backend_py/app/config.py` literally warns "Do not diverge"). Swapping it forces re-embedding every row.
- **Color naming moved HSV → SigLIP zero-shot.** The HSV 56-d signature ([5]) stays as a *matching* feature (fused in the matcher, default weight 0); the color **name** users see is now SigLIP zero-shot (prompt-ensembled over 10 colors), which fixed the ~70% HSV mislabels on achromatic paint. This **supersedes** the "dominant HSV bin → name" note in Detail [5].
- **What's profile-swapped** (CityFlow ↔ India): the **detector(s)** and the **re-ID ONNX**. CityFlow runs *one* COCO YOLO for vehicles+people. India runs *two* detectors in parallel because the vehicle model (IISc **UVH-26 YOLOv11-X**, 14 Indian classes) has **no person class**, so it's paired with a dedicated **CrowdHuman YOLOv8n** person detector; the two `.track()` streams are zipped frame-aligned into separate id namespaces (`t`=vehicle, `p`=person) and merged with pedestrian-FP suppression (a two-wheeler/bicycle box a same-footprint person box explains is dropped; real riders kept). SigLIP, color, BoT-SORT, matcher, DB, API, UI are all domain-agnostic and shared.
- **Why CrowdHuman *nano* for india people (not a bigger model).** A/B on real Surat footage (SUR01, Jul 2026): CrowdHuman YOLOv8n **beat** both stock YOLO11-X-COCO and a proprietary YOLOv8-X person model on dense-crowd recall — **training-domain match (packed, occluded crowds) dominates model size**. The x-COCO/x-proprietary models were trained on cleaner, sparser people and under-fire in Indian street density. `imgsz=1280` (native-res recovers small/distant peds; the remaining far-crowd misses are a **resolution wall** only tiling/SAHI would break), `conf=0.25` (distant dets are low-confidence; BoT-SORT prunes single-frame flicker). Upgrade path if crop quality ever demands it: fine-tune a **CrowdHuman-*m/l*** ourselves (domain-matched *and* bigger) — not an off-the-shelf big model on the wrong domain.
- **Locked dims (never fork):** semantic 1152 / reid_appearance 2048 / reid_color 56.
- **CPU vs GPU:** re-ID and the entire live search path run on **CPU by design** (plain onnxruntime dodges the NixOS CUDA-discovery pain); detect + SigLIP run on GPU.

---

## GPU-first (required setup, RTX 4050 6 GB on NixOS)

The GPU is mandatory for reasonable speed of the heavy per-frame work — **YOLO detection + SigLIP embedding** (the re-ID model deliberately runs on CPU, see [6]). Two machine-specific gotchas, both must be handled or torch silently falls back to CPU / fails to import:

1. **NixOS driver path.** The pip/uv torch wheel looks for `libcuda.so` in `/usr/lib`, but on NixOS the driver lives at `/run/opengl-driver/lib`. **Before importing torch**, prepend that directory to `LD_LIBRARY_PATH` and re-exec the process once. Implement this as a tiny shared module imported first by every ML entry point:
   ```python
   # gpu_setup.py — import and call this BEFORE importing torch/ultralytics
   import os, sys
   def ensure_gpu_libs():
       driver = "/run/opengl-driver/lib"
       if not os.path.exists(f"{driver}/libcuda.so"):
           return                                   # not NixOS → no-op (teammates on Win/Fedora)
       if driver in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
           return                                   # already set → avoid infinite re-exec
       os.environ["LD_LIBRARY_PATH"] = f"{driver}:" + os.environ.get("LD_LIBRARY_PATH", "")
       os.execv(sys.executable, [sys.executable, *sys.argv])
   ```
2. **OpenCV import.** Use **`opencv-python-headless`** — the regular `opencv-python` wheel fails to import here (missing `libxcb.so.1`); we only write files, so headless is correct anyway.

**Rules for every ML entry point:** call `ensure_gpu_libs()` first; then **assert `torch.cuda.is_available()` and log the device** — a CPU fallback should be loud, not silent. Run models in **half-precision (fp16)** to fit 6 GB. Pick the largest YOLO that fits (start `yolo11m`; try `l`/`x` if VRAM allows).

**Memory: why 6 GB is plenty.** VRAM holds two things: **weights** (the model's fixed numbers, loaded once) and **activations** (temporary intermediate results during a forward pass). Weights here are tiny — and only **YOLO11m (~40 MB)** and **SigLIP-base (~400 MB; so400m ~1.75 GB)** sit on the GPU (the re-ID model runs on CPU), so GPU weights total well under ~1 GB. The only real variable is **activations**, which scale with **batch size** (all models) and **input resolution** (YOLO on full frames is the spiky one). Two things keep us safe:
- **Inference, not training** — we only *run* the models, so each layer's activations are freed once consumed; peak memory ≈ one layer's worth, not the whole network. (Training would keep them all and cost far more.) There's also no LLM-style KV cache to accumulate.
- **Staged passes** — the pipeline runs one model at a time (crops persist to disk between stages), so each pass has ~6 GB to itself:
  ```
  Pass 1 (per camera): YOLO detect+track → tracklets + crops to disk → free model
  Pass 2:              crops → SigLIP → semantic_vectors → DB → free model
  Pass 3:              crops → re-ID model (CPU / plain onnxruntime) → reid_vectors → DB
  ```
  Free a model between passes with `del model; torch.cuda.empty_cache()`.

If any single pass gets tight, **lower the batch size** (embeddings) or **`imgsz`/batch** (YOLO) — since ingest is offline, that only costs wall-clock time. At live search time only SigLIP's **text** encoder is resident (a few hundred MB); YOLO and the re-ID model aren't loaded at all. Check real peak with `torch.cuda.max_memory_allocated()` and assume ~1.3–1.5× napkin math for allocator/cuDNN overhead.

---

## Database schema (Postgres + pgvector)

One row per tracklet. Every capability that isn't computed yet is a **nullable** column, so adding people/plates/global_id later is `UPDATE`/append work — never a migration-and-rebuild.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE tracklets (
  tracklet_id     TEXT PRIMARY KEY,           -- e.g. 'S01_c001_t7'
  scene           TEXT NOT NULL,              -- 'S01'
  camera_id       TEXT NOT NULL,              -- 'c001'
  -- what it is
  entity_type     TEXT NOT NULL,              -- 'vehicle' | 'person'
  subtype         TEXT,                       -- 'car','truck','bus','motorcycle','bicycle','person'
  color           TEXT,                       -- color name via SigLIP zero-shot (Phase 1); RANKING signal only, never a hard filter
  -- when (global scene clock; see Detail)
  frame_start     INT,  frame_end   INT,
  ts_start_s      REAL, ts_end_s    REAL,     -- seconds from scene start
  wall_start      TIMESTAMPTZ, wall_end TIMESTAMPTZ,   -- SYNTHETIC display clock only (not real time); filter on ts_*_s
  -- evidence
  num_detections  INT,  avg_conf    REAL,
  crop_refs       TEXT[],                     -- RELATIVE keys to the K crops (grid thumbnails); see Media storage
  video_ref       TEXT,                       -- RELATIVE key to per-camera web-playable H.264/MP4; UI seeks ts_start/ts_end
  -- ground position (Phase 3): H⁻¹ of box bottom-center → GPS → local meters
  ground_x        REAL, ground_y    REAL,
  -- vectors  (dims LOCKED in Phase 0 — never change)
  semantic_vector vector(1152),              -- SigLIP so400m-patch14-224 (search)  Phase 1  [LOCKED]
  reid_appearance vector(2048),              -- VeRi re-ID model appearance (tracing); dim set by encoder  Phase 1
  reid_color      vector(56),                -- HSV color signature; fused with appearance at match time  Phase 1
  -- cross-camera + later work (nullable = additive)
  global_id       INT,                        -- Phase 3; scene-namespaced (matcher runs per scene — don't collide S01#42 with S02#42)
  plate_text      TEXT, plate_conf REAL,      -- Phase 5 (bonus)
  person_attrs    JSONB                       -- Phase 5 (outfit colors, hat, bag…)
);

-- vector index on the SEARCH vector only (cosine HNSW). At a few-thousand rows a flat scan
-- is already fine, so this is optional early — created here for when the table grows.
-- No index on reid_* — cross-camera matching is an offline batch job (which also frees
-- reid_appearance from pgvector's 2000-dim index cap).
CREATE INDEX ON tracklets USING hnsw (semantic_vector vector_cosine_ops);
CREATE INDEX ON tracklets (scene, camera_id, entity_type, color);
```
Search query shape: `... WHERE entity_type='vehicle' [AND time range] ORDER BY semantic_vector <=> :q LIMIT 20;` (`<=>` = cosine distance). **Note:** only hard-filter on *reliable* fields (type, time). **Do not** hard-filter on `color` — it's a soft ranking signal (SigLIP carries the color word; add an optional score boost for color agreement). See "Color is a ranking signal, never a hard filter" above.

**Media storage (share the pixels, not just the rows).** Postgres holds only the *rows* — including the *paths* to crops and videos, not the files themselves. If you host the DB (Supabase/Neon) but leave crops/MP4s on your laptop, a teammate gets tracklet rows whose `crop_refs`/`video_ref` point at files only you have → broken thumbnails, no playback. The blob store must be shared *alongside* the DB. Two options:
- **Object storage (clean):** upload crops + per-camera MP4s to a blob store (Supabase Storage pairs with hosted Supabase Postgres); store the public/signed URLs. Works from anywhere.
- **Tunnel (hackathon-fast):** keep files local, FastAPI mounts the output dir (`/media/...`, with range-request support), expose the backend via ngrok / Tailscale / cloudflared. No upload step, but all traffic flows through your machine being online.

**Lock now (cheap now, migration later):** store **relative keys** in `crop_refs`/`video_ref` (e.g. `S01/c001/crops/t7_0.jpg`), never laptop-absolute paths like `/home/zer0/...`. The API resolves a key → the right URL, so switching tunnel ↔ object-store is a config change, not a DB rewrite.

### Going live: recurring hourly ingest (DEFERRED — do NOT build for the demo)

The schema above assumes scene-based one-shot ingest (run once per camera, done). Production
use is different: police use it every day, footage arrives continuously, and we ingest it as
an **offline batch every hour or so**. The columns barely change, but four things do — written
down now so going live is a checklist, not a redesign. None of this is needed while ingest is
one-shot per scene.

1. **Batch-stamp `tracklet_id` (the one correctness item — MUST land with the first recurring
   ingest).** Today `tracklet_id = {scene}_{cam}_t{tid}` where `tid` is the tracker's per-run
   counter, which resets every run. Hour 2 therefore re-produces `SUR01_c001_t7`, and the
   store stage's `ON CONFLICT DO UPDATE` upsert makes hour 2 **silently overwrite hour 1's
   rows** (crop files under the same cam dir collide the same way). Fix: put the batch window
   in the id and output paths, e.g. `SUR01_c001_2026071514_t7`. Bonus: re-running a *failed*
   hour upserts the same ids — hourly ingest becomes naturally idempotent.

2. **Wall time becomes the real query axis.** Police queries are wall-clock ("yesterday
   14:00–16:00, camera X") and `ts_start_s` (seconds from scene start) stops meaning anything
   when the "scene" runs forever. `wall_start/wall_end` flip from synthetic display clock to
   real timestamps (batch start + in-clip offset); the backend filters on them instead of
   `ts_*_s`. Rows arrive in time order, so a **BRIN index on `wall_start`** is nearly free and
   ideal.

3. **Partition `tracklets` by day.** The scale math makes this real, not hypothetical: the
   c004 smoke test saw ~133 person tracklets in 30 s at a busy junction — with vehicles that's
   ~25–30k tracklets/hour/camera at peak, plausibly **1–3M rows/day across 5 cams** (~13 KB of
   vectors per row → tens of GB/day). Day partitions give: (a) **retention = `DROP PARTITION`**
   (instant, no vacuum debt) — Indian CCTV practice is ~30–90 day retention, and the sweep must
   also delete the crop/video files, which dwarf the DB; (b) a **local HNSW index per day**, so
   index builds stay small; (c) time-scoped queries prune to 1–2 partitions, which also
   neutralizes pgvector's filtered-ANN starvation (HNSW returns top-ef candidates *before* the
   WHERE runs — a tight time filter over a global index starves; over a pruned partition it
   doesn't). Note: the PK must then include the partition key (`wall_start`).

4. **`ingest_batches` bookkeeping table** — what makes an unattended hourly job operable:

   ```sql
   CREATE TABLE ingest_batches (
     camera_id  TEXT NOT NULL,
     hour_start TIMESTAMPTZ NOT NULL,        -- the footage window this batch covers
     file_path  TEXT,                        -- raw footage for this window (cold storage)
     status     TEXT NOT NULL,               -- 'running' | 'done' | 'failed'
     row_count  INT, error TEXT,
     started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
     PRIMARY KEY (camera_id, hour_start)
   );
   ```

   The scheduler asks it what's done / what to retry; ops sees ingest lag at a glance.
   `file_path` also makes this the pointer back to the raw footage of that window —
   a batch row *is* the video segment, so no separate `video_segments` table is needed.

Accepted trade-off: a tracklet straddling the hour cut is split in two. Search dedup already
collapses same-camera same-subtype time-overlapping fragments, so we accept the split rather
than engineer cross-batch track stitching. Optional when storage bites: `reid_appearance`
(8.2 KB/row, only read by the offline cross-cam matcher) can move to `halfvec` or get a
shorter retention than the searchable metadata.

---

## Implementation Detail — exact parameters (so nothing is ambiguous)

**[1] DECODE.** Decode with OpenCV/Ultralytics at **full frame rate (`vid_stride = 1`)**. **Do NOT subsample before tracking** — that's what fragments tracks (see [3]); the compute-saving subsample happens later at the embed stage. Source fps read from the file (fallback 10; note `c015` in S03 is 8 fps).

**[2] DETECT.** `ultralytics.YOLO("yolo11m.pt").track(..., imgsz=1280, stream=True)`. **Set `imgsz=1280`** (Ultralytics defaults to 640, which downscales 1080p ~3× and drops small/far vehicles — the "find every white truck" recall the GT metric won't catch); go 1920 if VRAM allows in fp16. Keep COCO class ids `{0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck}`. `conf=0.3`, `half=True`, `device=0`. Optional: load `roi.jpg`, drop detections whose box center falls outside the ROI mask (cuts off-road false positives).

**[3] TRACK.** Same `.track()` call with `tracker="botsort.yaml"`, `persist=True`, `stream=True`, at **full fps (no `vid_stride`)**. BoT-SORT links boxes by frame-to-frame overlap, so it needs small motion between frames: fast movers jump ~138 px/frame at 10 fps, but ~415 px if you skip to every 3rd frame — enough to break association and **spawn duplicate track IDs (fragmentation)**. Running full-rate keeps it locked on. BoT-SORT assigns a stable integer track id per object.

**[4] CROP (also where subsampling now lives).** For each detection, `crop = frame[y1:y2, x1:x2]`; score `quality = (h*w) * variance(Laplacian(gray(crop)))`. Keep the top **K=3** per track (min-heap: replace the weakest when a better one arrives). Save as `crops/t<id>_<0..2>.jpg`, best first. Because we only embed these K crops (not every frame), full-fps detection doesn't blow up embedding cost — the compute saving that used to come from `vid_stride` now comes from keeping just K crops per tracklet.

**[5] ATTRIBUTES.**
- *Subtype/type*: majority vote of the class id over all the track's detections (robust to per-frame flips). `entity_type = person` if class 0 else `vehicle`.
- *Color signature (56-dim)*: on the central region `crop[0.2h:0.8h, 0.2w:0.8w]` (skips road/background): convert to HSV; 2-D Hue×Saturation histogram, bins `[12, 4]` over ranges `[0,180]×[0,256]` (48 values) + 1-D Value histogram, `8` bins over `[0,256]` (so black/white/silver, where hue is meaningless, still separate). Concatenate → 56; normalize to sum 1; take element-wise `sqrt` (Hellinger, so histogram distance behaves under cosine); L2-normalize. **This 56-d signature is kept as a *match* feature only.** *(Superseded for naming: the `color` **column** is no longer the dominant HSV bin — it's now **SigLIP zero-shot** over a 10-color vocab, which reuses the already-computed `semantic_vector` and fixed the ~70% HSV mislabels on white/gray/silver. See "Models at a glance".)*
- *Time / global clock*: `ts = cam_time_offset + frame/fps`. Cross-camera tracing needs a *shared* clock; use CityFlow's per-camera `cam_timestamp` offsets (confirmed in the zip — e.g. S01: c001 0.0 → c005 2.24s). **S04's offsets reach ~176s**, so the offset-0 fallback is unsafe there — extract the real offsets, don't default. `ts_*_s` are **seconds from scene start** (this is what the UI time filter uses); `wall_*` is a **synthetic display clock only**, not real wall time.

**[6] EMBED** (mean-pool over the K crops, then L2-normalize each vector):
- *SigLIP semantic_vector*: **LOCKED in Phase 0** to `google/siglip2-so400m-patch14-224` (**1152-dim**), the schema hardcodes this dim. We picked the larger so400m over `base`/768 up front because search is the headline demo and so400m retrieves noticeably better; it fits 6 GB VRAM (only model resident during the SigLIP pass) and the +50% index size is trivial at a few-thousand rows. Preprocess per the model's processor. This same model's **text** encoder embeds the user's query at search time. (Never switch models later — it would force re-embedding every vector.)
- *Appearance (the re-ID model)*: the primary encoder is a **VeRi-trained vehicle re-ID model** — see the callout below for how to obtain and run it (ONNX export on CPU, or weights-into-ResNet50-IBN). Preprocess per that model's spec (typically resize to its training size, ImageNet-style normalization), batch ~128, mean-pool over the K crops, L2-normalize. *Ablation only* (to show how much worse a generic model is): DINOv2 ViT-S via `timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)` at 224×224 with ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]` → 384-dim.
- *reid_vector*: `L2normalize(concat[appearance, w · color(56)])`. Tune `w` by sweeping `{0, 0.5, 1, 1.5, 2, 3, 4, 5}` and picking the value that maximizes cross-camera rank-1 on S01 (sweet spot is usually small, ~1–2). The appearance dim depends on the encoder (DINOv2 ViT-S = 384; a VeRi ResNet-IBN = 2048), so **store `appearance` and `color(56)` as separate columns and fuse at match time** — this lets you re-tune `w` and swap encoders without re-embedding, and avoids pgvector's 2000-dim index cap.

> **Re-ID appearance encoder — VeRi-trained weights via ONNX (chosen approach).** A generic model (DINOv2) is weak at *vehicle identity* (~20% cross-camera rank-1); the accuracy engine for tracing is **FastReID's VeRi checkpoint** (SBS **ResNet50-IBN**, ~97% rank-1 / 82% mAP *on VeRi*, **2048-dim** — matches `reid_appearance`). The value is the *trained weights*, not FastReID's framework (which targets 2020-era `torch 1.6`/CUDA 10 and can't drive the 4050). **Do NOT install `JDAI-CV/fast-reid` into the pipeline.**
>
> **Chosen path — ONNX export, run on CPU (Option A):** load the FastReID VeRi checkpoint **once** in a throwaway old-torch environment (Docker or Colab), export to the framework-neutral **ONNX** format, then run the `.onnx` in the pipeline with **plain `onnxruntime` (CPU)** — *not* `onnxruntime-gpu`. **Why CPU:** re-ID embedding is an **offline batch pass over just the K crops per tracklet** (not per-frame), so CPU speed is fine, and the plain `onnxruntime` wheel has **no CUDA deps** — so it **completely sidesteps** the NixOS CUDA/cuDNN discovery pain that `onnxruntime-gpu` would hit on this box. Same weights → **identical embeddings → identical accuracy**; the only cost is a slightly slower *offline* ingest (acceptable). The export step is CPU-only and one-time too. Ship the single `.onnx` file. (GPU-first is about the heavy per-frame work — **YOLO + SigLIP stay on GPU**; the small re-ID pass on CPU is a fine trade.)
> - *Backup (Option B), if the export misbehaves:* rebuild **ResNet50-IBN** in modern torch and load the checkpoint's state-dict (remap layer names, tap the BNNeck 2048-d feature). Pure torch, no ONNX, but fiddlier.
> - *Fallback (Option E), only if we run out of time:* DINOv2 + color + the GPS trajectory cue — always works, but leans hard on geometry to carry the weak appearance baseline.
>
> **Two things that matter for the number:** (1) **match FastReID's preprocessing** (vehicle models use ~256×256 input + its specific normalization) — get it wrong and embeddings degrade silently; (2) **measure on CityFlow crops, don't assume** — 97%-on-VeRi ≠ 97%-on-CityFlow (viewpoint/resolution domain gap); expect a large lift over the ~20% generic baseline, but let `evaluate_reid.py` tell you the real figure. DINOv2 stays only as an offline ablation — and being 384-dim it won't fit `reid_appearance vector(2048)`, so run ablations into a separate table/untyped array, not the production column. The chosen encoder fixes `reid_appearance` at **2048-dim** once — keep it.

**[7] STORE.** Build the row, drop tracks with `num_detections < 2` (noise), upload crops to disk/object store, `INSERT ... ON CONFLICT (tracklet_id) DO UPDATE`.

**Person attributes — outfit color (SigLIP) + structured attrs (VLM).** People are near-identical under coarse description ("man, blue shirt, jeans"), so the SigLIP vector *alone* over-returns on person queries. The fix is **not more attributes — it's putting each attribute in the right role:**
- **Hard filters (reliable → safe to gate on):** `scene`, `camera_id`, the **time window** (`ts_*_s`), `entity_type`, `subtype`. These cut the vast majority of false positives at ~zero risk and are already indexed (`tracklets_filter`); almost every real query is scoped in space + time, which is the actual lever for "too many results", not richer attributes.
- **Soft ranking signals (brittle → NEVER a hard filter; always abstain when unsure):** everything person-specific. ANDing brittle attributes multiplies their error into recall (three 85%-accurate filters ≈ 61% chance of dropping the true match). SigLIP already does the soft-OR fusion, so attributes only **re-rank its top-K, confidence-weighted** — an absent or abstained attribute contributes 0, never a penalty.
- **What we store** (`person_attrs` JSONB, additive/nullable — no migration): apparent_gender/age/backpack/headwear **plus upper- and lower-body color**, all from one **VLM pass** (Qwen3-VL-4B via llama.cpp, see [6e]; colors vocab-constrained in the prompt so they stay canonical/filterable). **A region-split SigLIP color stage ([6d]) preceded the VLM colors and was removed (Jul 2026):** it cut fixed torso/legs crop fractions and mean-pooled them across the K crops, so a bystander in even 1 of 3 crops blended into the label, and with its commit margin at 0 it *never abstained* — in the merge ("SigLIP wins, VLM fills gaps") its mislabels therefore outranked better VLM answers (confirmed by the A/B color dump). The VLM avoids the failure structurally: it sees all K crops, is told they show one tracked subject, and may answer "unknown". **Why a VLM and not a supervised PAR classifier:** we tried one (PromptPAR, PETA-35 and PA100K checkpoints) and both failed on Surat with **high-confidence false positives** — the root cause is structural: a closed-set classifier under domain shift is *forced to answer* and is miscalibrated, so its wrong answers come at 0.95+. The VLM fixes that structurally — **"unknown" is an allowed answer**, so unreadable crops abstain instead of mislabeling (validated on SUR01/c004: correct female+headscarf and elderly reads, backpack from strap detail, all-"unknown" on a 41×77 blob; ~1.4 s/tracklet). **Multi-image per tracklet:** all K crops go in one request with a prompt stating they show the same tracked person — this rejects bystanders (a single-crop run absorbed a helmeted scooter rider behind the subject into "helmet") and combines views (backpack only visible from behind). **Schema pruned to what the footage supports:** footwear and fine age bands dropped (unreadable at CCTV resolution); age is child/adult/elderly; headwear vocab is India-appropriate (cap/helmet/turban/scarf). Attribute quality is **unscored until a hand-labeled eval set (~100 tracklets) exists** — until then everything VLM-derived stays a soft ranking signal, gender doubly so (*presentation, never identity*). Fine-tuning is the last resort, via VLM pseudo-labels, only if throughput ever demands a dedicated model. **Faces are out of scope** (low-res, privacy, not needed).
- **Region-split color, why:** one dominant color blends shirt + trousers into mush. Naming an **upper** region (torso, y≈.15–.50) and a **lower** region (legs, y≈.50–.85) independently against clothing-noun prompts (`"a red shirt"`, `"blue trousers"`) yields two independent signals — the single biggest discriminator for people — and is nearly free (reuses the SigLIP image + text towers). Each region gets `{name, conf}`; low top1-top2 margin abstains.
- **Recall > precision for CCTV investigation:** better the target sits at rank 12 in a slightly noisy list than a clean list that filtered them out via a misclassified attribute. So Tier-B stays strictly soft; any *hard* attribute filter is opt-in (an explicit UI checkbox) and only ever on the Tier-A-reliable fields.

**Geometry / space-time gate (only when calibration + sync exist).** To place a tracklet in the real world: take the box's **bottom-center** (the wheels-on-road point the homography is accurate about), and apply the **inverse** homography `H⁻¹`. CityFlow's `calibration.txt` stores the matrix **ground→image**, so applying it directly to pixels gives garbage; `H⁻¹` maps image→ground (GPS lat/lon). Convert lat/lon to **local meters** before any distance test (degrees are anisotropic — at ~42.5°N, 1° lat ≈ 111 km, 1° lon ≈ 82 km). Camera **c005** also carries camera intrinsics + lens-distortion coefficients — undistort its points before `H⁻¹`, and tolerate the extra lines when parsing. Keep gates **generous** (tens of meters; homography degrades near the horizon and reprojection errors run ~3–11 px). Payoff: once inverted, all of a scene's cameras land in **one shared GPS frame automatically** — no manual cross-camera alignment. Store the result in `ground_x`/`ground_y` (meters). **Using it:** this scene's cameras *overlap*, so treat space-time as a positive **trajectory-consistency** cue, not an exit→entrance veto — at overlapping timestamps require the two projected positions to agree (~15–30 m); at non-overlapping times require a plausible implied speed (≤ ~40 m/s). No hand-built topology table needed.

**Fragment merge (per camera, before cross-camera matching).** One car often becomes several tracklets in a single camera (occlusion, re-entry, a momentary ID drop) even at full fps. Within each camera, merge tracklet pairs that are **non-overlapping in time**, **temporally close**, and **high re-ID similarity** into one tracklet (stitch their crops + time-span). This is the mirror image of the cannot-link rule: same-camera *overlapping* = definitely different objects; same-camera *non-overlapping + very similar* = the same car returning. Run it before cross-camera grouping so fragments don't inflate the graph (extra nodes for bad links to chain) or duplicate search results.

**Re-ID matcher.** Offline, per scene (the gallery is small — numpy/FAISS on the fused appearance+color vectors). Steps: (1) cosine-score tracklet pairs across *different* cameras; (2) keep pairs that are **mutual-nearest** and above a similarity threshold; (3) drop any that violate a **cannot-link** — same-camera temporal overlap always, physically-impossible-position *when* calibration+sync exist; (4) optionally enforce **one-to-one per camera pair** via Hungarian assignment; (5) **constrained agglomerative clustering** (merge best-first, reject constraint-violating merges) → assign `global_id`. Tune the threshold + fusion weight `w` against `gt.txt` with the evaluator. **Do not use plain connected components** — one wrong edge over-merges the scene.

**Scale note.** Processing is per-camera, sequential on the single GPU. Detection+tracking runs at **full fps** (~2000 frames for most cameras, but up to ~4300 for S05 — a few minutes of YOLO11m at imgsz=1280 on the 4050 each); **embedding runs only on the K best crops per tracklet**, not every frame, so it stays cheap regardless. All 65 feeds is comfortably an overnight-or-less job; do S01 (5 cams) first for the demo, widen as time allows. Embeddings batch at ~128 crops.

---

## API contract (sketch — pin the exact JSON before Phase 2)

- `GET /search?q=<text>&type=<vehicle|person>&scene=<S01>&t0=<s>&t1=<s>&limit=20` → `{ results: [ { tracklet_id, scene, camera_id, subtype, color, ts_start_s, ts_end_s, score, crop_url, global_id|null } ], fell_back: <bool> }`. Results are **dedup'd** (one object once); `color` is a **soft** signal, never a hard filter; `fell_back:true` when a starving filter was dropped (never an empty grid).
- `GET /trace/{scene}/{global_id}` → `{ scene, global_id, hops: [ { camera_id, ts_start_s, ts_end_s, ground_x, ground_y, crop_url, video_url } ] }` ordered by time. **`global_id` is unique only within a scene** (the matcher runs per scene), so trace must be scene-scoped. A tracklet with **no `global_id` yet** returns as a **single-hop** trace — don't 404.
- `GET /media/{scene}/{camera}` → the camera's web-playable MP4 **with HTTP range support**; the UI seeks to `ts_start`/`ts_end`. **Scene-scoped** because camera IDs repeat across scenes (e.g. `c010` is a *different* video in S03 vs S05).
- **Auth (hackathon posture):** one shared secret or a **read-only DB role** for the API; no user accounts. State it so nobody builds NextAuth mid-crunch.

## Build phases (each is demo-able)

- **Phase 0 — Infra & GPU.** Fresh `apps/ingest` (**wipe the old spike's untracked files, re-scaffold** Python/uv) with `gpu_setup.py`; **enable + start Docker** — on NixOS this needs `virtualisation.docker.enable = true` + a `nixos-rebuild` (not just `systemctl start`; daemon is currently inactive), or use **Podman** as a drop-in — then `docker compose` Postgres+pgvector; run the schema above; a `verify_gpu.py` that asserts CUDA and prints the device. **SigLIP variant + vector dim LOCKED:** `google/siglip2-so400m-patch14-224` → `semantic_vector vector(1152)` (schema depends on it). **Extract `cam_timestamp/` from `cityflow.zip`** (and `eval/eval.py` only if you want it as a scoring cross-check). *Done:* CUDA True, empty `tracklets` table exists, SigLIP dim frozen, timestamps on disk.
- **Phase 1 — Ingest vehicles, scene S01.** Implement stages 1–7 → rows with real `semantic_vector`, `reid_appearance`+`reid_color`, `color`, crops. **Also: per-camera media prep** — probe each `vdo.avi` and produce a **web-playable MP4** (CityFlow's `msmpeg4v2` AVI needs a full H.264 transcode; store the `video_ref`), so Phase 2's player has a file to seek in. **And ship `evaluate_reid.py`** (IoU tracklet→GT match + scene-filtered GT + IDF1 via `motmetrics`/TrackEval + rank-1/5) — nothing downstream can be tuned without it. *Done:* S01 produces populated tracklet rows **and a web-playable MP4 per camera**; the scorer's **rank-1/5 runs on real output and its IDF1 passes an oracle (GT-grouping) smoke-test** (true IDF1 waits for Phase 3's grouping).
- **Phase 2 — Hybrid search API + UI (headline demo).** FastAPI `GET /search` (text → SigLIP text vector + SQL filters → pgvector ranked results) and `GET /media/{scene}/{camera}` (serves the camera's web-playable video with HTTP range requests; UI seeks to `ts_start`/`ts_end`, crops are the grid thumbnails); Next.js search bar + filter dropdowns + results grid (**dedup so one object appears once**). *Done:* "white truck" returns ranked, de-duplicated results whose playback seeks to the object.
- **Phase 3 — Cross-camera tracing + accuracy number.** Re-ID matcher assigns **scene-namespaced** `global_id` on S01 (guardrailed grouping; space-time gate where geometry allows); `GET /trace/{scene}/{global_id}`; UI detail view drawing the path on `cam_loc/S01.png` with a timeline (**needs a small hand-annotated `{camera: (px,py)}` table** — the map is drawn dot-to-dot between camera pixels, *not* homography-projected; ~30 min per scene). **Score IDF1 (headline) with `evaluate_reid.py`**, rank-1/5 as diagnostic. *Done:* clicking a car shows its cross-camera path and we can quote an IDF1 number.
- **Phase 4 — Scale & tune.** More scenes; accuracy levers: bigger YOLO, ROI masking, tuned fusion `w` + match threshold, and **fine-tuning or upgrading the re-ID encoder** (the VeRi model is already the Phase-1 default; here you push it further). Target: maximize IDF1.
- **Phase 5 — People & plates (additive).** A **person-attribute** pass (VLM attributes + outfit colors → `person_attrs`) and **plate OCR** (`plate_text`) as new passes over existing crops — fill null columns, no re-detection, no re-embedding. Free-text "man in red shirt" already works because SigLIP handles people out of the box.

---

## Suggested team split
- **Ingest/CV** — Phases 0–1 pipeline (detect/track/crop/attributes/embed), GPU setup, the **one-time ONNX export** of the re-ID model, and the per-camera media transcode.
- **Data/Backend** — schema, FastAPI search + trace endpoints, the re-ID matcher (Phase 3), and **`evaluate_reid.py`** (lands in Phase 1).
- **Frontend** — search UI, results grid, cross-camera map + timeline.

## Verification
- **GPU:** each stage asserts + logs `cuda:0`; CPU fallback fails loudly.
- **Ingest:** row counts per camera; eyeball sample crops and one annotated clip.
- **Search:** qualitative queries ("white truck", "dark sedan"); confirm free-text ranking works and that a starving filter **fails open** (falls back to unfiltered ranking, never an empty grid). **Color-naming sanity check:** hand-label ~50 crops and measure HSV color accuracy (especially achromatic white/gray/silver) so you *know* how brittle it is before relying on it as even a soft signal.
- **Re-ID (the money metric) — needs a real scorer, not just an intention.** "Score it against `gt.txt`" hides two traps and a metric mix-up; here is the whole thing so the team can review it:

  **Trap 1 — our IDs ≠ the truth's IDs (a translation step is missing).** Our pipeline invents its own tracklet IDs (`S01_c001_t7`); `gt.txt` calls that same car `vehicle 34`. Nothing connects them. Before scoring anything we must **translate**: for each predicted box, find the ground-truth box it overlaps in that frame (**IoU ≥ 0.5**, where IoU = overlap area ÷ union area, 0–1), majority-vote over the tracklet's frames → our tracklet **inherits a GT id**. We now detect/track at **full fps** (High #3), so match against GT on **every frame** (no stride).

  **Trap 2 — GT only labels multi-camera vehicles, and S06 has none.** `gt.txt` annotates *only* vehicles that pass through **≥2 cameras**, so our many single-camera / parked / distant detections have **no truth entry** — they must be **excluded** from scoring, not counted as errors (or the number looks terrible for no reason). **S06 has no GT at all → score S01–S05 only.**

  **Two metrics, two different jobs (this is the mix-up to avoid):**
  - **IDF1 = the headline number.** Our Phase 3 output is a *grouping* (which tracklets share a `global_id`), so we need a metric that scores the grouping as a whole. IDF1 rewards **keeping one real car as one identity** (no fragmentation) AND **not fusing different cars** (no over-merge) — so it's the only metric that can *catch the connected-components blob failure* from the matcher. **Compute IDF1 with a maintained library** (`motmetrics` or TrackEval) in our own harness — do **not** fight CityFlow's 2019 `eval/eval.py` (it **raises** on missing ROI directories, and its pinned C-extension deps `lapsolver`/`pytrec_eval` won't build on Python 3.11/NixOS). Keep `eval.py` only as an optional cross-check.
  - **rank-1 / rank-5 = tuning diagnostic ONLY.** These are *retrieval* metrics ("is the correct match near the top of a ranked list?"). Great for tuning the embedding, fusion weight `w`, and the match threshold — but they can read a healthy ~60% while the grouping is a broken blob, so **never quote rank-1 as the headline.**

  **Deliverable — `evaluate_reid.py` (our own harness; reuse the metric math, don't reinvent it).** The IDF1 *math* comes from a library (`motmetrics`/TrackEval); the *glue* is ours to write: **(1)** IoU-match our tracklets → GT ids (full fps, IoU ≥ 0.5) and **exclude unlabeled** ones; **(2)** build the scene's GT from the **per-camera `gt/gt.txt`** files (comma-delimited `[frame,id,x,y,w,h]`, camera from the folder name) — these are *already* scene+camera-scoped, so you sidestep the bundled `ground_truth_train.txt` (space-delimited, leading camera column, and it mixes S01+S03+S04 — scoring S01 against all of it craters IDF1). Pin one format/delimiter in the harness; **(3)** compute **IDF1/IDP/IDR** via the library, plus rank-1/5 for tuning; **(4)** an **oracle smoke-test** — feed it GT-inherited groupings and confirm it reports ~100%, proving the harness is wired right before you trust real numbers. Note IDF1 only means something **after** Phase 3's cross-camera grouping — on Phase 1's per-camera-only output it scores ~0 by construction (single-camera IDs), which is expected. Land it in **Phase 1** anyway: tuning `w`/threshold is "try → measure → repeat," and rank-1/5 already works there while the oracle proves the IDF1 path.
- **Team access:** point `DATABASE_URL` at a hosted Postgres; confirm a teammate queries the same tracklets **and that crops/videos resolve for them** (media is shared too, not just rows — see "Media storage").

## Target domain: Indian roads vs CityFlow

**CityFlow is our lab; Indian roads are production.** We develop and *measure* on CityFlow because it hands us three luxuries — **calibration** (homography), **synchronized time**, and **ground-truth labels**. The real deployment target (Indian traffic CCTV) will likely have **none** of them: cameras added ad hoc with no calibration, NVR clocks that drift, and no labelled answers to score against. So the guiding rule is: **never let a feature that depends on CityFlow's luxuries be load-bearing.**

Concretely, that shapes the design:
- **Matching core is infrastructure-free.** Appearance+color similarity, mutual-nearest-neighbor, one-to-one matching, and the same-camera-temporal-overlap cannot-link need *no* calibration or synced clocks — they work on any footage. The **space-time gate is an optional booster**, switched on for CityFlow (where it also enables the accuracy proof) and off where geometry/sync are missing. The physics — a vehicle traces a consistent path through space and time — is universal; only the *ability to localize cameras* is dataset-specific.
- **Media handling is format-agnostic.** We don't assume the source container/codec. A probe-and-normalize step (see "Per-camera media prep") converts to web-playable H.264/MP4 only when needed — CityFlow's AVI needs a full transcode; production MP4s likely need nothing (or a one-second remux). Never hardcode "transcode AVI."
- **Evaluation degrades gracefully.** On CityFlow we quote hard numbers vs `gt.txt`. On uncalibrated production footage there's no GT, so validation becomes qualitative/spot-check — plan for that, don't assume a metric.

**Domain-shift issues to expect on Indian roads (flagged now, not solved yet):**
1. **Vehicle mix.** CityFlow is cars/trucks/buses; Indian roads are dominated by **two-wheelers, auto-rickshaws, cycle-rickshaws, tractors**. YOLO's COCO classes have no "auto-rickshaw" (it'll mislabel as car/truck/motorcycle) — detection needs India-specific weights. **(→ now addressed — see *Current India-profile status* below.)**
2. **Re-ID domain gap.** A VeRi-trained encoder is strong on cars but weaker on autos and two-wheelers — the exact vehicles that dominate here. Future step: fine-tune the re-ID encoder on Indian vehicle imagery. **(→ still open — see status below.)**
3. **Conditions.** Denser occlusion, night/low-light, monsoon glare, more varied camera angles/quality — all degrade detection, color, and re-ID beyond what CityFlow's clean daytime footage implies.
4. **Plates.** Indian plate formats/scripts need India-specific OCR (relevant to the later plate branch).

**Current India-profile status (branch `pipeline-for-indian-roads`).** The pipeline is now profile-driven (`apps/ingest/pipeline/profiles.py`): one codebase, a `--profile {cityflow|india}` switch that flips **only** the detector, class map, re-ID encoder, and footage dir — everything else is the shared, domain-agnostic core above. Where each India knob stands:
- **Vehicle detector — DONE.** Official **AIM@IISc UVH-26-MV YOLOv11-X** (arXiv 2511.02563; Apache-2.0 repo, AGPL-3.0 underlying YOLO), wired and verified to load via Ultralytics. 14 India-specific classes (hatchback/sedan/SUV/MUV/bus/truck/three-wheeler/two-wheeler/LCV/mini-bus/tempo-traveller/bicycle/van/others); class-id order confirmed against the checkpoint's `model.names`. imgsz 640 (its training resolution). Resolves domain-shift #1 — auto-rickshaws and two-wheelers now have real classes instead of COCO mislabels. *(A third-party YOLO26x fine-tune, `Perception365/VehicleNet-Y26x`, was evaluated and rejected in favour of the official model for provenance/citability.)*
- **Person detector — DONE (india runs TWO detectors).** UVH-26 is vehicle-only, so people come from a paired **CrowdHuman YOLOv8n** (`models/crowdhuman/yolov8n_crowdhuman.pt`, class 0 = person, imgsz 1280, conf 0.25). Chosen by A/B on SUR01: it **beat** stock YOLO11-X-COCO and a proprietary YOLOv8-X person model on dense-crowd recall — **training-domain match beats model size**. The two `.track()` streams are zipped frame-aligned (`t`=vehicle / `p`=person id namespaces) and merged with pedestrian-FP suppression (a two-wheeler/bicycle box a same-footprint person box explains is dropped; real riders kept). Remaining miss = the far-distance packed crowd, a resolution wall only tiling/SAHI would break. Validated on all 5 SUR01 cams; **not yet run through a full ingest** (nothing person-searchable stored yet).
- **Re-ID — STILL A GAP (the weakest link for India).** The india profile uses the **same `veri_reid.onnx`** as a placeholder — VeRi/CityFlow-trained, so domain-shift #2 is unaddressed on autos/two-wheelers. No public Indian vehicle re-ID / MTMC dataset exists (CityFlow is the only MTMC benchmark), so closing this means sourcing Indian multi-camera footage and self-supervised / ANPR-labelled fine-tuning, then re-exporting a 2048-d ONNX (dim stays locked). Not started.
- **Footage — DELIVERED; india is now the DEFAULT profile on this branch.** 5 real Surat CCTV clips are prepped as india scene `SUR01` (`footage_data/india/SUR01/c001–c005`, real cam offsets, `cam_labels`). Bare `run_ingest.py` now targets SUR01 under `--profile india`. **CityFlow is retained as the dormant regression bench** — still the only footage with ground truth / a defensible IDF1 number — run it explicitly with `--profile cityflow --scene S01` (`evaluate_reid.py` still defaults to cityflow, so scored eval is unchanged). Caveat: SUR01 has **no person/vehicle ground truth**, so it can be eyeballed but not *scored* — quantitative regression still lives on CityFlow.
- **Color / SigLIP / matcher / DB / API / UI — unchanged**, domain-agnostic, work as-is under either profile.

None of this blocks the hackathon. With real Surat footage now in hand, **india is the build/demo target**; CityFlow is kept purely as the ground-truth regression bench. We keep the **portable core independent of any single dataset's conveniences** and stay honest about what transfers to Indian roads and what still needs work (**re-ID is the open gap** — still VeRi/CityFlow-trained).

## Open items / risks
- **Camera timestamps — resolved, now a Phase 0 task.** `cam_timestamp/` *is* in the zip (offsets S01 ≤2.2s, S02 ≤0.66s, S05/S06 0, S03 ≤8.7s, **S04 ≤176s**). Extract in Phase 0. The offset-0 fallback is fine on S01/S02/S05/S06, loose on S03, and **unsafe on S04** — never run S04 on the fallback. Frames also aren't perfectly aligned (keep gates ±2–3s slack) and `c015` (S03) is 8 fps.
- **VRAM (6 GB) — not a real risk, handled by staged passes** (see "Memory: why 6 GB is plenty"). Models run one at a time with crops persisted between stages; tune batch/`imgsz` if any pass is tight.
- **reid appearance dim is set by the chosen encoder** (e.g. 2048 for a VeRi ResNet-IBN). It's a plain stored column — cross-camera matching is an **offline batch job that needs no vector index**, so it isn't bound by pgvector's 2000-dim index cap. If you ever change the re-ID encoder, re-embed just `reid_appearance` (`semantic_vector` unaffected).
- **GPU de-scope path (contingency).** The re-ID pass already runs on CPU; the remaining GPU users are YOLO + SigLIP. If the GPU stack fails, both can run CPU (slower) — drop `imgsz` and use a smaller YOLO. Ingest just gets slower; the demo (an offline-built index + live text search) still works.
