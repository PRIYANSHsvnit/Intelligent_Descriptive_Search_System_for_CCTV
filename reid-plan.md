# reid-plan.md — Re-ID encoder upgrade (FastReID → DINOv3)

Companion to `plan.md` (design bible). This is the working plan for replacing the
appearance re-ID encoder to (a) get off CPU and (b) close the domain gap on Surat.

## Why

- Current encoder: **FastReID VeRi ResNet50-IBN**, `pipeline/embed_reid.py`, **CPU / ONNX**,
  2048-dim, **vehicle-only**. Trained on clean VeRi car crops → **domain gap** on dusty Surat
  street footage. CityFlow rank-1 = **47%** (the anchor to beat). Also: CPU-bound = slow (a
  standing pain point).
- Replacement: **DINOv3** self-supervised features. Better *instance* discrimination + domain
  transfer than a VeRi-only supervised net (DINO beats CLIP/SigLIP for instance re-ID; CLIP
  wins only *semantic/text* retrieval, which is a different job). Runs on **GPU torch** →
  fixes the CPU pain too.
- Naming: `ViT-<Capacity>/<PatchSize>`. Capacity S/B/L/H+ (21M/86M/300M/840M params →
  384/768/1024/1280 dim). `/16` = 16px patches. **Bigger input image = denser patch grid =
  finer detail = the main re-ID accuracy knob.** Use `lvd1689m` (web) weights, NOT `sat493m`
  (satellite). Ignore the ConvNeXt variants.
- Candidates (all through one encoder slot): **FastReID (baseline) → `dinov3-vitl16` @320px →
  `dinov3-vith16plus` @ same res.** ViT-H+ is the 6 GB ceiling (fp16, inference-only, small batch).
- Persons: current re-ID is vehicle-only. Person re-ID (OSNet MSMT17 as a 2nd encoder) is a
  **separate future track**, not in this plan.

## Key simplification: A/B is rebuild-free until a winner is picked

`evaluate_reid.py` scores **rank-1/5 over the fused re-ID vectors** — works today, needs no
`global_id`, and is **dimension-agnostic** (just cosine; 1024/1280/2048 all fine). So testing an
encoder = regenerate `vec/reid_appearance.npy` and re-run the harness. The **LOCKED
`reid_appearance vector(2048)` schema is irrelevant during A/B**; it only changes at the very
end (Phase 3) when the winner is stored in Postgres for production.

## Two evals

| Eval | Domain | Ground truth | Metric | Role |
|---|---|---|---|---|
| **CityFlow S01** | US highway | real labels | rank-1/5 (`evaluate_reid.py`) | objective encoder *ranking* (proxy; 47% today) |
| **Surat plate-eval** | Surat (the deliverable) | plates = pseudo-labels | within-cam AUC / rank-1 (new harness) | in-domain quality graders care about |

Absolute CityFlow number does NOT transfer to Surat; the *relative ranking* mostly does. Decide
by the **Surat** number, use CityFlow to prune cheaply first.

## Surat plate-eval design (WITHIN one camera)

Confirmed: **no plate spans two Surat cams** → all positive pairs are within a single camera
(same vehicle fragmented into multiple tracklets by occlusion / exit-reentry; these are the
14 `--plates-only` stitch groups).

- **Universe:** vehicle tracklets in one camera with validated `plate_text` (optionally
  `plate_conf ≥ thresh`).
- **Positive pair** = same camera, same plate. **Negative pair** = same camera, different plate.
- **Primary metric = verification AUC** (ROC-AUC of cosine sim, same-plate vs diff-plate).
  Chosen because N is small (~26 plates / ~95 tracklets total, fewer per cam) → AUC degrades
  gracefully where rank-1/mAP get noisy.
- **Secondary = retrieval:** each plated tracklet as query, gallery = other plated tracklets in
  that cam, rank-1/mAP (same plate = positive).
- **Confound control:** temporally adjacent fragments look near-identical ("too-easy"
  positives). Optionally require a **min time gap** (via `ts_start_s`) between paired tracklets;
  report AUC with and without the gap.
- **Always report N** (positives/negatives) — this is a pilot-sized signal that grows with more
  SUR01 ingest. Run on the camera with the richest plated-vehicle set (NOT c002, the crowd cam).

## Plan

### Phase 0 — Baselines + harness (no rebuild, no schema change)
1. Reproduce CityFlow baseline: `evaluate_reid.py --scene S01` w/ FastReID → confirm **47% rank-1**.
2. Build Surat plate-eval harness (new `evaluate_reid_plates.py`): reads plated vehicle
   tracklets + a `reid_appearance.npy`, computes within-cam AUC + rank-1 + N, per camera.
   Encoder-agnostic (takes any vector file).
3. Generate FastReID's SUR01 vectors (`.npy` only — SUR01 has ZERO re-ID vectors today) and
   record FastReID's Surat plate-eval number. Both baselines now exist.

### Phase 1 — DINOv3-L
4. Add a DINOv3 encoder path to `embed_reid` behind a config switch (model id + input
   resolution + `CLS ⊕ mean-pooled patch` pooling + **GeM** over K crops); keep FastReID/ONNX
   path intact.
5. Generate `.npy` for S01 + SUR01 with `dinov3-vitl16 @ 320px`.
6. Run both evals; compare to baselines.

### Phase 2 — DINOv3-H+
7. Flip config to `dinov3-vith16plus @ same resolution` (single-variable change). Regenerate, re-eval.
8. Compare L vs H+: does extra capacity move the **Surat** number enough to justify throughput cost?

### Phase 3 — Decide + productionize (only now touch schema)
9. Pick winner by the **Surat plate-eval** (CityFlow as tiebreak/sanity).
10. `ALTER` `reid_appearance` to winner's dim, update the LOCKED constant in `constants.py`, set
    the encoder as `embed_reid` default, ingest SUR01 re-ID into Postgres (searchable).
11. Persons (OSNet) = separate future track.

## Gotchas (build in from the start)
- **DINOv3 = GPU torch** (unlike CPU-ONNX FastReID). Needs `gpu_setup.ensure_gpu_libs()` shim,
  and per the `vlm_attrs` OOM lesson it must run in its **own `run_ingest` process** — leftover
  CUDA context from co-resident torch stages OOMs 6 GB.
- **fp16 mandatory** for H+ on 6 GB; watch batch size as resolution climbs to 320–384px.
- Appearance is a **tiebreaker** on real SUR01 (plates + cam topology + travel-time + VLM attrs
  do the heavy lifting) — don't over-invest past "clearly beats FastReID."

## Results log

### Baselines — FastReID VeRi ResNet50-IBN (the encoder to beat)
**CityFlow S01** (`evaluate_reid.py --scene S01`, oracle IDF1 = 1.0 ✓ harness sane):
- **Appearance-only (w=0): rank-1 = 47.08% / rank-5 = 63.23%** ← headline "47%", the A/B target
- Fused w=1.0: rank-1 = 40.98% / rank-5 = 59.7%; IDF1 = 0.4167
- Finding: color fusion *hurts* rank-1 (47→41) → **appearance-only is the isolation metric AND headline**.

**Surat plate-eval** (`evaluate_reid_plates.py`, within-cam, plates=pseudo-GT):
- Data reality: plate-eval is **c004-only** — 972 vehicles, 74 plated, 24 distinct plates,
  15 same-plate groups → **161 within-cam positive pairs from ~15 distinct vehicles**
  (c002/c005 = 0 plated; c001/c003 negligible). Pilot-sized → AUC is the primary metric.
- **FastReID Surat baselines (c004, 74 plated / 24 plates):**
  - easy (hard-neg=off, gap=0): **AUC 0.977** (161 pos / 2540 neg), rank-1 0.831, mAP 0.851 — but
    **near-saturated + inflated** by near-adjacent fragments (see gap sweep) and easy negatives.
  - gap sweep (hn=off): AUC 0.977→0.945(5s)→0.772(15s, only 5 pos)→cliff. Temporally-distant
    same-vehicle matching is where FastReID actually weakens (posSim 0.83→0.45), but SUR01 has
    too few such pairs (3–5) to measure reliably.
  - **CANONICAL A/B METRIC (discriminative): hard-neg=subtype_color, min-gap=5s → AUC = 0.9065**
    (48 pos / 383 neg; posSim 0.743 vs negSim 0.401). "Same white SUV's later fragment vs a
    *different* white SUV." Has real headroom AND enough pairs. This is what DINOv3 must beat.
  - FINDING: within-camera Surat re-ID (the actual per-cam deliverable) is already *decent* with
    FastReID. The robust, high-headroom discriminator remains **CityFlow cross-cam (47%)**; treat
    Surat as in-domain confirmation, not sole decider.

### Phase 1 — DINOv2-base (open; DINOv3 gated → 403, user unlocking in parallel)
Encoder: `embed_reid_dino.py` (standalone, GPU/fp16, CLS⊕mean-patch, mean-pool over crops → tagged
`.npy`). GPU confirmed under shim (RTX 4050). `facebook/dinov2-base` @322px, dim=1536.

**Surat plate-eval (c004) — DINOv2-base vs FastReID:**
| setting | FastReID AUC | DINOv2b AUC | DINOv2b negSim |
|---|---|---|---|
| easy (hn=off, gap=0) | 0.977 | 0.937 | 0.655 |
| canonical (subtype_color, gap5) | **0.9065** | **0.7825** | 0.777 |
| mAP | 0.851 | 0.795 | — |

**DINOv2-base is WORSE on Surat re-ID.** Tell: DINOv2 negSim 0.65–0.78 (vs FastReID 0.31–0.40) —
generic foundation features make all vehicle crops look alike → poor instance separation. Matches
lit nuance: zero-shot DINO beats *CLIP* for instances, but a *supervised metric-learning re-ID*
model (FastReID) beats zero-shot DINO on the re-ID task itself. **CityFlow S01 cross-cam (appearance-only rank-1) — THE DECISIVE TEST:**
| encoder | rank-1 | rank-5 |
|---|---|---|
| FastReID-VeRi | **47.08%** | 63.23% |
| DINOv2-base @322 cls_patch | **10.85%** | 27.82% |

**DINOv2-base COLLAPSES cross-camera** (11% vs 47%) despite MATCHING FastReID within-camera on
Surat (rank-1 0.83 both). Textbook zero-shot re-ID signature: foundation features encode holistic
appearance (bg/pose/lighting), NOT camera-invariant identity — which is exactly what FastReID's
supervised metric-learning supplies. Verified not-a-bug (within-cam parity proves the encoder
works). Ablation done: **cls-only = 12.6% cross-cam** (vs cls_patch 10.9%) — pooling is not the
culprit; the ~4× gap is fundamental to zero-shot features.

### Phase 4 — DINOv3 (user unlocked gated weights, `hf download` into HF cache)
DINOv3-L/16 (1024-d, 4 reg tokens, patch16) + H+/16 (1280-d) fetched. GOTCHA: **DINOv3 overflows in
fp16 → all-NaN vectors** (DINOv2-base survived fp16; DINOv3 does not). Fix: `--dtype bf16` (4050 ok).
`embed_reid_dino.py` reads reg-token count from config (skips 4), ImageNet norm matches. @320px cls_patch.

**DINOv3-L A/B (bf16) — the standout:**
| bench | FastReID | VeRi-Wild | DINOv2b | **DINOv3-L** |
|---|---|---|---|---|
| CityFlow cross-cam rank-1 | 47.1% | 38.0% | 10.9% | **46.7%** (rank-5 **63.8** > 63.2) |
| Surat canonical AUC | 0.907 | **0.975** | 0.783 | 0.969 |
| Surat canonical mAP | 0.851 | 0.900 | — | **0.909** |
| Surat canonical rank-1 | 0.831 | 0.846 | — | **0.877** |

**DINOv3-L is the best ALL-AROUND encoder**: ~TIES supervised FastReID on CityFlow cross-cam
(46.7 vs 47.1, rank-5 better) — a *zero-shot* model matching a VeRi-supervised one — AND best Surat
mAP/rank-1 (2nd AUC, just behind VeRi-Wild). No cross-cam collapse (unlike VeRi-Wild/DINOv2). The
DINOv2-base 11% was model-generation, not a foundation-model-category verdict.

**FINAL COMPARISON (all appearance-only; Surat = canonical hn=subtype_color gap5):**
| metric | FastReID | VeRi-Wild | DINOv2b | DINOv3-L | DINOv3-H+ |
|---|---|---|---|---|---|
| CityFlow cross-cam rank-1 | 47.1 | 38.0 | 10.9 | 46.7 | **49.9** |
| CityFlow rank-5 | 63.2 | 51.4 | 27.8 | 63.8 | **72.1** |
| Surat AUC | 0.907 | **0.975** | 0.783 | 0.969 | 0.959 |
| Surat mAP | 0.851 | 0.900 | — | **0.909** | 0.895 |
| Surat rank-1 | 0.831 | 0.846 | — | **0.877** | **0.877** |
| output dim (cls_patch) | 2048 | 2048 | 1536 | **2048** | 2560 |

**WINNER = DINOv3.** H+ is the cross-cam champion (49.9/72.1 — *beats supervised FastReID*); L is
the best Surat all-arounder (best mAP+rank-1). L vs H+ on Surat is within pilot noise. **Recommend
DINOv3-L**: best Surat all-around, ~ties/edges FastReID on CityFlow, lighter/faster (300M vs 840M),
and its **2048-d output is a DROP-IN match to the LOCKED reid dim → NO schema change**. Pick H+ only
if cross-camera tracing becomes a priority (then ALTER to 2560). Both are zero-shot (no training),
run on GPU bf16 (fixes the CPU-reid pain too).

### Productionization notes (when adopting DINOv3-L)
- reid stage moves CPU-ONNX → GPU-torch (transformers). Run in its OWN process (vlm_attrs OOM lesson).
- 2048-d = no `constants.py`/`schema.sql` change. Fold `embed_reid_dino.py` (@320 cls_patch bf16, L2
  mean-pool over K crops) into the `reid` stage; keep FastReID path as fallback.
- Persons: DINOv3 is entity-agnostic → could also give person re-ID for free (no OSNet needed). Untested.

### VERDICT (Phase 1, DINOv2-base only): zero-shot DINOv2-base is NOT a drop-in win (superseded by DINOv3-L above).
Strategy pivot needed. Options, best-effort-per-payoff:
1. **FastReID VeRi-Wild / VehicleNet weights** — drop-in (same 2048, no training), directly targets
   FastReID's domain gap. The most promising low-effort lever now. ← recommend next.
2. **DINOv3-L** (when ungated) — still zero-shot; test as the ceiling of the no-train path, temper
   expectations (base→large won't close 36 pts cross-cam alone).
3. **Fine-tuning** (foundation OR FastReID) on VeRi/CityFlow or plate-pseudo-labeled Surat — the
   path the user wanted to avoid, but the data says it's likely required for a real gain.
4. **Accept FastReID** — already the best; within-cam Surat (the per-cam deliverable) is decent (0.91).

### Phase 2 — FastReID VeRi-Wild weights (user-chosen direction)
Drop-in weight swap (same R50-IBN, 2048-d, no training). Checkpoint: FastReID zoo
`veriwild_bot_R50-ibn.pth` (config `configs/VERIWild/bagtricks_R50-ibn.yml`, SIZE_TEST 256×256,
IBN+GeM+BNNeck — compatible). VERI-Wild = ~40k IDs / ~400k imgs, far broader than VeRi → targets
FastReID's domain gap directly.
- Toolchain generalized: `reid/export_reid_onnx.py` now env-driven (REID_CFG/WEIGHTS/OUT) +
  auto-detects NUM_CLASSES from the checkpoint; `reid/export_in_docker.sh` passes those through.
  New `embed_reid_onnx.py` = non-destructive FastReID-ONNX embedder → tagged `.npy` (no clobber).
- Download gotcha: this env truncates large HTTP GETs at random points (curl got 105–435 MB of
  the 1 GB file); fixed with an `aria2c -c` resume loop (`scratchpad/dl_veriwild.sh`).
- Export DONE → `models/veriwild_reid.onnx` (auto-detected 30671 IDs, 2048-d). GOTCHA: bagtricks
  uses LAST_STRIDE=1 → onnxruntime-CPU runs it at ~3.5 s/crop (HOURS; 37-min hang). No ORT-GPU
  provider here, onnx2torch chokes on GeM's dynamic Clip. FIX: run fast-reid directly under the
  pipeline's torch 2.13 on GPU (`embed_reid_fastreid_gpu.py`, ~4 ms/crop) — feeds same raw-0-255-RGB
  as the ONNX so results are equivalent. fast-reid imports fine on torch 2.13 (needs runtime deps
  yacs/termcolor/tabulate/scikit-learn/Pillow via `uv run --with`).

**A/B RESULT — VeRi-Wild vs FastReID-VeRi (SPLIT):**
| bench | FastReID-VeRi | VeRi-Wild |
|---|---|---|
| CityFlow cross-cam rank-1 | **47.1%** | 38.0% |
| Surat c004 canonical AUC | 0.907 | **0.975** |
| Surat canonical mAP | 0.851 | **0.900** |
| Surat canonical negSim (↓ better) | 0.401 | **0.258** |
| Surat easy AUC | 0.977 | **0.992** |

**VeRi-Wild WINS on Surat (the deliverable, within-cam) but LOSES on CityFlow cross-cam.** Likely
causes: (a) CONFOUND — VeRi ckpt is *sbs* (stronger baseline: +Non-Local, more tricks), VeRi-Wild
ckpt is *bagtricks* (weaker arch); some CityFlow gap is arch, not data. (b) VeRi-Wild's broader/more
varied training (40k IDs) transfers better to Indian within-cam vehicles; VeRi-sbs+NL fits CityFlow's
US cross-cam better. Since SUR01 is per-camera (no cross-cam), VeRi-Wild's cross-cam weakness doesn't
hurt the current deliverable. CAVEAT: Surat metric is pilot-sized (15 vehicles) but AUC+mAP both up.
NEXT OPTION to win BOTH: try VeRiWild/**sbs**_R50-ibn (if in zoo) or VehicleID — SBS arch + broad data.

### Env fix (Phase 0)
- CityFlow footage had moved to `footage_data/Cityflow/train/` but the dormant `cityflow`
  profile still points at `footage_data/train`. Fixed with symlink
  `footage_data/train -> Cityflow/train` (untracked, matches SUR01 symlink pattern).

## Parked (separate, decided but deferred)
- **SigLIP → SigLIP 2** (`siglip2-so400m-patch16` FixRes, same 1152 dim): touches search/color/
  backend, NOT re-ID. Independent of this plan. See `[[reid-siglip2-upgrade-plan]]` memory.
