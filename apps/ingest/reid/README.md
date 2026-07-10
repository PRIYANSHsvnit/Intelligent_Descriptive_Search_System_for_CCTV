# Re-ID appearance encoder (VeRi FastReID ResNet50-IBN → ONNX, CPU)

`reid_appearance` (2048-dim, LOCKED) is produced by a **VeRi-trained FastReID
ResNet50-IBN (SBS)** run on **CPU via plain `onnxruntime`** (no CUDA deps → sidesteps
the NixOS CUDA discovery pain). We ship a single `models/veri_reid.onnx`; the pipeline
pass is `pipeline/embed_reid.py`, which SKIPS (leaves the column NULL) until that file
exists — so the rest of ingest runs without it.

## Why this is a separate, one-time step

The value is FastReID's **trained VeRi weights**, not its framework (it targets
2020-era torch 1.6 / CUDA 10 and can't drive the 4050). So we export the checkpoint to
framework-neutral ONNX **once**, in a throwaway old-torch environment, and never install
`JDAI-CV/fast-reid` into the pipeline.

## Acquire the checkpoint (it's a GitHub release, NOT Google Drive)

```bash
curl -L -o reid/veri_sbs_R50-ibn.pth \
  https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth
```
~190 MB, 97.0% rank-1 / 81.9% mAP on VeRi. Config: `configs/VeRi/sbs_R50-ibn.yml`.

## Export to ONNX (one command)

```bash
bash reid/export_in_docker.sh        # throwaway python:3.9-slim + torch 1.13 CPU
mv reid/veri_reid.onnx models/veri_reid.onnx
```
`export_in_docker.sh` clones fast-reid inside the container, and `export_reid_onnx.py`
builds the exact architecture (ResNet50-IBN + Non-Local + GeM + BNNeck), loads the
checkpoint, and does a plain `torch.onnx.export` (skipping fast-reid's fragile
onnx-simplify/optimize). Output: raw-RGB input `(B,3,256,256)` → 2048-d feature.

## Preprocessing — CONFIRMED

Input **256×256**, RGB. The checkpoint embeds `pixel_mean=[123.675,116.28,103.53]`,
`pixel_std=[58.395,57.12,57.375]` (0-255 scale), and the export **bakes normalization
into the ONNX graph** — so `pipeline/embed_reid.py` feeds RAW 0-255 RGB (no external
normalization). Verified correct by the rank-1 lift (garbage preprocessing would not
score ~47%).

## Backfill + evaluate

```bash
uv run python run_ingest.py --scene S01 --stages reid store   # fills reid_appearance
uv run python evaluate_reid.py --scene S01                     # real rank-1/5
```
Measured on S01: **rank-1 47.1% / rank-5 63.2%** appearance-only (`--fusion-w 0`), vs
14.8% / 30.0% color-only. Adding HSV color (`w>0`) *hurts* here because the VeRi
encoder already captures color and our color signature is noisy — so the matcher's
CityFlow sweet spot is `w≈0` (Phase 3/4 tuning).

## Fallbacks (plan.md)

- **Option B:** rebuild ResNet50-IBN in modern torch, load the state-dict (no ONNX).
- **Option E (time-boxed):** DINOv2 ViT-S (384-dim) + color — weak (~20% rank-1) and
  384≠2048, so it must go in a *separate* ablation table, never `reid_appearance`.
