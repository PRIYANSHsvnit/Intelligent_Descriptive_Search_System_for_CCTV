"""embed_reid_dino.py — DINO (v2 now / v3 when ungated) appearance vectors for the re-ID A/B.

Standalone by design: leaves the LOCKED pipeline (`pipeline/embed_reid.py`, FastReID/CPU/ONNX)
untouched so the A/B is non-destructive. Writes a TAGGED vector file
(`vec/reid_appearance_<tag>.npy`, aligned to tracklets.json order) so it never clobbers the
FastReID baseline — point `evaluate_reid_plates.py --vecfile` / `evaluate_reid.py --appfile`
at it to score.

Encoder-agnostic over the HF ViT interface (DINOv2 open; DINOv3 gated — same code path, just
`--model facebook/dinov3-vitl16-pretrain-lvd1689m` once access is granted; register tokens are
read from config so v3's are skipped automatically).

Per-crop fingerprint = CLS ⊕ mean(patch tokens) [--pool cls_patch] or CLS [--pool cls], L2-normed.
Per-tracklet = mean over its crops' fingerprints, L2-normed (same pooling as FastReID → fair A/B).

MUST run in its own process (GPU torch; co-resident CUDA ctx from other stages OOMs the 4050).

Usage:
  uv run python embed_reid_dino.py --scene SUR01 --cams c004
  uv run python embed_reid_dino.py --scene S01 --model facebook/dinov2-large --res 322
"""
from __future__ import annotations

from gpu_setup import ensure_gpu_libs
ensure_gpu_libs()

import argparse
import json

import cv2
import numpy as np
import torch
from transformers import AutoModel

import augment
from pipeline import paths

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _tag(model: str, res: int, pool: str) -> str:
    short = model.split("/")[-1].replace("-pretrain-lvd1689m", "").replace("facebook_", "")
    short = short.replace("dinov2-", "dinov2").replace("dinov3-", "dinov3").replace("-", "")
    return f"{short}_{res}_{pool}"


def _preprocess(crop_bgr: np.ndarray, res: int) -> np.ndarray:
    img = cv2.resize(crop_bgr, (res, res), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img.transpose(2, 0, 1)  # CHW


def run(scene: str, cam: str, model, patch_start: int, res: int, pool: str,
        batch: int, dev: str, dtype, transform=None, aug_tag: str = "") -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    idx: list[int] = []
    imgs: list[np.ndarray] = []
    for i, t in enumerate(tracklets):
        for k in t["crop_refs"]:
            im = cv2.imread(str(paths.OUTPUT_ROOT / k))
            if im is not None:
                if transform is not None:        # synthetic "camera B" degradation (augment.py)
                    im = transform(im)
                idx.append(i)
                imgs.append(_preprocess(im, res))

    dim = None
    sums = None
    counts = np.zeros(n, dtype=np.int32)
    with torch.inference_mode():
        for b in range(0, len(imgs), batch):
            arr = np.stack(imgs[b:b + batch])
            px = torch.from_numpy(arr).to(dev, dtype=dtype)
            hs = model(pixel_values=px, interpolate_pos_encoding=True).last_hidden_state
            cls = hs[:, 0]
            if pool == "cls_patch":
                patch = hs[:, patch_start:].mean(dim=1)
                feat = torch.cat([cls, patch], dim=1)
            else:
                feat = cls
            feat = torch.nn.functional.normalize(feat, dim=1).float().cpu().numpy()
            if sums is None:
                dim = feat.shape[1]
                sums = np.zeros((n, dim), dtype=np.float32)
            for j, ti in enumerate(idx[b:b + batch]):
                sums[ti] += feat[j]
                counts[ti] += 1

    if sums is None:
        return {"cam": cam, "skipped": "no crops", "tracklets": n}
    vecs = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        if counts[i]:
            v = sums[i] / counts[i]
            vecs[i] = v / (np.linalg.norm(v) + 1e-12)
    (out / "vec").mkdir(exist_ok=True)
    tag = _tag(model.name_or_path, res, pool)
    if aug_tag:
        tag = f"{tag}_{aug_tag}"
    np.save(out / "vec" / f"reid_appearance_{tag}.npy", vecs)
    return {"cam": cam, "tracklets": n, "crops": len(imgs), "dim": dim, "file": f"reid_appearance_{tag}.npy"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--res", type=int, default=322, help="input size; rounded to a multiple of patch")
    ap.add_argument("--pool", default="cls_patch", choices=["cls_patch", "cls"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="DINOv3 overflows in fp16 -> NaN; use bf16")
    # synthetic "camera B" (augment.py): --aug camB for experiment A; single knobs for the
    # experiment-B sweeps override the preset (e.g. --aug identity --jpeg 30).
    ap.add_argument("--aug", default=None, help="augment preset name (e.g. camB, identity)")
    ap.add_argument("--jpeg", type=int, default=None, help="knob override: JPEG quality")
    ap.add_argument("--blur", type=float, default=None, help="knob override: Gaussian sigma")
    ap.add_argument("--resolution", type=float, default=None, help="knob override: downscale factor")
    ap.add_argument("--perspective", type=float, default=None, help="knob override: warp fraction")
    ap.add_argument("--aug-tag", default=None, help="explicit output tag (else derived from aug/knobs)")
    ap.add_argument("--sweep", default=None, help="knob to sweep for experiment B (jpeg/blur/...)")
    ap.add_argument("--sweep-vals", default=None, help="comma list, e.g. 80,55,40,30,22,15")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    patch = getattr(model.config, "patch_size", 14)
    res = (args.res // patch) * patch
    patch_start = 1 + getattr(model.config, "num_register_tokens", 0)
    print(f"model={args.model} dev={dev} patch={patch} res={res} "
          f"reg_tokens={patch_start - 1} pool={args.pool} hidden={model.config.hidden_size}")

    cams = args.cams or [p.name for p in sorted((paths.OUTPUT_ROOT / args.scene).iterdir())
                         if p.is_dir()]

    # Build (transform, tag) jobs — model stays loaded across all of them:
    #   --sweep KNOB       : experiment-B degradation curve (one file per --sweep-vals value)
    #   --aug / knob flags : single synthetic camera (experiment A)
    #   neither            : untouched baseline (one file, no transform)
    jobs: list[tuple] = []
    if args.sweep:
        vals = [float(x) if "." in x else int(x) for x in args.sweep_vals.split(",")]
        for v in vals:
            _, tf = augment.transform_for("identity", **{args.sweep: v})
            jobs.append((tf, f"{args.sweep}{v}"))
        print(f"sweep {args.sweep} over {vals}")
    else:
        overrides = {k: getattr(args, k) for k in ("jpeg", "blur", "resolution", "perspective")
                     if getattr(args, k) is not None}
        if args.aug or overrides:
            spec, tf = augment.transform_for(args.aug, **overrides)
            base = args.aug if (args.aug and args.aug != "identity") else ""
            knobs = "_".join(f"{k}{v}" for k, v in overrides.items())
            tag = args.aug_tag or "_".join(x for x in (base, knobs) if x)
            print(f"aug: preset={args.aug} overrides={overrides} -> spec={spec} tag={tag!r}")
            jobs.append((tf, tag))
        else:
            jobs.append((None, ""))

    for tf, tag in jobs:
        for cam in cams:
            print("  ", run(args.scene, cam, model, patch_start, res, args.pool, args.batch,
                            dev, dtype, tf, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
