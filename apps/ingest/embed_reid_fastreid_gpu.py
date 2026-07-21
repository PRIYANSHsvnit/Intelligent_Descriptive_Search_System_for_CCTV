"""embed_reid_fastreid_gpu.py — run a FastReID checkpoint on the GPU (torch 2.13) for the A/B.

Why not the ONNX path: the VeRi-Wild *bagtricks* model uses LAST_STRIDE=1 (large res-5 maps),
which onnxruntime-CPU runs at ~3.5 s/crop (hours). fast-reid runs fine under the pipeline's
torch 2.13 on the 4050 (~4 ms/crop) via the gpu_setup shim. Feeds the SAME raw-0-255-RGB the
ONNX export bakes normalization for → results equivalent to the (locked) ONNX path.

Non-destructive: writes vec/reid_appearance_<tag>.npy (aligned to tracklets.json), mean-pooled
over a tracklet's K crops + L2-normed (same as pipeline/embed_reid.py) — fair A/B.

Needs fast-reid's deps at runtime:
  uv run --with yacs --with termcolor --with tabulate --with scikit-learn --with Pillow \
    python embed_reid_fastreid_gpu.py --scene S01 \
    --config reid/fast-reid/configs/VERIWild/bagtricks_R50-ibn.yml \
    --weights reid/veriwild_bot_R50-ibn.pth --tag veriwild
"""
from __future__ import annotations

from gpu_setup import ensure_gpu_libs
ensure_gpu_libs()

import argparse
import json
import sys

import cv2
import numpy as np
import torch

import augment
from pipeline import paths
from pipeline.cfg import REID_INPUT_HW

sys.path.insert(0, str(paths.INGEST_ROOT / "reid" / "fast-reid"))
from fastreid.config import get_cfg               # noqa: E402
from fastreid.modeling.meta_arch import build_model  # noqa: E402
from fastreid.utils.checkpoint import Checkpointer   # noqa: E402

_BATCH = 64


def _detect_num_classes(weights: str, default: int = 30671) -> int:
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    for k, v in sd.items():
        if "classifier" in k and hasattr(v, "ndim") and v.ndim >= 1:
            return int(v.shape[0])
    return default


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = REID_INPUT_HW
    img = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # raw 0-255; model bakes norm
    return img.transpose(2, 0, 1)


def run(scene: str, cam: str, model, dev: str, tag: str, transform=None) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    idx, imgs = [], []
    for i, t in enumerate(tracklets):
        for k in t["crop_refs"]:
            im = cv2.imread(str(paths.OUTPUT_ROOT / k))
            if im is not None:
                if transform is not None:        # synthetic "camera B" degradation (augment.py)
                    im = transform(im)
                idx.append(i)
                imgs.append(_preprocess(im))

    dim = None
    sums = None
    counts = np.zeros(n, dtype=np.int32)
    with torch.inference_mode():
        for b in range(0, len(imgs), _BATCH):
            x = torch.from_numpy(np.stack(imgs[b:b + _BATCH])).to(dev)
            feats = model(x)
            feats = torch.nn.functional.normalize(feats, dim=1).float().cpu().numpy()
            if sums is None:
                dim = feats.shape[1]
                sums = np.zeros((n, dim), dtype=np.float32)
            for j, ti in enumerate(idx[b:b + _BATCH]):
                sums[ti] += feats[j]
                counts[ti] += 1

    if sums is None:
        return {"cam": cam, "skipped": "no crops", "tracklets": n}
    vecs = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        if counts[i]:
            v = sums[i] / counts[i]
            vecs[i] = v / (np.linalg.norm(v) + 1e-12)
    (out / "vec").mkdir(exist_ok=True)
    np.save(out / "vec" / f"reid_appearance_{tag}.npy", vecs)
    return {"cam": cam, "tracklets": n, "crops": len(imgs), "dim": dim,
            "file": f"reid_appearance_{tag}.npy"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="S01")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    # synthetic "camera B" (augment.py), same hooks as embed_reid_dino.py:
    ap.add_argument("--aug", default=None, help="augment preset (e.g. camB) for experiment A")
    ap.add_argument("--sweep", default=None, help="knob to sweep for experiment B (jpeg/blur/...)")
    ap.add_argument("--sweep-vals", default=None, help="comma list, e.g. 80,55,40,30,22,15")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = dev
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = _detect_num_classes(args.weights)
    cfg.freeze()
    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    print(f"loaded {args.config} weights={args.weights} dev={dev} "
          f"num_classes={cfg.MODEL.HEADS.NUM_CLASSES}")

    cams = args.cams or [p.name for p in sorted((paths.OUTPUT_ROOT / args.scene).iterdir())
                         if p.is_dir()]

    # (transform, tag) jobs — model stays loaded across all. Tags append to --tag so the
    # clean gallery (args.tag) and its degraded queries (args.tag_<knob><val>) share an encoder.
    jobs: list[tuple] = []
    if args.sweep:
        vals = [float(x) if "." in x else int(x) for x in args.sweep_vals.split(",")]
        for v in vals:
            _, tf = augment.transform_for("identity", **{args.sweep: v})
            jobs.append((tf, f"{args.tag}_{args.sweep}{v}"))
        print(f"sweep {args.sweep} over {vals}")
    elif args.aug:
        _, tf = augment.transform_for(args.aug)
        jobs.append((tf, f"{args.tag}_{args.aug}"))
    else:
        jobs.append((None, args.tag))

    for tf, tag in jobs:
        for cam in cams:
            print("  ", run(args.scene, cam, model, dev, tag, tf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
