"""embed_reid_onnx.py — non-destructive FastReID-ONNX encoder for the re-ID A/B.

Same math as pipeline/embed_reid.py (CPU onnxruntime, raw 0-255 RGB, mean-pool over a
tracklet's K crops, L2-norm) but takes ANY exported FastReID ONNX via --onnx and writes a
TAGGED vector file (vec/reid_appearance_<tag>.npy) so it never clobbers the locked
reid_appearance.npy baseline. Point evaluate_reid*.py at the tag to A/B (e.g. VeRi vs VeRi-Wild).

Usage:
  uv run python embed_reid_onnx.py --scene S01    --onnx models/veriwild_reid.onnx --tag veriwild
  uv run python embed_reid_onnx.py --scene SUR01 --cams c004 --onnx models/veriwild_reid.onnx --tag veriwild
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import onnxruntime as ort

from pipeline import paths
from pipeline.cfg import REID_INPUT_HW

_BATCH = 64


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = REID_INPUT_HW
    img = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # raw 0-255; ONNX bakes norm
    return img.transpose(2, 0, 1)


def run(scene: str, cam: str, sess, in_name, out_name, tag: str) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    idx, imgs = [], []
    for i, t in enumerate(tracklets):
        for k in t["crop_refs"]:
            im = cv2.imread(str(paths.OUTPUT_ROOT / k))
            if im is not None:
                idx.append(i)
                imgs.append(_preprocess(im))

    dim = None
    sums = None
    counts = np.zeros(n, dtype=np.int32)
    for b in range(0, len(imgs), _BATCH):
        batch = np.stack(imgs[b:b + _BATCH]).astype(np.float32)
        feats = sess.run([out_name], {in_name: batch})[0]
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
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
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", required=True, help="output suffix: reid_appearance_<tag>.npy")
    args = ap.parse_args()

    sess = ort.InferenceSession(str(paths.INGEST_ROOT / args.onnx if not args.onnx.startswith("/")
                                    else args.onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    cams = args.cams or [p.name for p in sorted((paths.OUTPUT_ROOT / args.scene).iterdir())
                         if p.is_dir()]
    for cam in cams:
        print("  ", run(args.scene, cam, sess, in_name, out_name, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
