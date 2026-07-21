"""evaluate_reid_synthetic.py — SYNTHETIC cross-camera re-ID eval (experiment A).

Surat has no multi-camera footage, so `camera B` is a transform-degraded copy of camera A
(see augment.py). Because B is a copy, the identity link is EXACT and free: query vector for
tracklet i (embedded from its degraded crops) must retrieve gallery tracklet i (embedded from
its original crops) — the same object seen "by another camera". Same-plate fragments of one
vehicle count as relevant too.

    gallery  = camera A  (original crops)     -> --gallery reid_appearance_<enc>.npy
    query    = camera B  (augmented crops)    -> --query   reid_appearance_<enc>_camB.npy

Metrics:
  * cross-camera retrieval over ALL vehicles — rank-1 / mAP against the full gallery
    (relevant(i) = {i} + same-plate). The headline: "a camera-B query finds its camera-A
    twin as top-1, out of N distractors, X% of the time."
  * cross-view verification AUC on the PLATED subset with subtype+color HARD negatives
    (same definition as evaluate_reid_plates.py) — same-object-cross-view vs
    similar-but-different-vehicle.

Because B shares A's background and pose, every number here is an UPPER BOUND on a real second
camera; label it "synthetic cross-camera", never "cross-camera re-ID works".

Usage:
  uv run python evaluate_reid_synthetic.py --cams c004 \
      --gallery reid_appearance_dinov3vitl16_320_cls_patch.npy \
      --query   reid_appearance_dinov3vitl16_320_cls_patch_camB.npy
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pipeline import paths


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U with average-rank ties (matches evaluate_reid_plates.py)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=float)
    j, N = 0, len(scores)
    while j < N:
        k = j
        while k + 1 < N and s_sorted[k + 1] == s_sorted[j]:
            k += 1
        ranks_sorted[j:k + 1] = (j + k) / 2.0 + 1.0
        j = k + 1
    ranks = np.empty(N, dtype=float)
    ranks[order] = ranks_sorted
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _load(scene: str, cam: str, gallery: str, query: str, min_conf: float):
    """Aligned camera-A / camera-B vectors for VEHICLE tracklets valid (finite, non-zero) in
    BOTH files. Returns dict of parallel arrays or None if a file is missing."""
    out = paths.cam_out(scene, cam)
    tr = json.loads((out / "tracklets.json").read_text())
    gp, qp = out / "vec" / gallery, out / "vec" / query
    if not gp.exists() or not qp.exists():
        return None
    A, B = np.load(gp), np.load(qp)
    for name, arr in ((gallery, A), (query, B)):
        if len(arr) != len(tr):
            raise SystemExit(f"{cam}: {name} has {len(arr)} rows but {len(tr)} tracklets")

    keep, plate, sub, col = [], [], [], []
    for i, t in enumerate(tr):
        if t.get("entity_type") != "vehicle":
            continue
        a, b = A[i], B[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0 or not np.isfinite(na) or not np.isfinite(nb):
            continue
        keep.append(i)
        pt = t.get("plate_text")
        plate.append(pt if (pt and (t.get("plate_conf") or 0.0) >= min_conf) else None)
        sub.append(t.get("subtype"))
        col.append(t.get("color"))
    keep = np.array(keep)
    return dict(A=_norm(A[keep]), B=_norm(B[keep]),
                plate=np.array(plate, dtype=object),
                sub=np.array(sub, dtype=object), col=np.array(col, dtype=object))


def _retrieval(A, B, plate):
    """Query = B (camera B), gallery = A (camera A). relevant(i) = {i} + same valid plate."""
    n = len(A)
    S = B @ A.T                                     # (query i) x (gallery j)
    r1 = ap_sum = 0
    self_sim = S.diagonal().copy()
    for i in range(n):
        rel = np.zeros(n, dtype=bool)
        rel[i] = True                               # the same object, seen "by camera B"
        if plate[i] is not None:
            rel |= (plate == plate[i])
        order = np.argsort(-S[i], kind="mergesort")
        hit = rel[order]
        if hit[0]:
            r1 += 1
        cum = np.cumsum(hit)
        prec = cum / (np.arange(n) + 1)
        ap_sum += prec[hit].sum() / hit.sum()
    return dict(gallery=n, rank1=r1 / n, mAP=ap_sum / n, self_sim=float(self_sim.mean()))


def _verification(A, B, plate, sub, col, hard_neg: str):
    """pos = self-twin cross-view cos (same object). neg = cos(B_i, A_j) different vehicle,
    optionally restricted to same subtype (+color) = hard look-alikes. Plated rows only."""
    m = np.array([p is not None for p in plate])
    A, B, plate, sub, col = A[m], B[m], plate[m], sub[m], col[m]
    n = len(A)
    if n < 2:
        return None
    S = B @ A.T
    pos = S.diagonal().copy()                       # B_i vs A_i, same object
    iu, ju = np.where(~np.eye(n, dtype=bool))       # all ordered off-diagonal (B_i vs A_j)
    diff = plate[iu] != plate[ju]
    keep = diff.copy()
    if hard_neg in ("subtype", "subtype_color"):
        keep &= sub[iu] == sub[ju]
    if hard_neg == "subtype_color":
        keep &= col[iu] == col[ju]
    neg = S[iu[keep], ju[keep]]
    return dict(plated=n, npos=len(pos), nneg=int(len(neg)),
                auc=_auc(pos, neg), posSim=float(pos.mean()),
                negSim=float(neg.mean()) if len(neg) else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cams", nargs="*", default=["c004"])
    ap.add_argument("--gallery", required=True, help="camera-A vecfile (original crops)")
    ap.add_argument("--query", required=True, help="camera-B vecfile (augmented crops)")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--hard-neg", default="subtype_color",
                    choices=["off", "subtype", "subtype_color"])
    ap.add_argument("--dump", default=None, help="write metrics JSON here (for the showcase)")
    args = ap.parse_args()

    print(f"scene={args.scene}  gallery(A)={args.gallery}\n"
          f"                     query(B)  ={args.query}  hard_neg={args.hard_neg}\n")
    print(f"{'cam':6} {'vehicles':>8} {'rank1':>7} {'mAP':>7} {'selfSim':>8}  "
          f"{'plated':>6} {'AUC':>7} {'posSim':>7} {'negSim':>7}")

    dump = {}
    for cam in args.cams:
        d = _load(args.scene, cam, args.gallery, args.query, args.min_conf)
        if d is None:
            print(f"{cam:6}  (missing {args.gallery} or {args.query})")
            continue
        ret = _retrieval(d["A"], d["B"], d["plate"])
        ver = _verification(d["A"], d["B"], d["plate"], d["sub"], d["col"], args.hard_neg)
        vtxt = (f"{ver['plated']:>6} {ver['auc']:7.4f} {ver['posSim']:7.4f} {ver['negSim']:7.4f}"
                if ver else f"{'--':>6} {'--':>7} {'--':>7} {'--':>7}")
        print(f"{cam:6} {ret['gallery']:>8} {ret['rank1']:7.4f} {ret['mAP']:7.4f} "
              f"{ret['self_sim']:8.4f}  {vtxt}")
        dump[cam] = dict(retrieval=ret, verification=ver)

    if args.dump:
        json.dump(dump, open(args.dump, "w"), indent=2)
        print(f"\nwrote {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
