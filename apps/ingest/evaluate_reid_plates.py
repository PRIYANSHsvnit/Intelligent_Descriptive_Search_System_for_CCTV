"""evaluate_reid_plates.py — Surat-native re-ID eval using PLATES as pseudo-ground-truth.

CityFlow has identity labels; SUR01 does not. But a *validated* `plate_text` IS an identity
label. On Surat no plate spans two cameras (user-verified), so every positive pair is
WITHIN one camera (same vehicle fragmented into >1 tracklet by occlusion / exit-reentry).
This scores how well an appearance encoder separates same-vehicle from different-vehicle
tracklets ON THE ACTUAL DELIVERABLE FOOTAGE — the number CityFlow can't give.

Encoder-agnostic: point --vecfile at any per-tracklet vector file (aligned to tracklets.json
row order) to A/B encoders. Metrics:
  * PRIMARY  — verification AUC (ROC-AUC of cosine sim, same-plate vs diff-plate pairs).
               Robust at small N (SUR01 is pilot-sized: c004 ~= 15 vehicles / 161 pos pairs).
  * SECONDARY— retrieval rank-1 / mAP (each plated tracklet queries the same-cam plated gallery).
  * plus mean pos/neg sim and the N behind every number (never over-read a pilot).

Usage:
  uv run python evaluate_reid_plates.py                          # SUR01, plate-rich cams
  uv run python evaluate_reid_plates.py --cams c004 --min-conf 0.0
  uv run python evaluate_reid_plates.py --vecfile reid_appearance_dinov3l.npy --cams c004
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from pipeline import paths


def _load_cam(scene: str, cam: str, vecfile: str, min_conf: float):
    """Return (plates, vecs, ts) for vehicle tracklets that have a validated plate + a
    non-zero appearance vector. Rows are filtered from the full aligned arrays, so the
    vector for tracklet i is vecs row i (alignment preserved before filtering)."""
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    vpath = out / "vec" / vecfile
    if not vpath.exists():
        return None
    app = np.load(vpath)
    if len(app) != len(tracklets):
        raise SystemExit(f"{cam}: {vecfile} has {len(app)} rows but {len(tracklets)} tracklets")

    plates, vecs, ts, subs, cols = [], [], [], [], []
    for i, t in enumerate(tracklets):
        if t.get("entity_type") != "vehicle":
            continue
        pt = t.get("plate_text")
        if not pt:
            continue
        if (t.get("plate_conf") or 0.0) < min_conf:
            continue
        v = app[i]
        n = np.linalg.norm(v)
        if n == 0 or not np.isfinite(n):
            continue
        plates.append(pt)
        vecs.append(v / n)
        ts.append(t.get("ts_start_s") if t.get("ts_start_s") is not None else np.nan)
        subs.append(t.get("subtype"))
        cols.append(t.get("color"))
    if not plates:
        z = np.zeros(0)
        return np.array([]), np.zeros((0, app.shape[1])), z, np.array([]), np.array([])
    return (np.array(plates), np.stack(vecs), np.array(ts, dtype=float),
            np.array(subs, dtype=object), np.array(cols, dtype=object))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U with average-rank tie handling. Dependency-free."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=float)
    j, N = 0, len(scores)
    while j < N:                                   # average ranks over tie-runs (1-based)
        k = j
        while k + 1 < N and s_sorted[k + 1] == s_sorted[j]:
            k += 1
        ranks_sorted[j:k + 1] = (j + k) / 2.0 + 1.0
        j = k + 1
    ranks = np.empty(N, dtype=float)
    ranks[order] = ranks_sorted
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pair_metrics(plates, vecs, ts, subs, cols, min_gap: float, hard_neg: str):
    """Upper-triangle pair scores → same-plate positives vs diff-plate negatives.
    hard_neg: 'off' = any diff-plate neg; 'subtype' = neg must share subtype;
    'subtype_color' = neg must share subtype AND color (similar-looking different vehicles)."""
    n = len(plates)
    sims = vecs @ vecs.T
    iu, ju = np.triu_indices(n, k=1)
    same = plates[iu] == plates[ju]
    pair_sim = sims[iu, ju]
    pos_keep = np.ones(len(iu), dtype=bool)
    if min_gap > 0:                                # drop too-adjacent fragments (too-easy positives)
        pos_keep = np.abs(ts[iu] - ts[ju]) >= min_gap
    neg_keep = np.ones(len(iu), dtype=bool)
    if hard_neg in ("subtype", "subtype_color"):
        neg_keep &= subs[iu] == subs[ju]
    if hard_neg == "subtype_color":
        neg_keep &= cols[iu] == cols[ju]
    pos = pair_sim[same & pos_keep]
    neg = pair_sim[(~same) & neg_keep]
    return pos, neg


def _retrieval(plates, vecs):
    """Per plated tracklet: rank same-cam gallery, positives = same plate. rank-1 + mAP."""
    n = len(plates)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    r1 = ap_sum = valid = 0
    for i in range(n):
        rel = plates == plates[i]
        rel[i] = False
        if not rel.any():                          # no same-plate partner → undefined query
            continue
        valid += 1
        order = np.argsort(-sims[i])
        order = order[order != i]
        hit = rel[order]
        if hit[0]:
            r1 += 1
        # average precision
        cum = np.cumsum(hit)
        prec = cum / (np.arange(len(hit)) + 1)
        ap_sum += (prec[hit]).sum() / hit.sum()
    return {
        "queries": valid,
        "rank1": round(r1 / valid, 4) if valid else None,
        "mAP": round(ap_sum / valid, 4) if valid else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cams", nargs="*", default=None, help="default: all cams with plate data")
    ap.add_argument("--vecfile", default="reid_appearance.npy",
                    help="per-tracklet vector file under <cam>/vec/ (aligned to tracklets.json)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="min plate_conf to trust a plate")
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="seconds; drop same-plate pairs closer than this (anti too-easy-fragment)")
    ap.add_argument("--hard-neg", default="off", choices=["off", "subtype", "subtype_color"],
                    help="restrict negatives to similar-looking different vehicles")
    args = ap.parse_args()

    cams = args.cams or [p.name for p in sorted((paths.OUTPUT_ROOT / args.scene).iterdir())
                         if p.is_dir()]

    print(f"scene={args.scene}  vecfile={args.vecfile}  min_conf={args.min_conf}  "
          f"min_gap={args.min_gap}s  hard_neg={args.hard_neg}\n")
    print(f"{'cam':6} {'plated':>6} {'plates':>6} {'pos':>5} {'neg':>6} "
          f"{'AUC':>6} {'posSim':>7} {'negSim':>7} {'rank1':>6} {'mAP':>6}  {'queries':>7}")

    all_pos, all_neg = [], []
    for cam in cams:
        loaded = _load_cam(args.scene, cam, args.vecfile, args.min_conf)
        if loaded is None:
            print(f"{cam:6}  (no {args.vecfile})")
            continue
        plates, vecs, ts, subs, cols = loaded
        if len(plates) < 2:
            print(f"{cam:6} {len(plates):6}  (too few plated tracklets)")
            continue
        pos, neg = _pair_metrics(plates, vecs, ts, subs, cols, args.min_gap, args.hard_neg)
        if len(pos) == 0:
            print(f"{cam:6} {len(plates):6} {len(set(plates)):6} {0:5} {len(neg):6}  "
                  f"(no positive pairs)")
            continue
        all_pos.append(pos); all_neg.append(neg)
        ret = _retrieval(plates, vecs)
        print(f"{cam:6} {len(plates):6} {len(set(plates)):6} {len(pos):5} {len(neg):6} "
              f"{_auc(pos, neg):6.4f} {pos.mean():7.4f} {neg.mean():7.4f} "
              f"{str(ret['rank1']):>6} {str(ret['mAP']):>6}  {ret['queries']:7}")

    if all_pos:
        P, N = np.concatenate(all_pos), np.concatenate(all_neg)
        print(f"\n[POOLED across cams]  pos={len(P)} neg={len(N)}  "
              f"AUC={_auc(P, N):.4f}  posSim={P.mean():.4f}  negSim={N.mean():.4f}")
    else:
        print("\n[POOLED] no positive pairs found — nothing to score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
