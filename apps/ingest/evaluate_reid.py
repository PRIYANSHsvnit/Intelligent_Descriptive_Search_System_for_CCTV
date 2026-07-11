"""evaluate_reid.py — the money-metric harness (our glue; metric math from motmetrics).

Two jobs, two metrics (do not conflate — see plan.md Verification):
  * IDF1/IDP/IDR = headline (scores the cross-camera GROUPING as a whole).
  * rank-1/rank-5 = tuning diagnostic ONLY (retrieval).

Pipeline:
  (1) IoU-match our predicted tracklets -> GT ids (full fps, IoU>=0.5), majority vote;
      EXCLUDE tracklets with no GT (GT only labels multi-camera vehicles).
  (2) Build the scene GT from per-camera gt/gt.txt (comma-delimited frame,id,x,y,w,h).
  (3) rank-1/5 over the fused reid vectors (cross-camera).
  (4) IDF1 via motmetrics; an --oracle smoke-test feeds GT as the hypothesis and must
      report ~100%, proving the harness is wired right.

IDF1 on Phase-1 (per-camera-only) output is ~0 by construction — meaningful only after
Phase 3 assigns global_id. rank-1/5 and the oracle already work in Phase 1.

Usage:
  uv run python evaluate_reid.py --scene S01                 # rank-1/5 + oracle
  uv run python evaluate_reid.py --scene S01 --oracle-only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from pipeline import paths, profiles

# motmetrics 1.4 predates NumPy 2.0 and calls np.asfarray (removed in 2.0). Shim it.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

IOU_THRESH = 0.5


# --------------------------------------------------------------------------- GT / preds
def load_gt(scene: str, cam: str) -> dict[int, list[tuple]]:
    """gt/gt.txt (comma: frame,id,x,y,w,h,...) -> {frame: [(gt_id, x,y,w,h), ...]}."""
    gt: dict[int, list[tuple]] = defaultdict(list)
    f = paths.cam_gt(scene, cam)
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split(",")
        frame, gid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        gt[frame].append((gid, x, y, w, h))
    return gt


def load_pred(scene: str, cam: str):
    """detections.npy -> {frame: [(tid, x,y,w,h), ...]}, plus tracklets.json list."""
    out = paths.cam_out(scene, cam)
    det = np.load(out / "detections.npy")  # frame,tid,x1,y1,x2,y2,conf,cls
    by_frame: dict[int, list[tuple]] = defaultdict(list)
    for frame, tid, x1, y1, x2, y2, _conf, _cls in det:
        by_frame[int(frame)].append((int(tid), x1, y1, x2 - x1, y2 - y1))
    tracklets = json.loads((out / "tracklets.json").read_text())
    return by_frame, tracklets


def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _iou_dist_matrix(objs, hyps):
    """(len(objs), len(hyps)) matrix of 1-IoU, NaN where IoU < IOU_THRESH (motmetrics gate)."""
    m = np.full((len(objs), len(hyps)), np.nan, dtype=np.float64)
    for i, o in enumerate(objs):
        for j, h in enumerate(hyps):
            iou = _iou_xywh(o, h)
            if iou >= IOU_THRESH:
                m[i, j] = 1.0 - iou
    return m


def inherit_gt_ids(scene: str, cam: str) -> dict[int, int]:
    """Per predicted tracklet (track id), majority-vote its per-frame best-IoU GT id.
    Returns {track_id: gt_id} only for tracklets that matched GT (unlabeled excluded)."""
    gt = load_gt(scene, cam)
    by_frame, _ = load_pred(scene, cam)
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame, preds in by_frame.items():
        gboxes = gt.get(frame, [])
        if not gboxes:
            continue
        for tid, px, py, pw, ph in preds:
            best_iou, best_gid = 0.0, None
            for gid, gx, gy, gw, gh in gboxes:
                iou = _iou_xywh((px, py, pw, ph), (gx, gy, gw, gh))
                if iou > best_iou:
                    best_iou, best_gid = iou, gid
            if best_gid is not None and best_iou >= IOU_THRESH:
                votes[tid][best_gid] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items() if c}


# --------------------------------------------------------------------------- rank-1/5
def _fused_vectors(scene: str, cams: list[str], w: float):
    """Return per-tracklet fused reid vectors with inherited GT ids, across cams."""
    entries = []  # (cam, gt_id, fused_vec)
    for cam in cams:
        out = paths.cam_out(scene, cam)
        tracklets = json.loads((out / "tracklets.json").read_text())
        color = np.load(out / "vec" / "reid_color.npy")
        app_p = out / "vec" / "reid_appearance.npy"
        app = np.load(app_p) if app_p.exists() else None
        tid2gt = inherit_gt_ids(scene, cam)
        # track_id order in tracklets.json matches vec row order
        for i, t in enumerate(tracklets):
            gid = tid2gt.get(t["track_id"])
            if gid is None:
                continue
            if app is not None:
                fused = np.concatenate([app[i], w * color[i]])
            else:
                fused = color[i].copy()
            n = np.linalg.norm(fused)
            if n > 0:
                fused = fused / n
            entries.append((cam, gid, fused))
    return entries


def rank_cmc(scene: str, cams: list[str], w: float = 1.0) -> dict:
    entries = _fused_vectors(scene, cams, w)
    if not entries:
        return {"queries": 0, "rank1": None, "rank5": None}
    vecs = np.stack([e[2] for e in entries])
    cams_arr = np.array([e[0] for e in entries])
    gids = np.array([e[1] for e in entries])
    sims = vecs @ vecs.T
    r1 = r5 = valid = 0
    for i in range(len(entries)):
        other = cams_arr != cams_arr[i]
        if not other.any():
            continue
        # a query only counts if a correct cross-camera match exists in the gallery
        correct = other & (gids == gids[i])
        if not correct.any():
            continue
        valid += 1
        order = np.argsort(-sims[i][other])
        gal_gids = gids[other][order]
        if gal_gids[0] == gids[i]:
            r1 += 1
        if (gal_gids[:5] == gids[i]).any():
            r5 += 1
    return {
        "queries": valid,
        "rank1": round(r1 / valid, 4) if valid else None,
        "rank5": round(r5 / valid, 4) if valid else None,
        "appearance": (paths.cam_out(scene, cams[0]) / "vec" / "reid_appearance.npy").exists(),
        "fusion_w": w,
    }


# --------------------------------------------------------------------------- IDF1 (oracle)
def idf1_oracle(scene: str, cams: list[str]) -> dict:
    """Feed GT boxes+ids as BOTH reference and hypothesis -> IDF1 must be ~1.0.
    Proves the motmetrics plumbing (per (cam,frame) timestamp) before trusting real runs."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for ci, cam in enumerate(cams):
        base = ci * 10_000_000  # unique numeric (cam,frame) timestamp block
        gt = load_gt(scene, cam)
        for frame in sorted(gt):
            ids = [g[0] for g in gt[frame]]
            boxes = [(g[1], g[2], g[3], g[4]) for g in gt[frame]]  # x,y,w,h
            dist = _iou_dist_matrix(boxes, boxes)
            acc.update(ids, ids, dist, frameid=base + frame)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["idf1", "idp", "idr", "num_objects"], name="oracle")
    return {k: float(summary[k].iloc[0]) for k in ["idf1", "idp", "idr", "num_objects"]}


def idf1_real(scene: str, cams: list[str], gid_map: dict[str, int]) -> dict:
    """HEADLINE metric. Hypothesis = our predicted boxes labeled by predicted global_id,
    restricted to GT-matched tracklets (Trap 2: exclude unlabeled). Reference = GT boxes
    labeled by gt id. IDF1 over (cam,frame) timestamps via motmetrics."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for ci, cam in enumerate(cams):
        base = ci * 10_000_000  # unique numeric (cam,frame) timestamp block
        gt = load_gt(scene, cam)
        by_frame, _ = load_pred(scene, cam)
        tid2gt = inherit_gt_ids(scene, cam)  # track_id -> gt id (GT-matched only)
        for frame in sorted(gt):
            g = gt[frame]
            gids = [x[0] for x in g]
            gboxes = [(x[1], x[2], x[3], x[4]) for x in g]
            hids, hboxes = [], []
            for tid, x, y, wd, ht in by_frame.get(frame, []):
                if tid not in tid2gt:               # unlabeled prediction -> excluded
                    continue
                gid = gid_map.get(f"{scene}_{cam}_t{tid}")
                if gid is None:
                    continue
                hids.append(gid)
                hboxes.append((x, y, wd, ht))
            dist = _iou_dist_matrix(gboxes, hboxes)
            acc.update(gids, hids, dist, frameid=base + frame)

    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "idp", "idr", "num_objects", "num_predictions"], name="real")
    return {k: round(float(s[k].iloc[0]), 4) for k in ["idf1", "idp", "idr"]} | {
        "num_objects": int(s["num_objects"].iloc[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="S01")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--fusion-w", type=float, default=1.0)
    ap.add_argument("--oracle-only", action="store_true")
    ap.add_argument("--profile", default="cityflow", choices=list(profiles.PROFILES))
    args = ap.parse_args()
    profiles.use(args.profile)
    cams = args.cams or paths.list_cams(args.scene)

    print(f"scene={args.scene} cams={cams}")
    print("\n[oracle IDF1 smoke-test] (must be ~1.0)")
    print("  ", idf1_oracle(args.scene, cams))

    if not args.oracle_only:
        print("\n[rank-1/5 cross-camera] (tuning diagnostic only)")
        print("  ", rank_cmc(args.scene, cams, w=args.fusion_w))

        gid_path = paths.OUTPUT_ROOT / args.scene / "global_ids.json"
        if gid_path.exists():
            gid_map = {k: int(v) for k, v in json.loads(gid_path.read_text()).items()}
            print("\n[IDF1 HEADLINE] (real grouping vs gt.txt)")
            print("  ", idf1_real(args.scene, cams, gid_map))
        else:
            print("\n[IDF1 HEADLINE] no global_ids.json yet — run the Phase-3 matcher first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
