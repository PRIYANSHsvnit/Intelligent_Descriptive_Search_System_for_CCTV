"""evaluate_person_identity.py — can appearance alone tell two subjects apart?

Two free sources of ground truth, so this needs no hand labelling:

  SAME subject = two crops of the SAME tracklet (the tracker asserts continuity)
  DIFF subject = crops of two CO-VISIBLE tracklets — they overlap in time on one
                 camera, so both are on screen at once and cannot be the same human.

Part 1 (separability) scores how far apart those two distributions sit. The headline is
the confusion rate: how often a KNOWN-DIFFERENT subject outranks the target's own second
view. On SUR01/c002 (dense market, median person 100px tall) that is ~43% — meaning
appearance cannot assert identity there at any threshold.

Part 2 (gate audit) runs the SHIPPED gates from pipeline/matcher.py over the same
tracklets and attributes every rejected candidate merge to the gate that caught it (L1
cannot-link / gap window / L2 spatial reach / L3 margin). This is what turns Part 1's
verdict into a configuration you can defend.

Numbers are reported per camera because crowd density and pixel height dominate them.

Usage:
  uv run python evaluate_person_identity.py --scene SUR01
  uv run python evaluate_person_identity.py --scene SUR01 --cams c002 --sample 700
  uv run python evaluate_person_identity.py --scene SUR01 --entity vehicle
  # A/B a different encoder: any npz with the vec/semantic_crops.npz schema
  # (vectors, tracklet_indices, crop_indices) plus its tracklet-level .npy
  uv run python evaluate_person_identity.py --crop-vectors reid_crops_msmt.npz \
      --vec-file reid_appearance_msmt.npy
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from pipeline import matcher, paths


def _load_camera(scene: str, cam: str, entity: str, crop_file: str, vec_file: str,
                 sample: int, rng: np.random.Generator):
    """-> (indices, tracklets, per-tracklet crop vectors, tracklet-level vectors)."""
    d = paths.cam_out(scene, cam)
    tracklets = json.loads((d / "tracklets.json").read_text())
    npz = np.load(d / "vec" / crop_file)
    crop_vecs, crop_ti = npz["vectors"], npz["tracklet_indices"]
    try:
        track_vecs = np.load(d / "vec" / vec_file)
    except FileNotFoundError:
        track_vecs = None

    by_tracklet: dict[int, list[int]] = {}
    for row, ti in enumerate(crop_ti):
        by_tracklet.setdefault(int(ti), []).append(row)

    keep = [i for i, t in enumerate(tracklets)
            if t.get("entity_type") == entity and len(by_tracklet.get(i, ())) >= 2]
    if sample and len(keep) > sample:
        keep = sorted(rng.choice(keep, size=sample, replace=False).tolist())

    vecs = {}
    for i in keep:
        V = crop_vecs[by_tracklet[i]].astype(np.float32)
        vecs[i] = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return keep, tracklets, vecs, track_vecs


def _separability(keep, tracklets, vecs) -> dict:
    same: list[float] = []
    for i in keep:
        V = vecs[i]
        s = V @ V.T
        same += s[np.triu_indices(len(V), k=1)].tolist()

    ts0 = np.array([tracklets[i]["ts_start_s"] for i in keep])
    ts1 = np.array([tracklets[i]["ts_end_s"] for i in keep])
    covisible = (ts0[:, None] <= ts1[None, :]) & (ts0[None, :] <= ts1[:, None])
    np.fill_diagonal(covisible, False)

    diff: list[float] = []
    confused = checked = 0
    for a, ia in enumerate(keep):
        VA = vecs[ia]
        own = float((VA @ VA.T)[np.triu_indices(len(VA), k=1)].min())
        best = -np.inf
        for b in np.flatnonzero(covisible[a]):
            s = float((VA @ vecs[keep[b]].T).max())
            diff.append(s)
            best = max(best, s)
        if np.isfinite(best):
            checked += 1
            confused += best >= own
    return {"same": np.array(same), "diff": np.array(diff),
            "confused": confused, "checked": checked,
            "covisible_pairs": int(covisible.sum() // 2)}


def _gate_audit(scene, cam, keep, tracklets, track_vecs, entity, threshold) -> dict | None:
    """Attribute every above-threshold same-camera candidate to the gate that rejects it."""
    if track_vecs is None:
        return None
    pos = matcher._positions(scene, cam)
    gate = matcher.gate_for(entity)
    tk = []
    for i in keep:
        t = tracklets[i]
        v = track_vecs[i].astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        tk.append(matcher.Tracklet(t["tracklet_id"], cam, t["ts_start_s"], t["ts_end_s"], v,
                                   t.get("plate_text"), entity,
                                   pos.get((entity == "person", int(t["track_id"])))))
    n = len(tk)
    if n < 2:
        return None
    V = np.stack([t.vec for t in tk])
    sims = V @ V.T

    above = covis = beyond_gap = unreachable = 0
    for a in range(n):
        for b in range(a + 1, n):
            if sims[a, b] < threshold:
                continue
            above += 1
            if matcher._overlap(tk[a], tk[b]):
                covis += 1
            elif matcher._gap(tk[a], tk[b]) > gate.max_gap:
                beyond_gap += 1
            elif not matcher._spatial_ok(tk[a], tk[b], gate):
                unreachable += 1

    A = list(range(n))
    with_margin = len(matcher._mutual_edges(tk, sims, A, A, threshold, True, gate))
    no_margin = len(matcher._mutual_edges(tk, sims, A, A, threshold, True,
                                          gate._replace(margin=0.0)))
    return {"above": above, "covis": covis, "beyond_gap": beyond_gap,
            "unreachable": unreachable, "no_margin": no_margin, "with_margin": with_margin,
            "positions": sum(1 for t in tk if t.h is not None), "n": n, "gate": gate}


def _report(cam, keep, stats, audit, threshold):
    same, diff = stats["same"], stats["diff"]
    print(f"\n=== {cam}: {len(keep)} tracklets (>=2 crops), "
          f"{stats['covisible_pairs']} co-visible pairs ===")
    if not len(same) or not len(diff):
        print("  not enough pairs to score")
        return
    q = np.quantile
    print(f"  SAME subject   n={len(same):7d}  median {q(same, .5):.3f}  p10 {q(same, .1):.3f}")
    print(f"  DIFF (covis)   n={len(diff):7d}  median {q(diff, .5):.3f}  "
          f"p90 {q(diff, .9):.3f}  max {diff.max():.3f}")
    # cap the AUC cross-product; subsample evenly so the estimate is not biased toward
    # whichever tracklets happen to sort first
    cap = diff if len(diff) <= 4000 else diff[np.linspace(0, len(diff) - 1, 4000).astype(int)]
    print(f"  separability AUC (same > diff)  {(same[:, None] > cap[None, :]).mean():.3f}"
          "   [0.5 = coin flip]")
    for thr in (0.80, 0.85, 0.90, 0.95):
        print(f"    thr {thr:.2f}: keeps {100 * (same >= thr).mean():5.1f}% of true pairs, "
              f"admits {100 * (diff >= thr).mean():5.1f}% of KNOWN-different pairs")
    pct = 100 * stats["confused"] / max(stats["checked"], 1)
    print(f"  CONFUSION: a known-different subject outranks the target's own second view "
          f"in {stats['confused']}/{stats['checked']} = {pct:.1f}% of tracklets")

    if audit is None:
        print("  gate audit: skipped (tracklet-level vector file missing)")
        return
    g = audit["gate"]
    print(f"  gate audit @ sim>={threshold:.2f}  (positions known for "
          f"{audit['positions']}/{audit['n']} tracklets)")
    print(f"    candidate same-camera pairs above threshold : {audit['above']}")
    print(f"    L1 rejected, co-visible                     : {audit['covis']}")
    print(f"    rejected, gap > {g.max_gap:.0f}s window                : {audit['beyond_gap']}")
    print(f"    L2 rejected, not physically reachable       : {audit['unreachable']}")
    print(f"    surviving mutual-NN edges (margin off)      : {audit['no_margin']}")
    print(f"    L3 kept after margin>={g.margin:.2f} (abstained {audit['no_margin'] - audit['with_margin']})"
          f"      : {audit['with_margin']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--entity", default="person", choices=("person", "vehicle"))
    ap.add_argument("--sample", type=int, default=800,
                    help="tracklets per camera (0 = all; pair counts grow quadratically)")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="candidate-merge similarity used for the gate audit")
    ap.add_argument("--crop-vectors", default="semantic_crops.npz",
                    help="per-crop npz under vec/ (vectors, tracklet_indices, crop_indices)")
    ap.add_argument("--vec-file", default="semantic.npy",
                    help="tracklet-level npy under vec/, used for the gate audit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cams = args.cams or paths.list_cams(args.scene)
    rng = np.random.default_rng(args.seed)
    print(f"scene={args.scene} entity={args.entity} vectors={args.crop_vectors} "
          f"sample={args.sample or 'all'}/cam")
    for cam in cams:
        try:
            keep, tracklets, vecs, track_vecs = _load_camera(
                args.scene, cam, args.entity, args.crop_vectors, args.vec_file,
                args.sample, rng)
        except FileNotFoundError as exc:
            print(f"\n=== {cam}: skipped ({exc.filename or exc}) ===")
            continue
        if len(keep) < 2:
            print(f"\n=== {cam}: skipped (only {len(keep)} usable {args.entity} tracklets) ===")
            continue
        stats = _separability(keep, tracklets, vecs)
        audit = _gate_audit(args.scene, cam, keep, tracklets, track_vecs,
                            args.entity, args.threshold)
        _report(cam, keep, stats, audit, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
