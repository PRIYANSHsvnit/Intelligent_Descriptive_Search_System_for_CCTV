"""reid_preproc_probe.py — is the 47% rank-1 a PREPROCESSING BUG or the DOMAIN GAP?

Context (do not conflate): the "47%" is cross-camera retrieval on S01 CityFlow
(evaluate_reid.rank_cmc), NOT VeRi-776. The same FastReID checkpoint scores 97.0%
rank-1 on VeRi's own test set. VeRi-776 is license-gated and not on disk, so instead
of a VeRi probe we ABLATE the preprocessing: re-encode the S01 crops with several
candidate `_preprocess` variants and score cross-camera rank-1/5/mAP for each.

What each variant tells us:
  * baseline      = exactly pipeline/embed_reid._preprocess (raw 0-255 RGB, cubic, 256).
                    Must reproduce the documented ~47.1% (validates the harness).
  * bilinear/area = interpolation sensitivity (FastReID test-time uses bilinear).
  * bgr_swap_off  = SANITY: wrong channel order. Must score clearly WORSE (proves the
                    model uses color and that our RGB swap matters).
  * ext_norm      = DECISIVE: subtract pixel_mean / divide pixel_std ON TOP of baseline.
                    The export bakes normalization into the ONNX graph (reid/README
                    "CONFIRMED"), so this should DESTROY accuracy (double-normalized).
                    If ext_norm instead BEATS baseline, normalization is NOT baked in and
                    embed_reid feeds un-normalized input = a real preprocessing bug.
  * letterbox     = aspect-ratio preserved (vs the 256x256 squash the model trained on).

Verdict logic: if baseline wins / ties the interpolation variants and ext_norm tanks,
the 47% is the genuine VeRi->CityFlow domain gap, not a bug. If some variant clearly
beats baseline, that variant is the fix for embed_reid._preprocess.

  uv run python reid_preproc_probe.py --scene S01              # best crop per tracklet (fast)
  uv run python reid_preproc_probe.py --scene S01 --all-crops  # all K crops (matches stored ~47.1%)
  uv run python reid_preproc_probe.py --scene S01 --variants baseline ext_norm
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

import evaluate_reid
from pipeline import paths
from pipeline.cfg import REID_APPEARANCE_DIM, REID_INPUT_HW, REID_MEAN, REID_STD

MODEL_PATH = paths.INGEST_ROOT / "models" / "veri_reid.onnx"
_H, _W = REID_INPUT_HW
_MEAN = np.asarray(REID_MEAN, dtype=np.float32)
_STD = np.asarray(REID_STD, dtype=np.float32)


# --------------------------------------------------------------------------- preprocess variants
def _rgb(bgr, interp):
    r = cv2.resize(bgr, (_W, _H), interpolation=interp)
    return cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)


def pp_baseline(bgr):                       # == embed_reid._preprocess
    return _rgb(bgr, cv2.INTER_CUBIC).transpose(2, 0, 1)


def pp_bilinear(bgr):
    return _rgb(bgr, cv2.INTER_LINEAR).transpose(2, 0, 1)


def pp_area(bgr):
    return _rgb(bgr, cv2.INTER_AREA).transpose(2, 0, 1)


def pp_bgr_swap_off(bgr):                   # sanity: keep BGR (wrong order)
    r = cv2.resize(bgr, (_W, _H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return r.transpose(2, 0, 1)


def pp_ext_norm(bgr):                       # decisive: external mean/std on top of baked-in
    img = (_rgb(bgr, cv2.INTER_CUBIC) - _MEAN) / _STD
    return img.transpose(2, 0, 1)


def pp_letterbox(bgr):                      # aspect-ratio preserved, gray pad
    h, w = bgr.shape[:2]
    s = min(_W / w, _H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((_H, _W, 3), 128, np.uint8)
    y0, x0 = (_H - nh) // 2, (_W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)


VARIANTS = {
    "baseline": pp_baseline,
    "bilinear": pp_bilinear,
    "area": pp_area,
    "bgr_swap_off": pp_bgr_swap_off,
    "ext_norm": pp_ext_norm,
    "letterbox": pp_letterbox,
}


# --------------------------------------------------------------------------- embed + metrics
def _run_batched(sess, in_name, out_name, imgs, batch=64):
    outs = []
    for b in range(0, len(imgs), batch):
        arr = np.stack(imgs[b:b + batch]).astype(np.float32)
        f = sess.run([out_name], {in_name: arr})[0]
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)   # per-crop L2 (matches embed_reid)
        outs.append(f)
    return np.concatenate(outs, 0)


def _entries_for_variant(sess, in_name, out_name, scene, cams, pp, use_all_crops):
    """Re-encode every GT-matched tracklet with `pp` -> [(cam, gt_id, mean-pooled L2 vec)]."""
    entries = []
    for cam in cams:
        out = paths.cam_out(scene, cam)
        tracklets = json.loads((out / "tracklets.json").read_text())
        tid2gt = evaluate_reid.inherit_gt_ids(scene, cam)

        imgs, owner, keep = [], [], []
        for i, t in enumerate(tracklets):
            gid = tid2gt.get(t["track_id"])
            if gid is None:                                 # GT only labels multi-camera vehicles
                continue
            refs = t["crop_refs"] if use_all_crops else t["crop_refs"][:1]
            for k in refs:
                im = cv2.imread(str(paths.OUTPUT_ROOT / k))
                if im is not None:
                    imgs.append(pp(im))
                    owner.append(i)
            keep.append((i, gid))
        if not imgs:
            continue

        feats = _run_batched(sess, in_name, out_name, imgs)
        sums = defaultdict(lambda: np.zeros(REID_APPEARANCE_DIM, dtype=np.float32))
        cnt = defaultdict(int)
        for f, oi in zip(feats, owner):
            sums[oi] += f
            cnt[oi] += 1
        for i, gid in keep:
            if cnt[i] == 0:
                continue
            v = sums[i] / cnt[i]
            entries.append((cam, gid, v / (np.linalg.norm(v) + 1e-12)))
    return entries


def _cmc_map(entries):
    """Cross-camera rank-1/5 + mAP (same query rule as evaluate_reid.rank_cmc)."""
    if not entries:
        return {"queries": 0, "rank1": None, "rank5": None, "mAP": None}
    vecs = np.stack([e[2] for e in entries])
    cams = np.array([e[0] for e in entries])
    gids = np.array([e[1] for e in entries])
    sims = vecs @ vecs.T

    r1 = r5 = valid = 0
    aps = []
    for i in range(len(entries)):
        other = cams != cams[i]
        correct = other & (gids == gids[i])
        if not correct.any():                                # no cross-cam ground truth -> skip
            continue
        valid += 1
        order = np.argsort(-sims[i][other])
        match = (gids[other][order] == gids[i])
        if match[0]:
            r1 += 1
        if match[:5].any():
            r5 += 1
        hit_ranks = np.where(match)[0]
        aps.append(float(np.mean([(j + 1) / (rk + 1) for j, rk in enumerate(hit_ranks)])))
    return {
        "queries": valid,
        "rank1": round(r1 / valid, 4) if valid else None,
        "rank5": round(r5 / valid, 4) if valid else None,
        "mAP": round(float(np.mean(aps)), 4) if aps else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="S01")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--all-crops", action="store_true",
                    help="use all K crops per tracklet (matches stored ~47.1%%); default = best crop only")
    args = ap.parse_args()
    cams = args.cams or paths.list_cams(args.scene)

    if not MODEL_PATH.exists():
        print(f"missing {MODEL_PATH} — nothing to probe")
        return 1

    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    print(f"scene={args.scene} cams={cams} crops={'all-K' if args.all_crops else 'best-1'}")
    print(f"{'variant':>13} {'queries':>8} {'rank1':>7} {'rank5':>7} {'mAP':>7}")
    results = {}
    for name in args.variants:
        entries = _entries_for_variant(sess, in_name, out_name, args.scene, cams,
                                        VARIANTS[name], args.all_crops)
        m = _cmc_map(entries)
        results[name] = m
        print(f"{name:>13} {m['queries']:>8} "
              f"{(m['rank1'] or 0):>7.4f} {(m['rank5'] or 0):>7.4f} {(m['mAP'] or 0):>7.4f}", flush=True)

    base = results.get("baseline", {}).get("rank1")
    if base is not None:
        print("\ninterpretation:")
        en = results.get("ext_norm", {}).get("rank1")
        if en is not None:
            if en > base + 0.02:
                print(f"  ext_norm ({en}) > baseline ({base}) -> normalization is NOT baked in; "
                      "embed_reid._preprocess should normalize == PREPROCESSING BUG.")
            else:
                print(f"  ext_norm ({en}) <= baseline ({base}) -> normalization IS baked in; "
                      "current raw-0-255 preprocessing is correct.")
        best = max((v["rank1"] or 0, k) for k, v in results.items())
        if best[1] != "baseline" and best[0] > base + 0.02:
            print(f"  best variant is '{best[1]}' ({best[0]}) > baseline ({base}) -> switch _preprocess to it.")
        else:
            print(f"  baseline is best/among-best -> the 47% is the VeRi->CityFlow DOMAIN GAP, not a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
