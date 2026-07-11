"""veri_rank1_probe.py — the REAL VeRi-776 rank-1/mAP for our ONNX encoder.

Settles "is the 47% a preprocessing bug or the domain gap?" the direct way: score
models/veri_reid.onnx on VeRi's OWN test set, through the EXACT pipeline preprocessing
(pipeline/embed_reid._preprocess). The published number for this checkpoint is
97.0% rank-1 / 81.9% mAP. So:
  * ~97% here  -> encoder + our preprocessing are correct; the 47% on S01 is purely the
                 VeRi->CityFlow domain gap (confirms reid_preproc_probe.py's verdict).
  * <<97% here -> our ONNX/preprocessing path is dropping accuracy; fix _preprocess.

Canonical VeRi protocol (uses the official index files, no ID parsing needed):
  name_query.txt / name_test.txt  : ordered query / gallery file lists
  gt_index.txt line i             : 1-indexed gallery positions that are TRUE matches
                                    (same vehicle, DIFFERENT camera) for query i
  jk_index.txt line i             : 1-indexed "junk" positions (same vehicle, SAME
                                    camera, incl. the query image itself) — removed
                                    from the ranking before scoring.

  uv run python veri_rank1_probe.py                         # full test set (~13k imgs, CPU)
  uv run python veri_rank1_probe.py --max-queries 300       # quick subset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline import paths
from pipeline.embed_reid import MODEL_PATH, _preprocess
from pipeline.cfg import REID_APPEARANCE_DIM

VERI_ROOT = paths.REPO_ROOT / "VeRi"
_BATCH = 64


def _read_names(p) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _read_index(p) -> list[np.ndarray]:
    """Each line -> 0-indexed int array of gallery positions (files are 1-indexed)."""
    rows = []
    for ln in p.read_text().splitlines():
        toks = ln.split()
        rows.append(np.array([int(t) - 1 for t in toks], dtype=np.int64) if toks
                    else np.empty(0, dtype=np.int64))
    return rows


def _embed(sess, in_name, out_name, img_dir, names) -> np.ndarray:
    """Embed images (in list order) through the pipeline's exact _preprocess. L2-normed."""
    feats = np.zeros((len(names), REID_APPEARANCE_DIM), dtype=np.float32)
    buf, rows = [], []
    t0 = time.time()

    def flush():
        if not buf:
            return
        arr = np.stack(buf).astype(np.float32)
        f = sess.run([out_name], {in_name: arr})[0]
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
        for r, fr in zip(rows, f):
            feats[r] = fr
        buf.clear()
        rows.clear()

    for i, name in enumerate(names):
        im = cv2.imread(str(img_dir / name))
        if im is None:
            continue
        buf.append(_preprocess(im))
        rows.append(i)
        if len(buf) >= _BATCH:
            flush()
        if i and i % 2000 == 0:
            print(f"    embedded {i}/{len(names)}  ({time.time()-t0:.0f}s)", flush=True)
    flush()
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri-root", default=str(VERI_ROOT))
    ap.add_argument("--max-queries", type=int, default=None, help="score only the first N queries (quick check)")
    args = ap.parse_args()
    root = Path(args.veri_root)

    if not MODEL_PATH.exists():
        print(f"missing {MODEL_PATH}")
        return 1
    for req in ("name_query.txt", "name_test.txt", "gt_index.txt", "jk_index.txt"):
        if not (root / req).exists():
            print(f"missing {root/req} — is --veri-root correct?")
            return 1

    q_names = _read_names(root / "name_query.txt")
    g_names = _read_names(root / "name_test.txt")
    gt = _read_index(root / "gt_index.txt")
    jk = _read_index(root / "jk_index.txt")
    print(f"VeRi: {len(q_names)} queries, {len(g_names)} gallery  (root={root})")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    print("  embedding gallery ...", flush=True)
    g_feat = _embed(sess, in_name, out_name, root / "image_test", g_names)
    print("  embedding queries ...", flush=True)
    q_feat = _embed(sess, in_name, out_name, root / "image_query", q_names)

    nq = len(q_names) if args.max_queries is None else min(args.max_queries, len(q_names))
    r1 = r5 = 0
    aps = []
    valid = 0
    for i in range(nq):
        good = gt[i]
        if good.size == 0:
            continue
        sims = q_feat[i] @ g_feat.T                  # (G,)
        order = np.argsort(-sims)                     # best-first gallery indices

        junk = np.zeros(len(g_names), dtype=bool)
        junk[jk[i]] = True
        keep = order[~junk[order]]                    # drop junk (same-cam same-id)

        is_good = np.isin(keep, good)
        if not is_good.any():
            continue
        valid += 1
        if is_good[0]:
            r1 += 1
        if is_good[:5].any():
            r5 += 1
        # average precision (Market/VeRi convention)
        hit_ranks = np.where(is_good)[0]
        aps.append(float(np.mean([(j + 1) / (rk + 1) for j, rk in enumerate(hit_ranks)])))

    print(f"\nscored {valid} queries  (preprocess = pipeline embed_reid._preprocess)")
    print(f"  rank-1 = {r1/valid:.4f}")
    print(f"  rank-5 = {r5/valid:.4f}")
    print(f"  mAP    = {np.mean(aps):.4f}")
    print("\n  published for veri_sbs_R50-ibn: rank-1 0.970 / mAP 0.819")
    if r1 / valid >= 0.90:
        print("  => encoder + _preprocess CORRECT; the 47% on S01 is the VeRi->CityFlow domain gap.")
    else:
        print("  => WELL below published -> our ONNX/preprocessing path is dropping accuracy; fix _preprocess.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
