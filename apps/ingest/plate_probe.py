"""Sanity tool (throwaway): plate-OCR feasibility probe for SUR01.

Runs the open-image-models YOLOv9 license-plate detector over the pipeline's
existing K-best vehicle crops and reports, per camera and overall:

  - what fraction of vehicle tracklets have a detectable plate at all
  - the distribution of best plate height (px) per tracklet — the gate for OCR
    (rule of thumb: <16 px tall is unreadable by any OCR; two-row Indian
    bike/auto plates need ~2x that since each row gets half the pixels)

Also dumps sample plate crops (4x upscaled) per height bucket plus the vehicle
crop with the plate box drawn, under sanity_out/plate_probe/<bucket>/, so the
"is this readable?" question can be eyeballed.

No torch — onnxruntime CPU, no gpu_setup shim needed (same as the re-ID pass).

Usage (from apps/ingest):
    uv run python plate_probe.py                       # all SUR01 cams
    uv run python plate_probe.py --cams c001 c004
    uv run python plate_probe.py --samples-per-bucket 12
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from pipeline import paths

# best-plate-height buckets (px); labels sort lexicographically in report order
BUCKETS = [
    (0, 12, "a_lt12px_hopeless"),
    (12, 16, "b_12-16px_marginal"),
    (16, 20, "c_16-20px_hard"),
    (20, 28, "d_20-28px_ok"),
    (28, 40, "e_28-40px_good"),
    (40, 10**9, "f_40px+_easy"),
]

OUT_DIR = paths.INGEST_ROOT / "sanity_out" / "plate_probe"


def bucket_of(h: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= h < hi:
            return name
    return BUCKETS[-1][2]


def probe_cam(detector, scene: str, cam: str, batch: int = 64):
    """Returns per-tracklet best-plate records for one camera."""
    tj = paths.cam_out(scene, cam) / "tracklets.json"
    tracklets = json.loads(tj.read_text())
    vehicles = [t for t in tracklets if t.get("entity_type", "vehicle") == "vehicle"]

    # flat list of (tracklet_idx, crop_path), then batch through the detector
    work = [(i, paths.OUTPUT_ROOT / ref)
            for i, t in enumerate(vehicles) for ref in t["crop_refs"]
            if (paths.OUTPUT_ROOT / ref).exists()]

    best: dict[int, dict] = {}  # tracklet_idx -> best plate record
    t0 = time.time()
    for b in range(0, len(work), batch):
        chunk = work[b:b + batch]
        results = detector.predict([str(p) for _, p in chunk])
        if chunk and not isinstance(results[0], list):  # single-image API quirk
            results = [results]
        for (ti, crop_path), dets in zip(chunk, results):
            for d in dets:
                bb = d.bounding_box
                h = bb.y2 - bb.y1
                cur = best.get(ti)
                if cur is None or h > cur["plate_h"]:
                    best[ti] = {
                        "plate_h": int(h),
                        "plate_w": int(bb.x2 - bb.x1),
                        "conf": round(float(d.confidence), 3),
                        "crop_path": str(crop_path),
                        "box": [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)],
                    }
        done = min(b + batch, len(work))
        print(f"  {cam}: {done}/{len(work)} crops "
              f"({done / max(time.time() - t0, 1e-9):.0f}/s)", flush=True)

    records = []
    for i, t in enumerate(vehicles):
        r = best.get(i)
        records.append({
            "tracklet_id": t["tracklet_id"],
            "subtype": t["subtype"],
            "cam": cam,
            **(r or {}),
        })
    return records


def save_samples(records: list[dict], per_bucket: int) -> None:
    """Dump upscaled plate crops + annotated vehicle crops, spread per bucket."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if "plate_h" in r:
            by_bucket[bucket_of(r["plate_h"])].append(r)
    for name, rs in by_bucket.items():
        d = OUT_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        # spread evenly instead of taking the first N (avoids one-cam bias)
        step = max(len(rs) // per_bucket, 1)
        for r in rs[::step][:per_bucket]:
            img = cv2.imread(r["crop_path"])
            if img is None:
                continue
            x1, y1, x2, y2 = r["box"]
            plate = img[max(y1, 0):y2, max(x1, 0):x2]
            if plate.size == 0:
                continue
            stem = f"{r['tracklet_id']}_h{r['plate_h']}"
            up = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(d / f"{stem}_plate4x.jpg"), up)
            ann = img.copy()
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imwrite(str(d / f"{stem}_vehicle.jpg"), ann)


def report(records: list[dict]) -> None:
    total = len(records)
    hit = [r for r in records if "plate_h" in r]
    print(f"\n=== plate probe: {total} vehicle tracklets, "
          f"{len(hit)} ({100 * len(hit) / max(total, 1):.0f}%) with a detected plate ===")

    hist = Counter(bucket_of(r["plate_h"]) for r in hit)
    print("\nbest plate height per tracklet (px):")
    for _, _, name in BUCKETS:
        n = hist.get(name, 0)
        bar = "#" * round(50 * n / max(len(hit), 1))
        print(f"  {name:<22} {n:>5}  {bar}")

    print("\nper camera (tracklets with plate / total, median best height):")
    for cam in sorted({r["cam"] for r in records}):
        rs = [r for r in records if r["cam"] == cam]
        hs = sorted(r["plate_h"] for r in rs if "plate_h" in r)
        med = hs[len(hs) // 2] if hs else 0
        print(f"  {cam}: {len(hs):>4}/{len(rs):<4}  median {med}px")

    print("\nby subtype (with plate / total, median best height):")
    for st in sorted({r["subtype"] for r in records}):
        rs = [r for r in records if r["subtype"] == st]
        hs = sorted(r["plate_h"] for r in rs if "plate_h" in r)
        med = hs[len(hs) // 2] if hs else 0
        print(f"  {st:<16} {len(hs):>4}/{len(rs):<4}  median {med}px")

    # the headline number: tracklets whose best plate clears the readability bar
    readable = sum(1 for r in hit if r["plate_h"] >= 20)
    maybe = sum(1 for r in hit if 16 <= r["plate_h"] < 20)
    print(f"\nOCR-candidate tracklets (best plate >=20px): {readable} "
          f"({100 * readable / max(total, 1):.0f}% of all vehicles); "
          f"borderline 16-20px: {maybe}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--model", default="yolo-v9-s-608-license-plate-end2end")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--samples-per-bucket", type=int, default=8)
    args = ap.parse_args()

    from open_image_models import LicensePlateDetector
    detector = LicensePlateDetector(detection_model=args.model, conf_thresh=args.conf)

    scene_out = paths.OUTPUT_ROOT / args.scene
    cams = args.cams or sorted(p.name for p in scene_out.iterdir()
                               if (p / "tracklets.json").exists())
    print(f"probing {args.scene} cams={cams} model={args.model}")

    records: list[dict] = []
    for cam in cams:
        records.extend(probe_cam(detector, args.scene, cam))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plate_probe_results.json").write_text(json.dumps(records, indent=1))
    save_samples(records, args.samples_per_bucket)
    report(records)
    print(f"\nsamples + results under {OUT_DIR}")


if __name__ == "__main__":
    main()
