"""Sanity tool (throwaway): A/B fast-plate-ocr vs RapidOCR on the plate-probe output.

Reads sanity_out/plate_probe/plate_probe_results.json (from plate_probe.py), re-crops
each tracklet's best plate, runs both OCR engines, normalizes reads against the Indian
plate format, and reports per-engine valid-read rates plus cross-engine agreement.
Two engines agreeing on a 9-10 char registration by accident is ~impossible, so the
agreement set doubles as a pseudo-ground-truth eval set (verify a sample by eye).

Usage (from apps/ingest):  uv run python plate_ab.py [--min-h 14]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from pipeline import paths

PROBE_DIR = paths.INGEST_ROOT / "sanity_out" / "plate_probe"

# LL DD L{1,3} DDDD (current series) — GJ05AB1234; older/odd series get 'loose'
PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
LOOSE_RE = re.compile(r"^[A-Z0-9]{8,10}$")


def normalize(raw: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    # state-code prior: Surat plates are GJ; fix a single bad char in the prefix
    if len(s) >= 2 and s[:2] != "GJ" and (s[0] in "GC0O6U" and s[1] in "JI1U"):
        s = "GJ" + s[2:]
    return s


def grade(s: str) -> str:
    if PLATE_RE.match(s):
        return "valid"
    if LOOSE_RE.match(s):
        return "loose"
    return "bad"


def crop_plate(rec: dict, margin: float = 0.08, min_up_h: int = 40) -> np.ndarray | None:
    img = cv2.imread(rec["crop_path"])
    if img is None:
        return None
    x1, y1, x2, y2 = rec["box"]
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    h, w = img.shape[:2]
    p = img[max(y1 - my, 0):min(y2 + my, h), max(x1 - mx, 0):min(x2 + mx, w)]
    if p.size == 0:
        return None
    if p.shape[0] < min_up_h:  # upscale small plates so both engines get enough pixels
        f = min_up_h / p.shape[0]
        p = cv2.resize(p, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return p


def read_fpo(recognizer, plate: np.ndarray) -> tuple[str, float]:
    """fast-plate-ocr; two-row plates (squarish box) also tried as split halves."""
    def one(img: np.ndarray) -> tuple[str, float]:
        p = recognizer.run(img, return_confidence=True)[0]
        n = len(p.plate)
        conf = float(np.min(p.char_probs[:n])) if n else 0.0
        return p.plate, conf

    text, conf = one(plate)
    h, w = plate.shape[:2]
    if w / max(h, 1) < 2.3:  # two-row: OCR each half, concatenate
        (t, ct), (b, cb) = one(plate[: int(h * 0.55)]), one(plate[int(h * 0.45):])
        if t and b and min(ct, cb) > conf:
            text, conf = t + b, min(ct, cb)
    return normalize(text), round(conf, 3)


def read_rapid(ocr, plate: np.ndarray) -> tuple[str, float]:
    """RapidOCR det+rec; concatenate all detected lines top-to-bottom."""
    res, _ = ocr(plate)
    if not res:
        return "", 0.0
    res = sorted(res, key=lambda r: min(pt[1] for pt in r[0]))  # top-to-bottom
    text = "".join(t for _, t, _ in res)
    conf = min(float(c) for _, _, c in res)
    return normalize(text), round(conf, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-h", type=int, default=14)
    args = ap.parse_args()

    records = json.loads((PROBE_DIR / "plate_probe_results.json").read_text())
    work = [r for r in records if r.get("plate_h", 0) >= args.min_h]
    print(f"{len(work)} tracklets with plate >= {args.min_h}px "
          f"(of {len(records)} vehicle tracklets)")

    from fast_plate_ocr import LicensePlateRecognizer
    from rapidocr_onnxruntime import RapidOCR
    fpo = LicensePlateRecognizer("cct-s-v2-global-model")
    rapid = RapidOCR()

    rows = []
    for i, rec in enumerate(work):
        plate = crop_plate(rec)
        if plate is None:
            continue
        ft, fc = read_fpo(fpo, plate)
        rt, rc = read_rapid(rapid, plate)
        agree = bool(ft) and ft == rt
        rows.append({
            "tracklet_id": rec["tracklet_id"], "cam": rec["cam"],
            "subtype": rec["subtype"], "plate_h": rec["plate_h"],
            "fpo": ft, "fpo_conf": fc, "fpo_grade": grade(ft),
            "rapid": rt, "rapid_conf": rc, "rapid_grade": grade(rt),
            "agree": agree, "crop_path": rec["crop_path"],
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(work)}", flush=True)

    out_csv = PROBE_DIR / "plate_ab.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    fpo_valid = sum(r["fpo_grade"] == "valid" for r in rows)
    rap_valid = sum(r["rapid_grade"] == "valid" for r in rows)
    agree = [r for r in rows if r["agree"]]
    agree_valid = [r for r in agree if r["fpo_grade"] == "valid"]
    print(f"\n=== A/B over {n} plates ===")
    print(f"fast-plate-ocr valid-format reads : {fpo_valid} ({100 * fpo_valid / n:.0f}%)")
    print(f"rapidocr       valid-format reads : {rap_valid} ({100 * rap_valid / n:.0f}%)")
    print(f"exact agreement (any format)      : {len(agree)}")
    print(f"exact agreement + valid format    : {len(agree_valid)}  <- pseudo-GT eval set")
    for r in agree_valid[:25]:
        print(f"   {r['cam']} {r['tracklet_id']:<18} {r['fpo']:<11} h={r['plate_h']}px")
    print(f"\nfull table: {out_csv}")


if __name__ == "__main__":
    main()
