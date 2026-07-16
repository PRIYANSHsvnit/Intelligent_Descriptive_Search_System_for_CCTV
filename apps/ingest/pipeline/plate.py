"""Stage (plate OCR): license-plate detection + reading for VEHICLE tracklets.

Two-tier output per tracklet (probe + 50-plate hand-labeled eval, see the
plate-ocr-sur01 memory / plan.md):

  plate_text  canonical validated registration — the ASSERTION tier. Only set when
              a read survives the full constraint stack below; NULL = abstain.
  plate_raw   best raw OCR read as produced (e.g. 'JO5AY1139') — the RECALL tier.
              Never displayed as fact; searched fuzzily (pg_trgm / substring) so a
              read that failed validation still makes the vehicle findable.
  plate_conf  confidence for plate_text (agreement-weighted).

Pipeline per tracklet: YOLOv9 plate detector (open-image-models, MIT) on each of the
K crops → largest plate box per crop (ignores neighbours' plates bleeding into the
box) → upscale → RapidOCR (PP-OCR/ONNX; reads two-row Indian plates as separate
lines, joined top-to-bottom) → normalize → bounded template repair → gates.

Constraint stack for plate_text (each read):
  1. charset normalize (A-Z0-9 only)
  2. template repair against `SS D(1-2) L(1-3) DDDD`: positional confusion map
     (O<->0, I/L<->1, S<->5, B<->8, Z<->2, G<->6), at most ONE substitution, plus an
     optional leading-G restore (RapidOCR drops the edge 'G' of GJ on tight crops);
     any repair flags the read — a repaired read can never be accepted alone
  3. state code must be a real RTO code (kills QL/SJ/JO garble)
  4. length >= 9 post-repair (9-char single-series plates are legal, 8 is not)
  5. OCR line-confidence >= 0.70 (garbage starts below ~0.7 on the eval set)
Acceptance: two reads agreeing exactly post-repair (different crops), OR a single
UNREPAIRED read with conf >= 0.85. Bharat-series (22BH1234AB) is a known abstain.

2nd-chance pass (extra-frame sampling): the K saved crops optimize VEHICLE quality,
not plate quality. Tracklets that showed a plate box (or a read) but no accepted
plate_text get revisited: their EXTRA_SCAN tallest detection boxes are pulled from
the source video in ONE sequential decode (grab/skip), and the EXTRA_OCR_K sharpest
plate crops (fixed-height Laplacian variance) are OCR'd; the merged read pool goes
through the same acceptance rule, so the precision gates are unchanged.

CPU-only (onnxruntime, same as re-ID) — no gpu_setup shim needed. ~10 crops/s;
full SUR01 (~10k vehicle crops) ≈ 20 min + the 2nd-chance pass. Run any time
after `detect` (needs detections.npy + the source video for the 2nd pass).
"""

from __future__ import annotations

import json
import re
from collections import Counter

import cv2
import numpy as np

from . import paths

DETECTOR_MODEL = "yolo-v9-s-608-license-plate-end2end"
DETECTOR_CONF = 0.25
MIN_PLATE_H = 14          # px; probe showed nothing readable below
OCR_TARGET_H = 64         # upscale plate crops to at least this height
CROP_MARGIN = 0.08
OCR_MIN_CONF = 0.70       # reads below this are discarded for plate_text
SINGLE_ACCEPT_CONF = 0.85  # lone unrepaired read needs at least this
RAW_MIN_CONF = 0.50       # plate_raw keeps lower-conf fragments for fuzzy search
RAW_MIN_LEN = 4

# Extra-frame sampling (2nd-chance pass): the K saved crops were picked for VEHICLE
# quality, not plate quality — the frame where the plate is closest & blur-free is
# usually not among them. For tracklets with plate evidence but no accepted text,
# revisit the source video via detections.npy.
EXTRA_SCAN = 10       # tallest-box frames examined per unresolved tracklet
EXTRA_OCR_K = 4       # sharpest plate crops among them that get OCR'd
MIN_FRAME_GAP = 3     # sampled frames must be this far apart (avoid near-duplicates)
SHARPNESS_H = 32      # plate crops resized to this height before Laplacian scoring
# two extra-pass reads only count as independent agreement this many frames apart —
# adjacent frames share the same motion-blur and can agree on the same misread
# (caught live: GJ19Y5038 accepted as GJ19Y5033 from two frames 0.15s apart)
INDEP_FRAME_GAP = 16

# official RTO state/UT codes (constraint 3)
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB",
}

_TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8",
             "Z": "2", "G": "6"}
_TO_LETTER = {v: k for k, v in _TO_DIGIT.items() if k not in ("Q", "L")}

_detector = None
_ocr = None


def _models():
    """Load detector + OCR once per stage pass (both ONNX/CPU)."""
    global _detector, _ocr
    if _detector is None:
        from open_image_models import LicensePlateDetector
        from rapidocr_onnxruntime import RapidOCR
        _detector = LicensePlateDetector(
            detection_model=DETECTOR_MODEL, conf_thresh=DETECTOR_CONF)
        _ocr = RapidOCR()
    return _detector, _ocr


def _normalize(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _fit_template(s: str, max_repairs: int = 1) -> str | None:
    """Repair `s` onto SS D(1-2) L(1-3) DDDD with <= max_repairs substitutions.
    Returns the canonical plate or None. State code must whitelist post-repair."""
    best: tuple[int, str] | None = None
    for dd in (2, 1):
        for ser in (2, 1, 3):
            if len(s) != 2 + dd + ser + 4:
                continue
            classes = "LL" + "D" * dd + "L" * ser + "DDDD"
            out, repairs = [], 0
            for ch, cls in zip(s, classes):
                want_digit = cls == "D"
                if ch.isdigit() == want_digit:
                    out.append(ch)
                    continue
                fixed = (_TO_DIGIT if want_digit else _TO_LETTER).get(ch)
                if fixed is None:
                    repairs = max_repairs + 1
                    break
                out.append(fixed)
                repairs += 1
            if repairs > max_repairs:
                continue
            cand = "".join(out)
            if cand[:2] not in STATE_CODES or int(cand[2:2 + dd]) == 0:
                continue
            if best is None or repairs < best[0]:
                best = (repairs, cand)
    return best[1] if best else None


def _repair(raw: str) -> tuple[str | None, bool]:
    """(canonical_plate, was_repaired). Tries as-is, then with a restored leading G
    (RapidOCR often clips the first char of 'GJ' on tight crops)."""
    canon = _fit_template(raw, max_repairs=0)
    if canon:
        return canon, False
    canon = _fit_template(raw, max_repairs=1)
    if canon:
        return canon, True
    # visually-confused G in the state code: 'CJ05..'/'SJ05..'/'UJ05..' -> 'GJ05..'
    if len(raw) >= 9 and raw[1] == "J" and raw[0] in "CSUO06Q":
        canon = _fit_template("GJ" + raw[2:], max_repairs=1)
        if canon:
            return canon, True
    if len(raw) in (8, 9) and raw[0] == "J":  # dropped edge G: 'JO5AY1139'
        canon = _fit_template("G" + raw, max_repairs=1)
        if canon:
            return canon, True
    return None, False


def _read_plate(ocr, plate_bgr: np.ndarray) -> tuple[str, float]:
    """RapidOCR det+rec; lines joined top-to-bottom (two-row plates)."""
    res, _ = ocr(plate_bgr)
    if not res:
        return "", 0.0
    res = sorted(res, key=lambda r: min(pt[1] for pt in r[0]))
    text = _normalize("".join(t for _, t, _ in res))
    conf = min(float(c) for _, _, c in res)
    return text, conf


def _plate_crop(img: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    mx, my = int((x2 - x1) * CROP_MARGIN), int((y2 - y1) * CROP_MARGIN)
    h, w = img.shape[:2]
    p = img[max(y1 - my, 0):min(y2 + my, h), max(x1 - mx, 0):min(x2 + mx, w)]
    if p.size == 0:
        return None
    if p.shape[0] < OCR_TARGET_H:
        f = OCR_TARGET_H / p.shape[0]
        p = cv2.resize(p, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return p


def _independent(a: dict, b: dict) -> bool:
    """Whether two agreeing reads are independent evidence. Saved crops (no frame)
    were picked track-wide; extra-pass frames must be INDEP_FRAME_GAP apart."""
    fa, fb = a.get("frame"), b.get("frame")
    if fa is None or fb is None:
        return True
    return abs(fa - fb) >= INDEP_FRAME_GAP


def _resolve(reads: list[dict], can_agree: bool = True,
             ) -> tuple[str | None, float | None, str | None]:
    """Acceptance rule over a tracklet's per-crop reads -> (text, conf, raw).
    can_agree=False for tracklets shorter than INDEP_FRAME_GAP frames: every
    frame shares one blur regime, so agreeing reads are not independent evidence
    and only the single high-confidence path may assert."""
    raw_pool = [r for r in reads
                if r["conf"] >= RAW_MIN_CONF and len(r["raw"]) >= RAW_MIN_LEN]
    plate_raw = max(raw_pool, key=lambda r: r["conf"])["raw"] if raw_pool else None

    valid = [r for r in reads if r["canon"] and r["conf"] >= OCR_MIN_CONF]
    if valid:
        counts = Counter(r["canon"] for r in valid)
        canon, n = counts.most_common(1)[0]
        support = [r for r in valid if r["canon"] == canon]
        agree = can_agree and n >= 2 and any(
            _independent(support[i], support[j])
            for i in range(len(support)) for j in range(i + 1, len(support)))
        if agree:  # two INDEPENDENT reads agree post-repair
            return canon, round(min(0.99, max(r["conf"] for r in support)), 3), plate_raw
        r = max(support, key=lambda r: r["conf"])
        if not r["repaired"] and r["conf"] >= SINGLE_ACCEPT_CONF:
            return canon, round(r["conf"], 3), plate_raw
    return None, None, plate_raw


def _sharpness(plate_bgr: np.ndarray) -> float:
    """Motion-blur score, scale-fair: Laplacian variance at a fixed height."""
    g = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    if g.shape[0] != SHARPNESS_H:
        f = SHARPNESS_H / g.shape[0]
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _detect_plate(detector, img: np.ndarray):
    """Largest plate box in a vehicle crop (own plate, not a neighbour's), or None."""
    dets = detector.predict(img)
    if not dets:
        return None
    box = max(dets, key=lambda d: d.bounding_box.y2 - d.bounding_box.y1).bounding_box
    return box if box.y2 - box.y1 >= MIN_PLATE_H else None


def _extra_reads(scene: str, cam: str, todo: dict[int, int],
                 detector, ocr) -> tuple[dict[int, list[dict]], int]:
    """2nd-chance pass for tracklets in `todo` (tracklet index -> track_id): sample
    the EXTRA_SCAN tallest-box frames per track from detections.npy (tallest vehicle
    box = closest to camera = biggest plate), stream the source video once, keep
    plate crops, OCR only the EXTRA_OCR_K sharpest. Returns (reads per tracklet
    index, frames decoded)."""
    det_path = paths.cam_out(scene, cam) / "detections.npy"
    video = paths.cam_video(scene, cam)
    if not todo or not det_path.exists() or not video.exists():
        return {}, 0
    det = np.load(det_path)
    if det.size == 0:
        return {}, 0

    jobs: dict[int, list[tuple[int, np.ndarray]]] = {}  # frame -> [(ti, xyxy)]
    for ti, tid in todo.items():
        rows = det[(det[:, 1] == tid) & (det[:, 7] != 100)]  # cls 100 = person rows
        rows = rows[np.argsort(-(rows[:, 5] - rows[:, 3]))]  # tallest box first
        frames: list[int] = []
        for r in rows:
            f = int(r[0])
            if len(frames) >= EXTRA_SCAN:
                break
            if all(abs(f - g) >= MIN_FRAME_GAP for g in frames):
                frames.append(f)
                jobs.setdefault(f, []).append((ti, r[2:6]))
    if not jobs:
        return {}, 0

    cand: dict[int, list[tuple[float, np.ndarray]]] = {ti: [] for ti in todo}
    cap = cv2.VideoCapture(str(video))
    last, frame_idx, decoded = max(jobs), 0, 0
    while frame_idx < last:
        frame_idx += 1  # 1-based, matching detect_track's frame numbering
        if frame_idx not in jobs:
            if not cap.grab():  # cheap skip: no decode for frames without jobs
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        decoded += 1
        H, W = img.shape[:2]
        for ti, (x1, y1, x2, y2) in jobs[frame_idx]:
            vcrop = img[max(int(y1), 0):min(int(y2), H), max(int(x1), 0):min(int(x2), W)]
            if vcrop.size == 0:
                continue
            box = _detect_plate(detector, vcrop)
            if box is None:
                continue
            region = vcrop[max(box.y1, 0):box.y2, max(box.x1, 0):box.x2]
            plate = _plate_crop(vcrop, box)
            if region.size == 0 or plate is None:
                continue
            cand[ti].append((_sharpness(region), frame_idx, plate))
    cap.release()

    out: dict[int, list[dict]] = {}
    for ti, lst in cand.items():
        lst.sort(key=lambda c: -c[0])
        reads = []
        for _sharp, frame, plate in lst[:EXTRA_OCR_K]:
            raw, conf = _read_plate(ocr, plate)
            if not raw:
                continue
            canon, repaired = _repair(raw)
            reads.append({"raw": raw, "conf": conf, "canon": canon,
                          "repaired": repaired, "frame": frame})
        if reads:
            out[ti] = reads
    return out, decoded


def run(scene: str, cam: str) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    vehicles = [(i, t) for i, t in enumerate(tracklets)
                if t.get("entity_type", "vehicle") == "vehicle"]
    if not vehicles:
        return {"cam": cam, "vehicles": 0}

    detector, ocr = _models()
    pass1: dict[int, list[dict]] = {}
    todo: dict[int, int] = {}  # tracklet index -> track_id, for the 2nd-chance pass
    for ti, t in vehicles:
        reads, had_box = [], False
        for ref in t["crop_refs"]:
            img = cv2.imread(str(paths.OUTPUT_ROOT / ref))
            if img is None:
                continue
            box = _detect_plate(detector, img)
            if box is None:
                continue
            had_box = True
            plate = _plate_crop(img, box)
            if plate is None:
                continue
            raw, conf = _read_plate(ocr, plate)
            if not raw:
                continue
            canon, repaired = _repair(raw)
            reads.append({"raw": raw, "conf": conf, "canon": canon,
                          "repaired": repaired})
        can_agree = t["frame_end"] - t["frame_start"] >= INDEP_FRAME_GAP
        text, conf, raw = _resolve(reads, can_agree)
        t["plate_text"], t["plate_conf"], t["plate_raw"] = text, conf, raw
        pass1[ti] = reads
        if text is None and had_box:  # plate evidence but no accepted read: retry
            todo[ti] = t["track_id"]

    # 2nd chance: sample sharper/closer frames from the source video
    extra, decoded = _extra_reads(scene, cam, todo, detector, ocr)
    rescued = 0
    for ti, reads in extra.items():
        t = tracklets[ti]
        can_agree = t["frame_end"] - t["frame_start"] >= INDEP_FRAME_GAP
        text, conf, raw = _resolve(pass1[ti] + reads, can_agree)
        if text is not None:
            rescued += 1
        # keep the better raw (higher conf pool wins inside _resolve already)
        t["plate_text"], t["plate_conf"] = text, conf
        t["plate_raw"] = raw or t["plate_raw"]

    n_text = sum(t["plate_text"] is not None for _, t in vehicles)
    n_raw = sum(t["plate_raw"] is not None for _, t in vehicles)
    (out / "tracklets.json").write_text(json.dumps(tracklets, indent=1))
    return {"cam": cam, "vehicles": len(vehicles), "plate_text": n_text,
            "plate_raw": n_raw, "retried": len(todo), "rescued": rescued,
            "frames_decoded": decoded}
