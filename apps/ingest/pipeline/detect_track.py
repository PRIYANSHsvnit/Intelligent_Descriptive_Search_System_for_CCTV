"""Stages 1-4: DECODE (full fps) → DETECT (YOLO11m) → TRACK (BoT-SORT) → CROP (K best).

Runs on GPU. ONE decode pass: each frame is read once and handed to both detectors
(vehicle + optional person), each keeping its own persistent BoT-SORT state / id
namespace. Persists, per camera:
  - detections.npy : every per-frame box (for the evaluator's IoU→GT match, full fps,
                     INCLUDING boxes of tracks later dropped as noise)
  - crops/         : the K=3 best thumbnails per tracklet (area × sharpness)
  - tracklets.json : per-tracklet metadata (subtype vote, time, crop files, ...) —
                     tracks under MIN_TRACK_DETECTIONS are dropped HERE, so no
                     downstream stage (SigLIP/VLM/plate/...) ever pays for noise

Call gpu_setup.ensure_gpu_libs() BEFORE importing this module.
"""

from __future__ import annotations

import heapq
import json

import cv2
import numpy as np
from ultralytics import YOLO

from . import paths, profiles
from .cfg import K_CROPS


def _laplacian_var(crop_bgr: np.ndarray) -> float:
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class _TrackAcc:
    """Accumulates one track's detections + a min-heap of its K best crops."""

    __slots__ = ("tid", "frames", "confs", "clses", "heap", "_seq")

    def __init__(self, tid: int):
        self.tid = tid
        self.frames: list[int] = []
        self.confs: list[float] = []
        self.clses: list[int] = []
        self.heap: list[tuple[float, int, np.ndarray]] = []  # (quality, seq, crop)
        self._seq = 0

    def add(self, frame: int, conf: float, cls: int, crop_bgr: np.ndarray) -> None:
        self.frames.append(frame)
        self.confs.append(conf)
        self.clses.append(cls)
        h, w = crop_bgr.shape[:2]
        quality = (h * w) * _laplacian_var(crop_bgr)
        self._seq += 1
        # .copy() only on actual insert (crop_bgr is a view into the frame — storing it
        # uncopied would pin whole frames in memory); most boxes lose and need no copy.
        if len(self.heap) < K_CROPS:
            heapq.heappush(self.heap, (quality, self._seq, crop_bgr.copy()))
        elif quality > self.heap[0][0]:
            heapq.heapreplace(self.heap, (quality, self._seq, crop_bgr.copy()))

    def best_crops(self) -> list[np.ndarray]:
        # heap is a min-heap on quality; return best-first
        return [c for _, _, c in sorted(self.heap, key=lambda t: t[0], reverse=True)]


# Sentinel class id for person dets in detections.npy — COCO 'person' is id 0, which
# would collide with a vehicle class id 0 (india: hatchback) in the flat array.
PERSON_CLS = 100

# Tracks with fewer detections are detector/tracker flicker: store.py and the matcher
# already refuse them, so writing their crops/rows only feeds dead work to every
# downstream stage. detections.npy is NOT filtered (the evaluator needs every box).
MIN_TRACK_DETECTIONS = 2


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _explained_by_person(vbox, person_boxes, iou_thr=0.5, area_band=(0.4, 1.9)) -> bool:
    """True if some person box has the same footprint as this vehicle box — i.e. the
    'vehicle' is really a standing/walking pedestrian. A genuine rider's bike box is
    ~2x+ the rider-person box area, so it stays above the band and is kept."""
    va = max(1.0, (vbox[2] - vbox[0]) * (vbox[3] - vbox[1]))
    for pb in person_boxes:
        if _iou(vbox, pb) < iou_thr:
            continue
        pa = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
        if area_band[0] <= va / pa <= area_band[1]:
            return True
    return False


def _accumulate(tracks, detections, frame_idx, tid, box, conf, cls, img, W, H) -> None:
    """Clamp a box to the frame, record the raw detection, and add its crop to the track."""
    x1, y1, x2, y2 = box
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(W, int(x2)), min(H, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return
    detections.append((frame_idx, tid, x1, y1, x2, y2, float(conf), cls))
    acc = tracks.get(tid)
    if acc is None:
        acc = tracks[tid] = _TrackAcc(tid)
    acc.add(frame_idx, float(conf), int(cls), img[yi1:yi2, xi1:xi2])


def run(scene: str, cam: str, max_frames: int | None = None, device: int = 0) -> dict:
    video = paths.cam_video(scene, cam)
    out = paths.cam_out(scene, cam)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    offset = paths.load_cam_offsets(scene)[cam]
    prof = profiles.active()
    pdet = prof.person_detector

    # Optional 2nd detector for 'person' entities (india: UVH-26 has no person class);
    # each model keeps its own persistent tracker / id namespace ('t' vehicles, 'p' persons).
    veh_model = YOLO(prof.yolo_model)
    per_model = YOLO(pdet.weights) if pdet is not None else None
    common = dict(persist=True, tracker=prof.tracker, half=True, device=device, verbose=False)

    # ONE decode pass (full fps): read each frame once and hand the SAME image to both
    # trackers. (Two zipped .track(source=...) streams each decoded the video themselves —
    # every frame decompressed twice; verified bit-identical detections after the switch.)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or paths.DEFAULT_FPS  # fallback 10; c015 in S03 is 8
    if not fps or fps <= 0:
        fps = paths.DEFAULT_FPS

    veh_tracks: dict[int, _TrackAcc] = {}
    person_tracks: dict[int, _TrackAcc] = {}
    detections: list[tuple] = []  # frame,tid,x1,y1,x2,y2,conf,cls
    frame_idx = 0  # 1-based to match MOTChallenge GT

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, img = cap.read()
        if not ok:
            break
        frame_idx += 1
        H, W = img.shape[:2]

        veh_r = veh_model.track(
            img, imgsz=prof.yolo_imgsz, conf=prof.yolo_conf,
            classes=list(prof.yolo_classes) if prof.yolo_classes else None, **common)[0]
        per_r = None
        if per_model is not None:
            per_r = per_model.track(img, imgsz=pdet.imgsz, conf=pdet.conf,
                                    classes=[pdet.class_id], **common)[0]

        # person pass: accumulate person tracks + collect boxes for FP suppression
        person_boxes: list[tuple] = []
        if per_r is not None and per_r.boxes is not None and per_r.boxes.id is not None:
            pxyxy = per_r.boxes.xyxy.cpu().numpy()
            pids = per_r.boxes.id.cpu().numpy().astype(int)
            pconfs = per_r.boxes.conf.cpu().numpy()
            for box, pid, conf in zip(pxyxy, pids, pconfs):
                person_boxes.append(tuple(box))
                _accumulate(person_tracks, detections, frame_idx, int(pid), box, conf, PERSON_CLS, img, W, H)

        # vehicle pass: drop person-confusable boxes (2-wheeler/bicycle) that a same-sized
        # person box explains — the principled version of pedestrian-FP suppression
        vb = veh_r.boxes
        if vb is not None and vb.id is not None:
            vxyxy = vb.xyxy.cpu().numpy()
            vids = vb.id.cpu().numpy().astype(int)
            vconfs = vb.conf.cpu().numpy()
            vclses = vb.cls.cpu().numpy().astype(int)
            for box, tid, conf, cls in zip(vxyxy, vids, vconfs, vclses):
                if (person_boxes and cls in pdet.suppress_vehicle_classes
                        and _explained_by_person(tuple(box), person_boxes)):
                    continue
                _accumulate(veh_tracks, detections, frame_idx, int(tid), box, conf, cls, img, W, H)

    cap.release()

    # persist detections (for the evaluator) — full, noise tracks included
    det_arr = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 8), np.float32)
    np.save(out / "detections.npy", det_arr)

    def build(tracks, prefix, resolve):
        """Turn a track-accumulator dict into tracklet metadata + write its crops.
        prefix namespaces ids/crops/tracklet_ids ('t' vehicles, 'p' persons).
        Noise tracks (< MIN_TRACK_DETECTIONS) are skipped: no crops, no row."""
        rows = []
        for tid, acc in sorted(tracks.items()):
            if len(acc.frames) < MIN_TRACK_DETECTIONS:
                continue
            subtype, entity_type = resolve(acc)
            f_start, f_end = min(acc.frames), max(acc.frames)
            ts_start, ts_end = offset + f_start / fps, offset + f_end / fps
            crop_files = []
            for k, crop in enumerate(acc.best_crops()):
                fn = f"{prefix}{tid}_{k}.jpg"
                cv2.imwrite(str(crops_dir / fn), crop)
                crop_files.append(paths.rel_key(crops_dir / fn))
            rows.append({
                "tracklet_id": f"{scene}_{cam}_{prefix}{tid}",
                "scene": scene,
                "camera_id": cam,
                "track_id": int(tid),
                "entity_type": entity_type,
                "subtype": subtype,
                "num_detections": len(acc.frames),
                "avg_conf": float(np.mean(acc.confs)),
                "frame_start": int(f_start),
                "frame_end": int(f_end),
                "ts_start_s": float(ts_start),
                "ts_end_s": float(ts_end),
                "wall_start": paths.wall_time(ts_start).isoformat(),
                "wall_end": paths.wall_time(ts_end).isoformat(),
                "crop_refs": crop_files,
            })
        return rows

    tracklets = (build(veh_tracks, "t", lambda acc: profiles.subtype_vote(acc.clses))
                 + build(person_tracks, "p", lambda acc: ("person", "person")))
    (out / "tracklets.json").write_text(json.dumps(tracklets, indent=2))

    del veh_model
    if per_model is not None:
        del per_model
    n_veh = sum(1 for t in tracklets if t["entity_type"] != "person")
    return {"cam": cam, "frames": frame_idx, "tracks": n_veh,
            "persons": len(tracklets) - n_veh,
            "dropped_noise": len(veh_tracks) + len(person_tracks) - len(tracklets),
            "detections": len(detections)}
