"""Stages 1-4: DECODE (full fps) → DETECT (YOLO11m) → TRACK (BoT-SORT) → CROP (K best).

Runs on GPU. Persists, per camera:
  - detections.npy : every per-frame box (for the evaluator's IoU→GT match, full fps)
  - crops/         : the K=3 best thumbnails per tracklet (area × sharpness)
  - tracklets.json : per-tracklet metadata (subtype vote, time, crop files, ...)

Call gpu_setup.ensure_gpu_libs() BEFORE importing this module.
"""

from __future__ import annotations

import heapq
import json

import cv2
import numpy as np
from ultralytics import YOLO

from . import paths
from .cfg import (
    K_CROPS,
    YOLO_CLASSES,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL,
    coco_subtype,
)


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
        item = (quality, self._seq, crop_bgr.copy())
        if len(self.heap) < K_CROPS:
            heapq.heappush(self.heap, item)
        elif quality > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def best_crops(self) -> list[np.ndarray]:
        # heap is a min-heap on quality; return best-first
        return [c for _, _, c in sorted(self.heap, key=lambda t: t[0], reverse=True)]


def run(scene: str, cam: str, max_frames: int | None = None, device: int = 0) -> dict:
    video = paths.cam_video(scene, cam)
    out = paths.cam_out(scene, cam)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # fps from the file (fallback 10; c015 in S03 is 8)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or paths.DEFAULT_FPS
    cap.release()
    if not fps or fps <= 0:
        fps = paths.DEFAULT_FPS
    offset = paths.load_cam_offsets(scene)[cam]

    model = YOLO(YOLO_MODEL)
    results = model.track(
        source=str(video),
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        classes=YOLO_CLASSES,
        half=True,
        device=device,
        vid_stride=1,          # FULL fps — never subsample before tracking
        verbose=False,
    )

    tracks: dict[int, _TrackAcc] = {}
    detections: list[tuple] = []  # frame,tid,x1,y1,x2,y2,conf,cls
    frame_idx = 0  # 1-based to match MOTChallenge GT

    for r in results:
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue
        img = r.orig_img
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        H, W = img.shape[:2]
        for (x1, y1, x2, y2), tid, conf, cls in zip(xyxy, ids, confs, clses):
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(W, int(x2)), min(H, int(y2))
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            detections.append((frame_idx, tid, x1, y1, x2, y2, float(conf), cls))
            crop = img[yi1:yi2, xi1:xi2]
            acc = tracks.get(tid)
            if acc is None:
                acc = tracks[tid] = _TrackAcc(tid)
            acc.add(frame_idx, float(conf), int(cls), crop)

    # persist detections (for the evaluator)
    det_arr = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 8), np.float32)
    np.save(out / "detections.npy", det_arr)

    # build tracklet metadata + write crops
    tracklets = []
    for tid, acc in sorted(tracks.items()):
        num = len(acc.frames)
        subtype, entity_type = coco_subtype(acc.clses)
        f_start, f_end = min(acc.frames), max(acc.frames)
        ts_start = offset + f_start / fps
        ts_end = offset + f_end / fps
        crop_files = []
        for k, crop in enumerate(acc.best_crops()):
            fn = f"t{tid}_{k}.jpg"
            cv2.imwrite(str(crops_dir / fn), crop)
            crop_files.append(paths.rel_key(crops_dir / fn))
        tracklets.append(
            {
                "tracklet_id": f"{scene}_{cam}_t{tid}",
                "scene": scene,
                "camera_id": cam,
                "track_id": int(tid),
                "entity_type": entity_type,
                "subtype": subtype,
                "num_detections": num,
                "avg_conf": float(np.mean(acc.confs)),
                "frame_start": int(f_start),
                "frame_end": int(f_end),
                "ts_start_s": float(ts_start),
                "ts_end_s": float(ts_end),
                "wall_start": paths.wall_time(ts_start).isoformat(),
                "wall_end": paths.wall_time(ts_end).isoformat(),
                "crop_refs": crop_files,
            }
        )

    (out / "tracklets.json").write_text(json.dumps(tracklets, indent=2))
    del model
    return {"cam": cam, "frames": frame_idx, "tracks": len(tracks), "detections": len(detections)}
