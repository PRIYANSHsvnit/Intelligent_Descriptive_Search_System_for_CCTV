"""LiveEngine — the streaming driver, PIPELINED.

Four stages run concurrently instead of one serial loop, so the GPU stops idling during
decode and paperwork:

  [decoder thread]  grab/retrieve, skip every `stride`-th frame, read ahead   -> frame_q
  [detect loop]     (main) both .track() + accumulate; ended tracks           -> fin_q
  [finalize worker] best-K crops -> SigLIP embed -> color -> crop-save -> INSERT -> OCR.submit
  [OCR pool]        async CPU plate read (ocr_worker)                          -> UPDATE

Co-residency is what makes this legal: the detectors (main thread) and SigLIP (finalize
thread) are already in VRAM, so they run without reload. The detect/embed GPU work still
serializes on the one card, but the decode (thread 1) and the crop-save + DB-insert IO
(thread 3) now overlap detection instead of stalling it — that's where the ~22 s of inline
"paperwork" from the serial version goes.

`--stride N` processes every Nth frame (N=2 halves the GPU load; a car doesn't move much in
1/20 s, so tracking survives). Timestamps use the TRUE frame index, and footage_secs counts
ALL decoded frames, so realtime_x stays honest under any stride.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import cv2
import numpy as np

from pipeline import paths, profiles
from pipeline.detect_track import (
    MIN_TRACK_DETECTIONS,
    PERSON_CLS,
    _TrackAcc,
    _accumulate,
    _explained_by_person,
)

TRACK_TIMEOUT_FRAMES = 15  # a track unseen this many PROCESSED frames is finalized
_SINK: list = []           # per-frame throwaway detections sink (cleared each frame)


class LiveEngine:
    def __init__(self, pool, writer, ocr=None) -> None:
        self.pool = pool
        self.writer = writer  # used ONLY by the finalize worker thread (single-threaded use)
        self.ocr = ocr

    # ---- stage 1: decoder -------------------------------------------------
    def _decode(self, cap, stride: int, max_frames, frame_q: "queue.Queue") -> None:
        true_idx = 0
        while True:
            if max_frames is not None and true_idx >= max_frames:
                break
            if not cap.grab():                       # advance; cheap vs full retrieve
                break
            true_idx += 1
            if (true_idx - 1) % stride != 0:         # skip every stride-th frame (decode only)
                continue
            ok, img = cap.retrieve()                 # format-convert only kept frames
            if not ok:
                break
            frame_q.put((true_idx, img))             # blocks if main is behind = backpressure
        self._frames_read = true_idx
        frame_q.put(None)                            # end sentinel

    # ---- stage 3: finalize worker ----------------------------------------
    def _finalize_loop(self, fin_q: "queue.Queue", scene, cam, offset, fps, crops_dir) -> None:
        while True:
            item = fin_q.get()
            if item is None:
                fin_q.task_done()
                return
            kind, tid, acc = item
            try:
                if self._finalize_one(kind, tid, acc, scene, cam, offset, fps, crops_dir):
                    self._written += 1
            finally:
                fin_q.task_done()

    def _finalize_one(self, kind, tid, acc, scene, cam, offset, fps, crops_dir) -> bool:
        if len(acc.frames) < MIN_TRACK_DETECTIONS:   # detector/tracker noise — drop (matches batch)
            return False
        if kind == "veh":
            prefix, (subtype, entity_type) = "t", profiles.subtype_vote(acc.clses)
        else:
            prefix, (subtype, entity_type) = "p", ("person", "person")

        f_start, f_end = min(acc.frames), max(acc.frames)
        ts_start, ts_end = offset + f_start / fps, offset + f_end / fps
        crop_records = acc.best_crop_records()
        crops = [record["crop"] for record in crop_records]
        crop_refs = []
        _io0 = time.perf_counter()
        for k, crop in enumerate(crops):
            fn = f"{prefix}{tid}_{k}.jpg"
            cv2.imwrite(str(crops_dir / fn), crop)
            crop_refs.append(paths.rel_key(crops_dir / fn))
        self._io_secs += time.perf_counter() - _io0

        _e0 = time.perf_counter()
        sem, crop_vectors = self.pool.embed_views(crops)
        color = self.pool.color_name(sem)[0] if entity_type == "vehicle" else None
        self._embed_secs += time.perf_counter() - _e0

        row = {
            "tracklet_id": f"{scene}_{cam}_{prefix}{tid}",
            "scene": scene, "camera_id": cam,
            "entity_type": entity_type, "subtype": subtype, "color": color,
            "frame_start": int(f_start), "frame_end": int(f_end),
            "ts_start_s": float(ts_start), "ts_end_s": float(ts_end),
            "wall_start": paths.wall_time(ts_start).isoformat(),
            "wall_end": paths.wall_time(ts_end).isoformat(),
            "num_detections": len(acc.frames), "avg_conf": float(np.mean(acc.confs)),
            "crop_refs": crop_refs,
            "video_ref": None,          # live: no GPU transcode; crops carry the UI
            "semantic_vector": sem,
            "reid_appearance": None, "reid_color": None, "person_attrs": None,
            "plate_text": None, "plate_conf": None, "plate_raw": None,
        }
        _w0 = time.perf_counter()
        crop_rows = [
            (row["tracklet_id"], k, int(record["frame_no"]), crop_refs[k],
             float(record["quality"]), crop_vectors[k])
            for k, record in enumerate(crop_records)
        ]
        self.writer.insert(row, crop_rows)
        self._io_secs += time.perf_counter() - _w0
        if entity_type == "vehicle" and self.ocr is not None:
            self.ocr.submit(row["tracklet_id"], int(f_end - f_start), crops)
        return True

    # ---- stage 2: detect + track (main) ----------------------------------
    def run(self, scene: str, cam: str, max_frames: int | None = None, stride: int = 1) -> dict:
        pool = self.pool
        prof, pdet = pool.prof, pool.pdet
        out = paths.cam_out(scene, cam)
        crops_dir = out / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        offset = paths.load_cam_offsets(scene)[cam]

        # LIVE_TRACKER overrides the profile's tracker yaml (e.g. a GMC-off config for
        # fixed CCTV). Relative paths resolve against apps/ingest/.
        tracker = os.environ.get("LIVE_TRACKER", prof.tracker)
        common = dict(persist=True, tracker=tracker, quantize=16,
                      device=pool.device, verbose=False)
        print(f"[live] tracker={tracker}")
        cap = cv2.VideoCapture(str(paths.cam_video(scene, cam)))
        fps = cap.get(cv2.CAP_PROP_FPS) or paths.DEFAULT_FPS
        if not fps or fps <= 0:
            fps = paths.DEFAULT_FPS

        veh_tracks: dict[int, _TrackAcc] = {}
        person_tracks: dict[int, _TrackAcc] = {}
        veh_last: dict[int, int] = {}
        person_last: dict[int, int] = {}
        veh_done: set[int] = set()
        person_done: set[int] = set()

        frame_q: "queue.Queue" = queue.Queue(maxsize=8)     # read-ahead depth
        fin_q: "queue.Queue" = queue.Queue(maxsize=256)     # ended-track backlog cap
        self._frames_read = 0
        self._written = 0
        # --- profiling accumulators (near-free; ultralytics already computes .speed) ---
        self._pp = self._inf = self._post = 0.0   # ms summed over detector calls
        self._n_det_calls = 0
        self._embed_secs = 0.0                    # SigLIP embed (finalize thread)
        self._io_secs = 0.0                       # crop-write + DB insert (finalize thread)
        # per-frame boxes for the player overlay — same detections.npy schema the batch
        # pipeline writes (frame,tid,x1,y1,x2,y2,conf,cls; person rows carry PERSON_CLS),
        # read back by the backend's /tracklets/{id}/boxes. Kept for tracks we accumulate.
        det_rows: list = []

        dec = threading.Thread(target=self._decode, args=(cap, stride, max_frames, frame_q),
                               daemon=True)
        fin = threading.Thread(target=self._finalize_loop,
                               args=(fin_q, scene, cam, offset, fps, crops_dir), daemon=True)
        dec.start()
        fin.start()

        frames_proc = 0
        detect_secs = 0.0
        loop_t0 = time.perf_counter()

        while True:
            got = frame_q.get()
            if got is None:
                break
            frame_idx, img = got
            frames_proc += 1
            H, W = img.shape[:2]

            d0 = time.perf_counter()
            veh_r = pool.veh.track(
                img, imgsz=prof.yolo_imgsz, conf=prof.yolo_conf,
                classes=list(prof.yolo_classes) if prof.yolo_classes else None, **common)[0]
            per_r = None
            if pool.per is not None:
                per_r = pool.per.track(img, imgsz=pdet.imgsz, conf=pdet.conf,
                                       classes=[pdet.class_id], **common)[0]
            detect_secs += time.perf_counter() - d0
            # decompose the detect time: preprocess/inference/postprocess (ms/img from
            # ultralytics); tracker+glue = detect_secs - these. If inference is a small
            # slice of detect_secs, the GPU is idle-waiting and the bottleneck is CPU.
            for r in (veh_r, per_r):
                if r is not None and getattr(r, "speed", None):
                    self._pp += r.speed.get("preprocess", 0.0)
                    self._inf += r.speed.get("inference", 0.0)
                    self._post += r.speed.get("postprocess", 0.0)
                    self._n_det_calls += 1

            person_boxes: list[tuple] = []
            if per_r is not None and per_r.boxes is not None and per_r.boxes.id is not None:
                pxyxy = per_r.boxes.xyxy.cpu().numpy()
                pids = per_r.boxes.id.cpu().numpy().astype(int)
                pconfs = per_r.boxes.conf.cpu().numpy()
                for box, pid, conf in zip(pxyxy, pids, pconfs):
                    person_boxes.append(tuple(box))
                    if int(pid) in person_done:
                        continue
                    _accumulate(person_tracks, _SINK, frame_idx, int(pid), box, conf, PERSON_CLS, img, W, H)
                    person_last[int(pid)] = frame_idx
                    det_rows.append((frame_idx, int(pid), box[0], box[1], box[2], box[3],
                                     float(conf), PERSON_CLS))

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
                    if int(tid) in veh_done:
                        continue
                    _accumulate(veh_tracks, _SINK, frame_idx, int(tid), box, conf, cls, img, W, H)
                    veh_last[int(tid)] = frame_idx
                    det_rows.append((frame_idx, int(tid), box[0], box[1], box[2], box[3],
                                     float(conf), int(cls)))
            _SINK.clear()

            # hand quiet tracks to the finalize worker (no inline paperwork)
            self._sweep(veh_tracks, veh_last, veh_done, frame_idx, "veh", fin_q)
            self._sweep(person_tracks, person_last, person_done, frame_idx, "per", fin_q)

        # flush tracks still active at end of stream
        for tid in list(veh_tracks):
            fin_q.put(("veh", tid, veh_tracks.pop(tid)))
        for tid in list(person_tracks):
            fin_q.put(("per", tid, person_tracks.pop(tid)))

        fin_q.put(None)     # drain + stop the finalize worker
        fin.join()
        dec.join()
        cap.release()

        # persist per-frame boxes for the overlay (matches batch detections.npy layout)
        np.save(str(out / "detections.npy"),
                np.array(det_rows, dtype=np.float32).reshape(-1, 8))
        loop_secs = time.perf_counter() - loop_t0

        # ---- profiling breakdown -------------------------------------------
        # detect_secs is the MAIN-thread cost (decode-wait + both .track() calls, since
        # the loop blocks on frame_q.get and runs detection inline). Split the .track()
        # cost into GPU inference vs CPU pre/post via ultralytics .speed; the remainder
        # is tracker(BoT-SORT)+glue+decode-wait. embed/io are FINALIZE-thread costs that
        # overlap detection, so they don't add to wall — shown to see if finalize is a
        # co-resident GPU competitor.
        inf_s, pp_s, post_s = self._inf / 1e3, self._pp / 1e3, self._post / 1e3
        track_glue = max(0.0, detect_secs - inf_s - pp_s - post_s)
        print(f"[profile] {cam}: wall={loop_secs:.1f}s  main-loop={detect_secs:.1f}s "
              f"({100*detect_secs/loop_secs:.0f}% of wall)")
        print(f"[profile]   detector .speed over {self._n_det_calls} calls: "
              f"inference={inf_s:.1f}s ({100*inf_s/loop_secs:.0f}%)  "
              f"preprocess={pp_s:.1f}s  postprocess={post_s:.1f}s")
        print(f"[profile]   tracker+glue+decode-wait (main-loop remainder) = {track_glue:.1f}s "
              f"({100*track_glue/loop_secs:.0f}% of wall)")
        print(f"[profile]   finalize thread (overlapped): SigLIP embed={self._embed_secs:.1f}s  "
              f"crop-write+insert={self._io_secs:.1f}s")

        frames_read = self._frames_read
        footage_secs = frames_read / fps if fps else 0.0
        return {
            "cam": cam, "stride": stride,
            "frames_read": frames_read, "frames_proc": frames_proc, "written": self._written,
            "wall_secs": round(loop_secs, 1),
            "footage_secs": round(footage_secs, 1),
            "realtime_x": round(footage_secs / loop_secs, 2) if loop_secs else 0.0,  # >1 = keeps up
            "detect_fps": round(frames_proc / detect_secs, 2) if detect_secs else 0.0,
            "gpu_path_fps": round(frames_proc / loop_secs, 2) if loop_secs else 0.0,
            "detect_secs": round(detect_secs, 1),
        }

    def _sweep(self, tracks, last, done, frame_idx, kind, fin_q) -> None:
        """Evict + enqueue any track unseen for TRACK_TIMEOUT_FRAMES (frees its crops from
        the main thread — ownership moves to the finalize worker)."""
        for tid in [t for t in tracks
                    if frame_idx - last.get(t, frame_idx) >= TRACK_TIMEOUT_FRAMES]:
            fin_q.put((kind, tid, tracks.pop(tid)))
            last.pop(tid, None)
            done.add(tid)
