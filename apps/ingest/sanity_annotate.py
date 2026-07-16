"""Sanity-check detection+tracking visualization for the india profile.

Runs the SAME dual-detector merge as the pipeline (UVH-26 vehicles + YOLO11-X persons,
with pedestrian-FP suppression) over a capped segment of a SUR01 camera and writes a
WhatsApp-ready annotated MP4 (H.264/720p) plus sample JPGs. Vehicles green, persons blue.
NOT part of the real pipeline — visual smoke test only; reuses detect_track's merge helpers.

Usage:
  INGEST_PROFILE=india uv run python sanity_annotate.py --cam c001 --max-frames 600
"""
from __future__ import annotations

import argparse
import subprocess
import time

from gpu_setup import ensure_gpu_libs

ensure_gpu_libs()

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from pipeline import detect_track, paths, profiles  # noqa: E402

OUT_ROOT = paths.INGEST_ROOT / "sanity_out"
VEHICLE_COLOR = (0, 200, 0)    # BGR green
PERSON_COLOR = (255, 128, 0)   # BGR blue


def _ffmpeg_writer(out_path, w: int, h: int, fps: float) -> subprocess.Popen:
    """Pipe BGR frames to ffmpeg → H.264/yuv420p/faststart mp4 (WhatsApp-playable,
    unlike OpenCV's mpeg4). Downscale to 720p + crf 28 so a busy 30s clip stays
    under WhatsApp's ~16MB video cap (WhatsApp downscales anyway); labels stay legible."""
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-vf", "scale=-2:720",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow",
         "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )


def _draw(img, box, color, label) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def _boxes(res):
    """(xyxy, ids, confs, clses) numpy arrays, or None if no tracked boxes this frame."""
    b = res.boxes
    if b is None or b.id is None:
        return None
    return (b.xyxy.cpu().numpy(), b.id.cpu().numpy().astype(int),
            b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cam", default="c001")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--sample-jpgs", type=int, default=6, help="evenly-spaced annotated frames to dump")
    ap.add_argument("--conf", type=float, default=0.4, help="vehicle conf threshold (tuned up from profile's 0.3)")
    ap.add_argument("--no-agnostic-nms", action="store_true", help="disable class-agnostic NMS")
    # Person-detector overrides (default to the active profile's PersonDetector) — lets us
    # A/B a different person model (e.g. CrowdHuman) into a separate --out-root without
    # touching the profile, so the baseline sanity_out/ stays intact for side-by-side compare.
    ap.add_argument("--person-weights", default=None, help="override person detector weights")
    ap.add_argument("--person-imgsz", type=int, default=None, help="override person imgsz")
    ap.add_argument("--person-conf", type=float, default=None, help="override person conf")
    ap.add_argument("--out-root", default=None, help="override output root (default sanity_out/)")
    args = ap.parse_args()

    prof = profiles.use("india")
    names = {i: s for i, (s, _) in prof.class_map.items()}
    src = paths.cam_video(args.scene, args.cam)
    out_root = paths.INGEST_ROOT / args.out_root if args.out_root else OUT_ROOT
    out_dir = out_root / args.scene / args.cam
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ff = _ffmpeg_writer(out_dir / "annotated.mp4", w, h, fps)
    writer = ff.stdin

    common = dict(source=str(src), stream=True, persist=True, tracker=prof.tracker,
                  half=True, vid_stride=1, verbose=False)
    agnostic = not args.no_agnostic_nms
    veh_model = YOLO(prof.yolo_model)
    veh_stream = veh_model.track(imgsz=prof.yolo_imgsz, conf=args.conf, agnostic_nms=agnostic,
                                 classes=list(prof.yolo_classes) if prof.yolo_classes else None, **common)
    pdet = prof.person_detector
    p_weights = args.person_weights or pdet.weights
    p_imgsz = args.person_imgsz or pdet.imgsz
    p_conf = args.person_conf if args.person_conf is not None else pdet.conf
    per_model = YOLO(p_weights)
    per_stream = per_model.track(imgsz=p_imgsz, conf=p_conf, classes=[pdet.class_id], **common)
    print(f"[{args.cam}] UVH-26 (conf={args.conf}, agnostic_nms={agnostic}) + "
          f"{p_weights} person (imgsz={p_imgsz}, conf={p_conf}) -> {out_dir}")

    sample_every = max(1, args.max_frames // args.sample_jpgs)
    n = 0
    n_veh = n_per = 0
    veh_ids: set[int] = set()
    per_ids: set[int] = set()
    t0 = time.time()
    for veh_r, per_r in zip(veh_stream, per_stream):
        img = veh_r.orig_img.copy()

        person_boxes = []
        pb = _boxes(per_r)
        if pb is not None:
            for box, pid, conf, _ in zip(*pb):
                person_boxes.append(tuple(box))
                per_ids.add(int(pid))
                n_per += 1
                _draw(img, box, PERSON_COLOR, f"id:{pid} person {conf:.2f}")

        vb = _boxes(veh_r)
        if vb is not None:
            for box, tid, conf, cls in zip(*vb):
                if (person_boxes and cls in pdet.suppress_vehicle_classes
                        and detect_track._explained_by_person(tuple(box), person_boxes)):
                    continue
                veh_ids.add(int(tid))
                n_veh += 1
                _draw(img, box, VEHICLE_COLOR, f"id:{tid} {names.get(int(cls), '?')} {conf:.2f}")

        writer.write(img.tobytes())
        if n % sample_every == 0:
            cv2.imwrite(str(out_dir / f"frame_{n:04d}.jpg"), img)
        n += 1
        if n >= args.max_frames:
            break

    writer.close()
    ff.wait()
    dt = time.time() - t0
    print(f"[{args.cam}] {n} frames in {dt:.1f}s ({n/dt:.1f} fps) | "
          f"vehicles: {n_veh} dets / {len(veh_ids)} ids | persons: {n_per} dets / {len(per_ids)} ids")
    print(f"[{args.cam}] wrote {out_dir}/annotated.mp4 + {min(args.sample_jpgs, n)} sample JPGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
