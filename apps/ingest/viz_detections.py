"""Overlay the pipeline's OWN detections.npy on the video — the honest view of what
detect_track kept (post person-merge/suppression). Vehicles green (india subtype),
persons blue. Sanity/debug only.

Usage: INGEST_PROFILE=india uv run python viz_detections.py --cam c004
"""
from __future__ import annotations

import argparse

from gpu_setup import ensure_gpu_libs

ensure_gpu_libs()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipeline import detect_track, paths, profiles  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cam", default="c004")
    ap.add_argument("--sample-jpgs", type=int, default=6)
    args = ap.parse_args()

    prof = profiles.use("india")
    names = {i: s for i, (s, _) in prof.class_map.items()}
    out = paths.cam_out(args.scene, args.cam)
    det = np.load(out / "detections.npy")  # frame,tid,x1,y1,x2,y2,conf,cls
    frames_with_det = sorted(set(det[:, 0].astype(int)))
    if not frames_with_det:
        print("no detections")
        return 1
    picks = frames_with_det[:: max(1, len(frames_with_det) // args.sample_jpgs)][: args.sample_jpgs]

    cap = cv2.VideoCapture(str(paths.cam_video(args.scene, args.cam)))
    for fno in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno - 1)  # detections are 1-based
        ok, img = cap.read()
        if not ok:
            continue
        for row in det[det[:, 0].astype(int) == fno]:
            _, tid, x1, y1, x2, y2, conf, cls = row
            is_person = int(cls) == detect_track.PERSON_CLS
            color = (255, 128, 0) if is_person else (0, 200, 0)  # BGR: person blue, vehicle green
            label = f"{'person' if is_person else names.get(int(cls), '?')} {conf:.2f}"
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, color, 2)
            cv2.putText(img, label, (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        fn = out / f"viz_{fno:04d}.jpg"
        cv2.imwrite(str(fn), img)
        n_p = int((det[(det[:, 0].astype(int) == fno)][:, 7] == detect_track.PERSON_CLS).sum())
        n_v = int((det[:, 0].astype(int) == fno).sum()) - n_p
        print(f"frame {fno}: {n_v} vehicles + {n_p} persons -> {fn}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
