"""render_camB.py — the VISUAL for experiment A: camera A vs synthetic camera B, side by side.

Applies the exact same augment.py preset the metric uses (on crops) to full frames of a short
segment, and lays the original next to the degraded copy so a viewer can SEE what "camera B" is.
Purely illustrative — it never touches the pipeline or the vectors. Encodes H.264/yuv420p via
ffmpeg (OpenCV's mpeg4 won't play in browsers / WhatsApp; kept small on purpose).

Usage:
  uv run python render_camB.py --scene SUR01 --cam c004 --start 60 --dur 20 --preset camB
"""
from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np

import augment
from pipeline import paths

_INK = (245, 245, 245)
_SHADOW = (20, 20, 20)


def _label(img: np.ndarray, text: str, accent) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (8, 34), accent, -1)
    cv2.putText(img, text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _SHADOW, 3, cv2.LINE_AA)
    cv2.putText(img, text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _INK, 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cam", default="c004")
    ap.add_argument("--start", type=float, default=60.0, help="segment start (seconds)")
    ap.add_argument("--dur", type=float, default=20.0, help="segment duration (seconds)")
    ap.add_argument("--preset", default="camB")
    ap.add_argument("--width", type=int, default=640, help="per-panel width")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = paths.cam_out(args.scene, args.cam) / "media" / f"{args.cam}.mp4"
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = args.width
    ph = int(round(sh * pw / sw))
    gap = 6
    ow, oh = pw * 2 + gap, ph
    spec, fn = augment.transform_for(args.preset)
    outdir = paths.INGEST_ROOT / "sanity_out"
    outdir.mkdir(exist_ok=True)
    out = args.out or str(outdir / f"camB_{args.cam}_{args.preset}.mp4")

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
         "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    n_frames = int(args.dur * fps)
    written = 0
    canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        a = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        b = fn(a.copy())
        la, lb = a.copy(), b.copy()
        _label(la, f"CAMERA A  -  {args.cam}  (original)", (90, 200, 90))
        _label(lb, f"CAMERA B  -  synthetic ({args.preset})", (80, 150, 240))
        canvas[:] = 40
        canvas[:, :pw] = la
        canvas[:, pw + gap:] = lb
        ff.stdin.write(canvas.tobytes())
        written += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    print(f"preset={args.preset} spec={spec}")
    print(f"wrote {out}  ({written} frames, {written / fps:.1f}s, {ow}x{oh})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
