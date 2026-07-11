"""Phase-1 ingest orchestrator for a scene.

Runs stage-major (all cams through one stage before the next) so each GPU model is
loaded/freed once per pass, per the VRAM strategy. Stages:
  detect  -> attributes -> media -> siglip -> reid -> store

Usage:
  uv run python run_ingest.py --scene S01
  uv run python run_ingest.py --scene S01 --cams c001 c002 --max-frames 300
  uv run python run_ingest.py --scene S01 --stages detect attributes siglip store

Call gpu_setup first (done here before importing torch-backed passes).
"""

from __future__ import annotations

import argparse

from gpu_setup import ensure_gpu_libs

ensure_gpu_libs()

from pipeline import (  # noqa: E402
    attributes,
    color_siglip,
    detect_track,
    embed_reid,
    embed_siglip,
    media,
    paths,
    store,
)

ALL_STAGES = ["detect", "attributes", "media", "siglip", "color", "reid", "store"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="S01")
    ap.add_argument("--cams", nargs="*", default=None, help="default: all cams in the scene")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stages", nargs="*", default=ALL_STAGES, choices=ALL_STAGES)
    args = ap.parse_args()

    cams = args.cams or paths.list_cams(args.scene)
    print(f"scene={args.scene} cams={cams} stages={args.stages} max_frames={args.max_frames}")

    def each(stage_name, fn):
        print(f"\n=== {stage_name} ===")
        for cam in cams:
            print(f"  [{stage_name}] {cam} ...", flush=True)
            print("   ", fn(cam))

    if "detect" in args.stages:
        each("detect", lambda c: detect_track.run(args.scene, c, max_frames=args.max_frames))
    if "attributes" in args.stages:
        each("attributes", lambda c: attributes.run(args.scene, c))
    if "media" in args.stages:
        each("media", lambda c: media.run(args.scene, c))
    if "siglip" in args.stages:
        each("siglip", lambda c: embed_siglip.run(args.scene, c))
    if "color" in args.stages:
        each("color", lambda c: color_siglip.run(args.scene, c))
    if "reid" in args.stages:
        each("reid", lambda c: embed_reid.run(args.scene, c))
    if "store" in args.stages:
        each("store", lambda c: store.run(args.scene, c))

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
