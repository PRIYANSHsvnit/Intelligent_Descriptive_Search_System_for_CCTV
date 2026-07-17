"""Phase-1 ingest orchestrator for a scene.

Runs stage-major (all cams through one stage before the next) so each GPU model is
loaded/freed once per pass, per the VRAM strategy. Stages:
  detect -> attributes -> media -> siglip -> color -> vlm_attrs -> plate -> reid -> store

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
    plate,
    profiles,
    store,
    vlm_attrs,
)

ALL_STAGES = ["detect", "attributes", "media", "siglip", "color",
              "vlm_attrs", "plate", "reid", "store"]
# `vlm_attrs` = Qwen3-VL-4B via llama.cpp: structured person attributes with "unknown"
# allowed, so unreadable crops abstain instead of mislabeling (validated on SUR01/c004 —
# see plan.md "Person attributes" / the qwen3vl-person-attrs memory). It is the SOLE
# source of person_attrs, outfit colors included (the region-split SigLIP color stage
# was removed: fixed crop fractions blended bystanders into wrong labels and never
# abstained, so its mislabels outranked better VLM answers).
DEFAULT_STAGES = list(ALL_STAGES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="SUR01")
    ap.add_argument("--cams", nargs="*", default=None, help="default: all cams in the scene")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stages", nargs="*", default=DEFAULT_STAGES, choices=ALL_STAGES)
    ap.add_argument("--profile", default="india", choices=list(profiles.PROFILES),
                    help="domain profile (default india = real Surat footage; "
                         "use --profile cityflow --scene S01 for the ground-truth regression bench)")
    args = ap.parse_args()

    prof = profiles.use(args.profile)
    cams = args.cams or paths.list_cams(args.scene)
    print(f"profile={prof.name} scene={args.scene} cams={cams} "
          f"stages={args.stages} max_frames={args.max_frames}")

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
    if "vlm_attrs" in args.stages:
        each("vlm_attrs", lambda c: vlm_attrs.run(args.scene, c))
    if "plate" in args.stages:
        each("plate", lambda c: plate.run(args.scene, c))
        vlm_attrs.shutdown()      # free the ~4.7 GB of VRAM before the next GPU stage
    if "reid" in args.stages:
        each("reid", lambda c: embed_reid.run(args.scene, c))
    if "store" in args.stages:
        each("store", lambda c: store.run(args.scene, c))

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
