"""LIVE tier entry point — streaming ingest driver.

Co-resident models, per-frame streaming, per-tracklet finalize+INSERT, async CPU plate OCR.
Writes searchable rows under --scene (default surat-live) WITHOUT touching the batch build:
rows are isolated by scene, and no schema change is made. Run per-camera to read clean
per-camera throughput numbers.

Usage:
  uv run python run_live.py --scene surat-live --cams c004
  uv run python run_live.py --scene surat-live --cams c004 --no-ocr        # pure GPU-path fps
  uv run python run_live.py --scene surat-live --cams c001 c002 --max-frames 500

Requires footage at footage_data/india/<scene>/<cam>/vdo.avi and an offsets file
footage_data/cam_timestamp/<scene>.txt (same layout the batch pipeline uses).
"""

from __future__ import annotations

from gpu_setup import ensure_gpu_libs

ensure_gpu_libs()

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

# Thread budget (must be set BEFORE torch/numpy/onnxruntime import). The live tier runs many
# CPU stages concurrently (decode, detector pre/post, per-tracklet finalize, 2 OCR workers);
# left uncapped, EACH library grabs all cores and they thrash (measured: load-avg 25 on 12
# cores, GPU starved to ~19%). Cap the shared pools so the GPU-feeding threads get scheduled.
# Respects anything the user already exported. See also live/ocr_worker (ONNX session cap).
_CPU_BUDGET = os.environ.setdefault("LIVE_CPU_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _CPU_BUDGET)

from pipeline import paths, profiles  # noqa: E402

from live.engine import LiveEngine  # noqa: E402
from live.models import ModelPool  # noqa: E402
from live.ocr_worker import AsyncOCR  # noqa: E402
from live.writer import DBWriter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="surat-live")
    ap.add_argument("--cams", nargs="*", default=None, help="default: all cams in the scene")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--profile", default="india", choices=list(profiles.PROFILES))
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (2 = skip every other; halves GPU load)")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip plate OCR (measure the pure GPU-path fps)")
    ap.add_argument("--ocr-workers", type=int, default=2)
    ap.add_argument("--tracker", default="pipeline/trackers/botsort_static.yaml",
                    help="tracker yaml; default is GMC-off for fixed CCTV (~1.7x faster than "
                         "the profile's botsort.yaml). Pass the profile default to restore GMC.")
    args = ap.parse_args()

    os.environ["LIVE_TRACKER"] = args.tracker   # read by LiveEngine.run

    # cap the runtime thread pools too (env only covers OMP-based libs at import)
    import cv2  # noqa: E402
    import torch  # noqa: E402
    n = int(os.environ["LIVE_CPU_THREADS"])
    cv2.setNumThreads(n)
    torch.set_num_threads(n)
    print(f"[live] cpu-thread budget={n} (cv2/torch/omp); ocr sessions capped in ocr_worker")

    prof = profiles.use(args.profile)
    cams = args.cams or paths.list_cams(args.scene)
    print(f"[live] profile={prof.name} scene={args.scene} cams={cams} stride={args.stride} "
          f"ocr={'off' if args.no_ocr else args.ocr_workers} max_frames={args.max_frames}")

    overall_t0 = time.perf_counter()
    t0 = time.perf_counter()
    pool = ModelPool()
    load_secs = time.perf_counter() - t0
    print(f"[live] models resident in {load_secs:.1f}s")

    ocr = None
    if not args.no_ocr:
        ocr = AsyncOCR(workers=args.ocr_workers)
        ocr.start()

    writer = DBWriter()
    engine = LiveEngine(pool, writer, ocr)

    tot_read = tot_footage = tot_wall = 0.0
    for cam in cams:
        print(f"\n=== [live] {cam} ===", flush=True)
        stats = engine.run(args.scene, cam, max_frames=args.max_frames, stride=args.stride)
        print("  ", stats)
        tot_read += stats["frames_read"]
        tot_footage += stats["footage_secs"]
        tot_wall += stats["wall_secs"]
        pool.reset_trackers()  # fresh BoT-SORT ids for the next camera

    ocr_secs = 0.0
    if ocr is not None:
        print("\n[live] draining OCR queue ...", flush=True)
        d0 = time.perf_counter()
        ocr.drain()
        ocr_secs = time.perf_counter() - d0
        print(f"[live] OCR done: {ocr.done}/{ocr.enqueued} tracklets, "
              f"{ocr.asserted} validated plates, +{ocr_secs:.1f}s")
        ocr.stop()
    writer.close()

    total_secs = time.perf_counter() - overall_t0
    rt = f"{tot_footage / tot_wall:.2f}x realtime" if tot_wall else "n/a"
    print(f"\n[live] TOTAL: {len(cams)} cam(s), {int(tot_read)} frames decoded, "
          f"{tot_footage:.0f}s footage processed in {tot_wall:.1f}s ({rt})")
    print(f"[live] wall clock end-to-end: {total_secs:.1f}s  "
          f"= model load {load_secs:.1f}s + processing {tot_wall:.1f}s + OCR drain {ocr_secs:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
