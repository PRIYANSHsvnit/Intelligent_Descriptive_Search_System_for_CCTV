"""Phase-3 cross-camera matcher runner.

  uv run python reid_match.py --scene S01 --sweep        # tune threshold x w by IDF1
  uv run python reid_match.py --scene S01 --threshold 0.5 --w 0   # commit global_id to DB
  uv run python reid_match.py --scene SUR01 --plates-only        # plate-based stitching

--sweep evaluates each (threshold, w) with the real IDF1 harness and reports the grid;
the plain run writes global_ids.json AND updates tracklets.global_id in Postgres.
--plates-only disables appearance edges (threshold > max cosine) and writes global_id
only for multi-tracklet plate groups — the stitching mode for SUR01, where appearance
matching is untrusted but a shared validated plate_text is proof of identity.
"""

from __future__ import annotations

import argparse

from pipeline import matcher, paths

import evaluate_reid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="S01")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--plates-only", action="store_true",
                    help="stitch by plate_text must-links only; no appearance edges; "
                         "write gids only for multi-tracklet groups")
    args = ap.parse_args()
    cams = args.cams or paths.list_cams(args.scene)

    if args.sweep:
        thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        ws = [0.0, 0.5, 1.0]
        best = None
        print(f"{'thr':>5} {'w':>4} {'clusters':>9} {'groups':>7} {'IDF1':>7} {'IDP':>7} {'IDR':>7}")
        for w in ws:
            for thr in thresholds:
                gid_of, st = matcher.match(args.scene, cams, threshold=thr, w=w)
                m = evaluate_reid.idf1_real(args.scene, cams, gid_of)
                print(f"{thr:>5} {w:>4} {st['clusters']:>9} {st['multi_cam_groups']:>7} "
                      f"{m['idf1']:>7} {m['idp']:>7} {m['idr']:>7}")
                if best is None or m["idf1"] > best[0]:
                    best = (m["idf1"], thr, w)
        print(f"\nbest IDF1={best[0]} at threshold={best[1]} w={best[2]}")
        print("commit it:  uv run python reid_match.py "
              f"--scene {args.scene} --threshold {best[1]} --w {best[2]}")
        return 0

    threshold = 2.0 if args.plates_only else args.threshold  # cosine <= 1: no appearance edges
    res = matcher.run(args.scene, cams, threshold=threshold, w=args.w,
                      only_groups=args.plates_only)
    print("matcher:", res)
    gid_map = {k: int(v) for k, v in
               __import__("json").loads((paths.OUTPUT_ROOT / args.scene / "global_ids.json").read_text()).items()}
    try:
        print("IDF1:", evaluate_reid.idf1_real(args.scene, cams, gid_map))
    except FileNotFoundError:
        print("IDF1: skipped (no ground truth for this scene)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
