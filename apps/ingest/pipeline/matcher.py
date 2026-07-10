"""Cross-camera re-ID matcher (Phase 3): assign scene-namespaced global_id.

Guardrailed grouping that a single bad edge cannot cascade (NOT connected components):
  1. per-camera-pair MUTUAL nearest neighbor above a similarity threshold  (candidate edges)
  2. cannot-link: two tracklets from the SAME camera overlapping in time are never merged
  3. constrained agglomerative clustering: merge edges best-first, reject any merge that
     would put two same-camera time-overlapping tracklets in one cluster
Solo tracklets get their own singleton global_id (so every row is traceable).

Fusion: normalize(concat[appearance, w*color]). Default w=0 (appearance-only) — on S01
the noisy HSV color hurts (VeRi already captures color); w is tunable.

Reads output/<scene>/<cam>/{tracklets.json, vec/reid_appearance.npy, vec/reid_color.npy}
(num_detections>=2, matching the DB). Writes output/<scene>/global_ids.json and, unless
dry_run, UPDATEs tracklets.global_id.
"""

from __future__ import annotations

import json

import numpy as np

from . import paths


class Tracklet:
    __slots__ = ("tid", "cam", "ts0", "ts1", "vec")

    def __init__(self, tid, cam, ts0, ts1, vec):
        self.tid, self.cam, self.ts0, self.ts1, self.vec = tid, cam, ts0, ts1, vec


def load(scene: str, cams: list[str], w: float) -> list[Tracklet]:
    out = []
    for cam in cams:
        d = paths.cam_out(scene, cam)
        tracklets = json.loads((d / "tracklets.json").read_text())
        app = np.load(d / "vec" / "reid_appearance.npy")
        color = np.load(d / "vec" / "reid_color.npy")
        for i, t in enumerate(tracklets):
            if t["num_detections"] < 2:
                continue
            fused = np.concatenate([app[i], w * color[i]]) if w else app[i].astype(np.float32)
            n = np.linalg.norm(fused)
            if n > 0:
                fused = fused / n
            out.append(Tracklet(t["tracklet_id"], t["camera_id"], t["ts_start_s"], t["ts_end_s"], fused))
    return out


def _overlap(a: Tracklet, b: Tracklet) -> bool:
    return a.ts0 <= b.ts1 and b.ts0 <= a.ts1


def _gap(a: Tracklet, b: Tracklet) -> float:
    """Seconds between two non-overlapping tracklets (0 if they touch/overlap)."""
    return max(0.0, a.ts0 - b.ts1, b.ts0 - a.ts1)


def _mutual_edges(tk, sims, A, B, threshold, same_cam, max_gap=10.0):
    """Mutual nearest neighbors between index lists A and B above threshold.
    For same_cam (fragment merge) only NON-overlapping pairs within max_gap seconds
    are eligible (a car briefly re-entering — not two look-alikes minutes apart)."""
    sub = sims[np.ix_(A, B)].copy()
    if same_cam:
        for ai, a in enumerate(A):
            for bj, b in enumerate(B):
                if a == b or _overlap(tk[a], tk[b]) or _gap(tk[a], tk[b]) > max_gap:
                    sub[ai, bj] = -np.inf
    if not np.isfinite(sub).any():
        return []
    a_best = sub.argmax(axis=1)
    b_best = sub.argmax(axis=0)
    edges = []
    for ai, a in enumerate(A):
        bj = a_best[ai]
        if b_best[bj] == ai and np.isfinite(sub[ai, bj]) and sub[ai, bj] >= threshold:
            edges.append((float(sub[ai, bj]), a, B[bj]))
    return edges


def _candidate_edges(tk: list[Tracklet], sims: np.ndarray, threshold: float, fragment: bool):
    """Cross-camera mutual-NN edges, plus (if fragment) same-camera non-overlapping
    mutual-NN edges to merge one camera's fragments of a car. Returns [(sim, i, j)]."""
    n = len(tk)
    cams = sorted({t.cam for t in tk})
    idx_by_cam = {c: [i for i in range(n) if tk[i].cam == c] for c in cams}
    edges = []
    for ci in range(len(cams)):
        A = idx_by_cam[cams[ci]]
        if fragment and len(A) > 1:  # per-camera fragment merge
            edges += _mutual_edges(tk, sims, A, A, threshold, same_cam=True)
        for cj in range(ci + 1, len(cams)):
            B = idx_by_cam[cams[cj]]
            if A and B:
                edges += _mutual_edges(tk, sims, A, B, threshold, same_cam=False)
    edges.sort(key=lambda e: e[0], reverse=True)
    return edges


# NOTE: fragment merge measured WORSE on S01 (IDF1 0.417->0.407 even gap-gated) — VeRi
# embeddings confuse same-camera look-alikes more than they stitch true fragments here.
# Default off; keep it as a documented lever for cleaner data / other scenes.
def match(scene: str, cams: list[str], threshold: float = 0.5, w: float = 0.0,
          fragment: bool = False):
    tk = load(scene, cams, w)
    n = len(tk)
    if n == 0:
        return {}, {"tracklets": 0, "clusters": 0, "multi_cam_groups": 0}
    V = np.stack([t.vec for t in tk])
    sims = V @ V.T
    edges = _candidate_edges(tk, sims, threshold, fragment)

    # union-find with a cannot-link guard checked against full cluster membership
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_merge(ra, rb):
        for i in members[ra]:
            for j in members[rb]:
                if tk[i].cam == tk[j].cam and _overlap(tk[i], tk[j]):
                    return False
        return True

    for _sim, i, j in edges:
        ra, rb = find(i), find(j)
        if ra == rb or not can_merge(ra, rb):
            continue
        parent[rb] = ra
        members[ra].extend(members[rb])
        del members[rb]

    # assign scene-local global_ids (1..K), stable by smallest member index
    roots = sorted(members.keys(), key=lambda r: min(members[r]))
    gid_of = {}
    for gid, r in enumerate(roots, start=1):
        for i in members[r]:
            gid_of[tk[i].tid] = gid

    groups = sum(1 for r in members if len(members[r]) > 1)
    return gid_of, {"tracklets": n, "clusters": len(members), "multi_cam_groups": groups}


def write_db(gid_of: dict[str, int]) -> int:
    from .db import connect

    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE tracklets SET global_id = %s WHERE tracklet_id = %s",
            [(gid, tid) for tid, gid in gid_of.items()],
        )
        conn.commit()
        return cur.rowcount


def run(scene: str, cams: list[str] | None = None, threshold: float = 0.55, w: float = 0.0,
        fragment: bool = False, dry_run: bool = False) -> dict:
    cams = cams or paths.list_cams(scene)
    gid_of, stats = match(scene, cams, threshold, w, fragment)
    (paths.OUTPUT_ROOT / scene).mkdir(parents=True, exist_ok=True)
    (paths.OUTPUT_ROOT / scene / "global_ids.json").write_text(json.dumps(gid_of, indent=2))
    if not dry_run:
        write_db(gid_of)
    return {"scene": scene, "threshold": threshold, "w": w, **stats}
