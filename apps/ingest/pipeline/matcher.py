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
    __slots__ = ("tid", "cam", "ts0", "ts1", "vec", "plate")

    def __init__(self, tid, cam, ts0, ts1, vec, plate=None):
        self.tid, self.cam, self.ts0, self.ts1, self.vec = tid, cam, ts0, ts1, vec
        self.plate = plate


def load(scene: str, cams: list[str], w: float) -> list[Tracklet]:
    out = []
    for cam in cams:
        d = paths.cam_out(scene, cam)
        tracklets = json.loads((d / "tracklets.json").read_text())
        try:
            app = np.load(d / "vec" / "reid_appearance.npy")
        except FileNotFoundError:  # reid stage not run (e.g. SUR01) — plate-only still works
            app = np.zeros((len(tracklets), 1), dtype=np.float32)
        color = np.load(d / "vec" / "reid_color.npy")
        for i, t in enumerate(tracklets):
            if t["num_detections"] < 2:
                continue
            fused = np.concatenate([app[i], w * color[i]]) if w else app[i].astype(np.float32)
            n = np.linalg.norm(fused)
            if n > 0:
                fused = fused / n
            out.append(Tracklet(t["tracklet_id"], t["camera_id"], t["ts_start_s"],
                                t["ts_end_s"], fused, t.get("plate_text")))
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


def _plate_edges(tk: list[Tracklet]) -> list[tuple[float, int, int]]:
    """Must-link edges from validated plate reads: two tracklets with the same
    plate_text ARE the same physical vehicle (assertion tier — the constraint stack
    already gated these). All pairs within a plate group, sim=2.0 so they outrank
    every cosine edge; the cannot-link guard still applies (a same-camera
    time-overlapping 'pair' means one crop read a neighbour's plate — never merge)."""
    by_plate: dict[str, list[int]] = {}
    for i, t in enumerate(tk):
        if t.plate:
            by_plate.setdefault(t.plate, []).append(i)
    edges = []
    for members in by_plate.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                edges.append((2.0, members[a], members[b]))
    return edges


# NOTE: fragment merge measured WORSE on S01 (IDF1 0.417->0.407 even gap-gated) — VeRi
# embeddings confuse same-camera look-alikes more than they stitch true fragments here.
# Default off; keep it as a documented lever for cleaner data / other scenes.
# Plate must-links are the exception: they stitch same-camera fragments safely
# (threshold > 1 disables appearance edges entirely = plates-only mode, used on SUR01).
def match(scene: str, cams: list[str], threshold: float = 0.5, w: float = 0.0,
          fragment: bool = False, plates: bool = True):
    tk = load(scene, cams, w)
    n = len(tk)
    if n == 0:
        return {}, {"tracklets": 0, "clusters": 0, "multi_cam_groups": 0}
    V = np.stack([t.vec for t in tk])
    sims = V @ V.T
    edges = _candidate_edges(tk, sims, threshold, fragment)
    if plates:
        edges = sorted(_plate_edges(tk) + edges, key=lambda e: e[0], reverse=True)

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


def write_db(gid_of: dict[str, int], scene: str) -> int:
    from .db import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE tracklets SET global_id = NULL WHERE scene = %s", (scene,))
        cur.executemany(
            "UPDATE tracklets SET global_id = %s WHERE tracklet_id = %s",
            [(gid, tid) for tid, gid in gid_of.items()],
        )
        conn.commit()
        return cur.rowcount


def run(scene: str, cams: list[str] | None = None, threshold: float = 0.55, w: float = 0.0,
        fragment: bool = False, dry_run: bool = False, plates: bool = True,
        only_groups: bool = False) -> dict:
    """only_groups: write global_id only for multi-tracklet clusters, leaving
    singletons NULL — used for plates-only stitching (SUR01) so the backend's
    heuristic same-camera dedup still applies to everything unstitched."""
    cams = cams or paths.list_cams(scene)
    gid_of, stats = match(scene, cams, threshold, w, fragment, plates)
    if only_groups:
        from collections import Counter
        sizes = Counter(gid_of.values())
        gid_of = {tid: g for tid, g in gid_of.items() if sizes[g] > 1}
    (paths.OUTPUT_ROOT / scene).mkdir(parents=True, exist_ok=True)
    (paths.OUTPUT_ROOT / scene / "global_ids.json").write_text(json.dumps(gid_of, indent=2))
    if not dry_run:
        write_db(gid_of, scene)
    return {"scene": scene, "threshold": threshold, "w": w,
            "gids_written": len(gid_of), **stats}
