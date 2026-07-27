"""Cross-camera re-ID matcher (Phase 3): assign scene-namespaced global_id.

Guardrailed grouping that a single bad edge cannot cascade (NOT connected components):
  1. per-camera-pair MUTUAL nearest neighbor above a similarity threshold  (candidate edges)
  2. cannot-link: two tracklets from the SAME camera overlapping in time are never merged
  3. constrained agglomerative clustering: merge edges best-first, reject any merge that
     would put two same-camera time-overlapping tracklets in one cluster
Solo tracklets get their own singleton global_id (so every row is traceable).

IDENTITY GATES. `evaluate_person_identity.py` measured SUR01: on the dense c002 a
KNOWN-DIFFERENT (co-visible) person outranks the target's own second view 43% of the
time, and no cosine threshold separates the two distributions. Appearance therefore
cannot assert identity in a crowd — these gates carry the decision instead:

  L1 cannot-link — co-visible tracklets (same camera, overlapping in time) are different
     subjects and are never merged, transitively across whole clusters. Entity type is a
     cannot-link too: a person is never a vehicle, so they no longer compete for the same
     mutual-NN slot.
  L2 spatial gate — a same-camera fragment merge must be physically reachable: the second
     fragment has to start near where the first one ended, within travel distance for the
     elapsed gap. Speed is expressed in body-heights/second and scaled by the subject's own
     pixel height, which is the only depth cue available from a single view.
  L3 margin test — a mutual top-1 match must also beat the runner-up by `margin`. In a
     same-clothing crowd the runner-up is nearly as good, the margin collapses, and the
     matcher ABSTAINS rather than assert a wrong identity (same principle as the VLM's
     "unknown", see plan.md).

Gate parameters are per entity type (`GATES`): persons get the full stack; vehicles keep
the CityFlow-benched behaviour (margin 0, no spatial gate) because they have plate
must-links and a scored IDF1 regression bench that must not move silently.

Fusion: normalize(concat[appearance, w*color]). Default w=0 (appearance-only) — on S01
the noisy HSV color hurts (VeRi already captures color); w is tunable.

Reads output/<scene>/<cam>/{tracklets.json, detections.npy, vec/<vec_file>,
vec/reid_color.npy} (num_detections>=2, matching the DB). Writes
output/<scene>/global_ids.json and, unless dry_run, UPDATEs tracklets.global_id.

SCALE GOTCHA: match() materializes an NxN similarity matrix. That is nothing for a
CityFlow scene (1.2k vehicles) but SUR01 has 22k persons = ~1.9 GB. Run persons a
camera at a time (`--cams c004 --entity person`) until this is chunked.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np

from . import paths

# person rows carry this sentinel class id in detections.npy (detect_track.PERSON_CLS);
# duplicated here so the matcher never imports the torch/ultralytics stack.
_PERSON_CLS = 100


class Gate(NamedTuple):
    """Per-entity identity gates (see module docstring)."""

    margin: float     # L3: required lead over the runner-up (0.0 = test disabled)
    max_gap: float    # seconds a same-camera fragment may be interrupted
    speed_bh: float   # L2: max travel speed, in subject body-heights per second
    slack_px: float   # L2: box jitter / foot-point tolerance
    spatial: bool     # L2 on/off


# person: 1.5 m/s over a 1.7 m body ~= 0.9 body-heights/s; 1.2 leaves room for a jog and
# for foot-point error. vehicle: benched defaults — margin 0 and no spatial gate reproduce
# the S01 IDF1=0.417 configuration exactly.
GATES: dict[str, Gate] = {
    "person": Gate(margin=0.05, max_gap=10.0, speed_bh=1.2, slack_px=60.0, spatial=True),
    "vehicle": Gate(margin=0.0, max_gap=10.0, speed_bh=10.0, slack_px=60.0, spatial=False),
}
_DEFAULT_GATE = GATES["vehicle"]


def gate_for(entity: str) -> Gate:
    return GATES.get(entity, _DEFAULT_GATE)


class Tracklet:
    __slots__ = ("tid", "cam", "ts0", "ts1", "vec", "plate", "entity", "p0", "p1", "h")

    def __init__(self, tid, cam, ts0, ts1, vec, plate=None, entity="vehicle", pos=None):
        self.tid, self.cam, self.ts0, self.ts1, self.vec = tid, cam, ts0, ts1, vec
        self.plate, self.entity = plate, entity
        # foot point at first/last detection + median box height, or None when
        # detections.npy is unavailable (L2 then fails open for this tracklet)
        self.p0, self.p1, self.h = pos if pos else (None, None, None)


def _positions(scene: str, cam: str) -> dict[tuple[bool, int], tuple]:
    """{(is_person, track_id): (first_foot_xy, last_foot_xy, median_height)}.

    The foot point (bottom-centre of the box) is the subject's ground contact, so it
    tracks the subject instead of drifting with box height. Missing detections.npy
    returns {} and L2 simply does not fire.
    """
    path = paths.cam_out(scene, cam) / "detections.npy"
    if not path.exists():
        return {}
    det = np.load(path)
    if det.size == 0:
        return {}
    is_person = det[:, 7] == _PERSON_CLS
    # group by (entity, track_id), ordered by frame within each group
    order = np.lexsort((det[:, 0], det[:, 1], is_person))
    det, is_person = det[order], is_person[order]
    key = np.stack([is_person.astype(np.float64), det[:, 1]], axis=1)
    bounds = np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1
    out: dict[tuple[bool, int], tuple] = {}
    for grp in np.split(np.arange(len(det)), bounds):
        if grp.size == 0:
            continue
        rows = det[grp]
        foot = np.stack([(rows[:, 2] + rows[:, 4]) / 2.0, rows[:, 5]], axis=1)
        height = float(np.median(rows[:, 5] - rows[:, 3]))
        out[(bool(is_person[grp[0]]), int(rows[0, 1]))] = (foot[0], foot[-1], height)
    return out


def load(scene: str, cams: list[str], w: float, vec_file: str = "reid_appearance.npy",
         entity: str | None = None) -> list[Tracklet]:
    out = []
    for cam in cams:
        d = paths.cam_out(scene, cam)
        tracklets = json.loads((d / "tracklets.json").read_text())
        try:
            app = np.load(d / "vec" / vec_file)
        except FileNotFoundError:  # reid stage not run (e.g. SUR01) — plate-only still works
            app = np.zeros((len(tracklets), 1), dtype=np.float32)
        color = np.load(d / "vec" / "reid_color.npy")
        pos = _positions(scene, cam)
        for i, t in enumerate(tracklets):
            if t["num_detections"] < 2:
                continue
            ent = t.get("entity_type", "vehicle")
            if entity and ent != entity:
                continue
            fused = np.concatenate([app[i], w * color[i]]) if w else app[i].astype(np.float32)
            n = np.linalg.norm(fused)
            if n > 0:
                fused = fused / n
            out.append(Tracklet(t["tracklet_id"], t["camera_id"], t["ts_start_s"],
                                t["ts_end_s"], fused, t.get("plate_text"), ent,
                                pos.get((ent == "person", int(t["track_id"])))))
    return out


def _overlap(a: Tracklet, b: Tracklet) -> bool:
    return a.ts0 <= b.ts1 and b.ts0 <= a.ts1


def _gap(a: Tracklet, b: Tracklet) -> float:
    """Seconds between two non-overlapping tracklets (0 if they touch/overlap)."""
    return max(0.0, a.ts0 - b.ts1, b.ts0 - a.ts1)


def _spatial_ok(a: Tracklet, b: Tracklet, gate: Gate) -> bool:
    """L2: could one subject actually cover the ground between these two fragments?

    reach = slack + speed_bh * body_height_px * gap_seconds. Using the subject's own
    pixel height makes the budget shrink with distance from the camera, which is what
    perspective requires. Fails OPEN when either fragment has no stored position.
    """
    if not gate.spatial or a.h is None or b.h is None:
        return True
    first, second = (a, b) if a.ts1 <= b.ts0 else (b, a)
    gap = max(0.0, second.ts0 - first.ts1)
    height = max(first.h, second.h, 1.0)
    reach = gate.slack_px + gate.speed_bh * height * gap
    return float(np.hypot(*(second.p0 - first.p1))) <= reach


def _clears_margin(sub: np.ndarray, ai: int, bj: int, best: float, margin: float) -> bool:
    """L3: the winner must lead the runner-up on BOTH its row and its column.

    A crowd of look-alikes produces a near-tie — exactly the case where the correct
    action is to abstain rather than to pick whichever twin scored 0.001 higher.
    """
    if margin <= 0:
        return True
    row = np.delete(sub[ai], bj)
    col = np.delete(sub[:, bj], ai)
    runner = max(row.max(initial=-np.inf), col.max(initial=-np.inf))
    return not np.isfinite(runner) or (best - runner) >= margin


def _mutual_edges(tk, sims, A, B, threshold, same_cam, gate: Gate):
    """Mutual nearest neighbors between index lists A and B above threshold.
    For same_cam (fragment merge) only NON-overlapping pairs (L1) within gate.max_gap
    seconds that are also physically reachable (L2) are eligible. Every surviving
    mutual top-1 must then clear the margin test (L3)."""
    sub = sims[np.ix_(A, B)].copy()
    if same_cam:
        for ai, a in enumerate(A):
            for bj, b in enumerate(B):
                if (a == b or _overlap(tk[a], tk[b]) or _gap(tk[a], tk[b]) > gate.max_gap
                        or not _spatial_ok(tk[a], tk[b], gate)):
                    sub[ai, bj] = -np.inf
    if not np.isfinite(sub).any():
        return []
    a_best = sub.argmax(axis=1)
    b_best = sub.argmax(axis=0)
    edges = []
    for ai, a in enumerate(A):
        bj = a_best[ai]
        best = sub[ai, bj]
        if b_best[bj] != ai or not np.isfinite(best) or best < threshold:
            continue
        if not _clears_margin(sub, ai, bj, float(best), gate.margin):
            continue
        edges.append((float(best), a, B[bj]))
    return edges


def _candidate_edges(tk: list[Tracklet], sims: np.ndarray, threshold: float, fragment: bool):
    """Cross-camera mutual-NN edges, plus (if fragment) same-camera non-overlapping
    mutual-NN edges to merge one camera's fragments of a subject. Candidates are pooled
    per (camera, entity_type) so persons and vehicles never compete (L1).
    Returns [(sim, i, j)]."""
    n = len(tk)
    groups = sorted({(t.cam, t.entity) for t in tk})
    idx_by_group = {g: [i for i in range(n) if (tk[i].cam, tk[i].entity) == g] for g in groups}
    edges = []
    for gi, (cam_i, ent_i) in enumerate(groups):
        A = idx_by_group[(cam_i, ent_i)]
        gate = gate_for(ent_i)
        if fragment and len(A) > 1:  # per-camera fragment merge
            edges += _mutual_edges(tk, sims, A, A, threshold, True, gate)
        for cam_j, ent_j in groups[gi + 1:]:
            if ent_j != ent_i or cam_j == cam_i:  # entity cannot-link; same cam done above
                continue
            B = idx_by_group[(cam_j, ent_j)]
            if A and B:
                edges += _mutual_edges(tk, sims, A, B, threshold, False, gate)
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
# Default off; keep it as a documented lever for cleaner data / other scenes. The L2/L3
# gates target exactly that failure, so re-bench before assuming it still holds.
# Plate must-links are the exception: they stitch same-camera fragments safely
# (threshold > 1 disables appearance edges entirely = plates-only mode, used on SUR01).
def match(scene: str, cams: list[str], threshold: float = 0.5, w: float = 0.0,
          fragment: bool = False, plates: bool = True,
          vec_file: str = "reid_appearance.npy", entity: str | None = None):
    tk = load(scene, cams, w, vec_file, entity)
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
                if tk[i].entity != tk[j].entity:  # L1: a person is never a vehicle
                    return False
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
        only_groups: bool = False, vec_file: str = "reid_appearance.npy",
        entity: str | None = None) -> dict:
    """only_groups: write global_id only for multi-tracklet clusters, leaving
    singletons NULL — used for plates-only stitching (SUR01) so the backend's
    heuristic same-camera dedup still applies to everything unstitched."""
    cams = cams or paths.list_cams(scene)
    gid_of, stats = match(scene, cams, threshold, w, fragment, plates, vec_file, entity)
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
