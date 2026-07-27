"use client";
/* eslint-disable @next/next/no-img-element -- evidence images are served dynamically by FastAPI */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import styles from "./page.module.css";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

type Result = {
  tracklet_id: string;
  scene: string;
  camera_id: string;
  camera_label?: string;
  entity_type?: "person" | "vehicle";
  subtype: string;
  color: string | null;
  ts_start_s: number;
  ts_end_s: number;
  video_start_s?: number;
  video_end_s?: number;
  score: number;
  crop_url: string | null;
  video_url: string | null;
  global_id: number | null;
  attrs?: string[];
  plate?: string | null;
  plate_conf?: number | null;
  sightings?: number;
  matched_on?: "exact" | "partial" | "fuzzy";
  matched_crop_ref?: string | null;
  component_scores?: {
    caption: string; score: number; kind?: "overall" | "component";
    match_strength?: "strong" | "moderate" | "weak";
    supporting_crop_ref?: string | null; supporting_crop_url?: string | null;
  }[];
};

type SearchEvidenceContext = {
  original_query: string;
  rewritten_query?: string | null;
  timestamp_utc: string;
  filters: Record<string, string | number | null>;
  method: string;
  aggregation?: string | null;
  composition?: string | null;
  search_captions?: string[];
};

type VerificationResult = {
  valid: boolean;
  status: "VALID" | "TAMPERED";
  export_id?: string | null;
  errors: string[];
  warnings: string[];
};

type Camera = {
  camera_id: string;
  camera_label: string;
  count: number;
  ts_start_s: number;
  ts_end_s: number;
};

type Hop = {
  tracklet_id: string;
  camera_id: string;
  camera_label?: string;
  ts_start_s: number;
  ts_end_s: number;
  video_start_s?: number;
  video_end_s?: number;
  crop_url: string | null;
  video_url: string | null;
};

type Suggestion = {
  action: "expand_time" | "remove_color" | "all_entity_types";
  label: string;
};

type CaseItem = Result & {
  case_id: string;
  status: "pinned" | "excluded";
  note?: string | null;
  result_snapshot?: Partial<Result> & { search?: SearchEvidenceContext; retrieval?: Record<string, unknown> };
  created_at: string;
  updated_at: string;
};

type AuditEvent = {
  event_id: string; event_type: string; occurred_at: string; officer: string;
  original_query?: string | null; latency_ms?: number | null;
  metadata?: Record<string, unknown>;
};

type CaseBoardData = {
  case: { case_id: string; title?: string | null; officer: string; created_at: string };
  items: CaseItem[];
};

// describe/plate examples share one bar — clicking a plate one flips the detected mode
const EXAMPLES = ["person with a backpack", "woman in a red saree", "rider without helmet", "GJ05JP4237", "4237"];

// what the operator can be looking for. "" = no entity_type filter at all.
const LOOKING_FOR: { value: "" | "person" | "vehicle"; label: string }[] = [
  { value: "", label: "Anything" },
  { value: "person", label: "People" },
  { value: "vehicle", label: "Vehicles" },
];

// the stored vehicle-colour vocabulary (pipeline/cfg.py COLOR_VOCAB) + a swatch each
const COLORS: { name: string; swatch: string }[] = [
  { name: "white", swatch: "#eef2f7" }, { name: "silver", swatch: "#c2cad5" },
  { name: "gray", swatch: "#7c8797" }, { name: "black", swatch: "#161b23" },
  { name: "red", swatch: "#d6444f" }, { name: "blue", swatch: "#3d78d6" },
  { name: "green", swatch: "#3ba168" }, { name: "yellow", swatch: "#e6c33f" },
  { name: "orange", swatch: "#e0813c" }, { name: "brown", swatch: "#8a5b3c" },
];

// scene-clock seconds -> time of day h:mm:ss (SUR01 offsets are seconds since midnight)
function fmt(t: number): string {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}
function fmtClip(t: number): string {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}
// how long the object stayed on this camera, in words an operator can act on
function fmtSpan(a: number, b: number): string {
  const d = Math.max(1, Math.round(b - a));
  return d < 60 ? `${d} sec` : `${Math.floor(d / 60)} min ${d % 60} sec`;
}
const pad2 = (n: number) => String(n).padStart(2, "0");
const secToHHMM = (s: number) => `${pad2(Math.floor(s / 3600) % 24)}:${pad2(Math.floor((s % 3600) / 60))}`;
const hhmmToSec = (s: string) => { const [h, m] = s.split(":").map(Number); return (h || 0) * 3600 + (m || 0) * 60; };
const titleise = (s: string) => s.replaceAll("_", " ");

/* ---------------------------- match strength ----------------------------
   Deliberately RELATIVE to the current result set (same rule the backend uses
   for component strengths: 35th/70th percentile), never a calibrated probability.
   Plate hits carry their own truth — how much of the registration matched. */
type Strength = "strong" | "moderate" | "weak";

function bucketer(values: number[]): (v: number) => Strength {
  if (values.length < 3) return () => "strong";
  const sorted = [...values].sort((a, b) => a - b);
  const at = (p: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))]!;
  const low = at(0.35), high = at(0.7);
  return (v) => (v >= high ? "strong" : v >= low ? "moderate" : "weak");
}

function strengthOf(r: Result, bucket: (v: number) => Strength): Strength {
  if (r.matched_on) return r.matched_on === "exact" ? "strong" : r.matched_on === "partial" ? "moderate" : "weak";
  const overall = r.component_scores?.find((c) => c.kind === "overall")?.match_strength;
  return overall ?? bucket(r.score);
}

function strengthCopy(r: Result, s: Strength): string {
  if (r.matched_on === "exact") return "Plate matches exactly";
  if (r.matched_on === "partial") return "Part of the plate matches";
  if (r.matched_on === "fuzzy") return "Plate looks similar";
  return s === "strong" ? "Very likely match" : s === "moderate" ? "Possible match" : "Weaker match";
}

// short form for the badge over the photo; the card's Match row carries the full sentence
function strengthShort(r: Result, s: Strength): string {
  if (r.matched_on === "exact") return "Plate exact";
  if (r.matched_on === "partial") return "Plate partial";
  if (r.matched_on === "fuzzy") return "Plate similar";
  return s === "strong" ? "Very likely" : s === "moderate" ? "Possible" : "Weaker";
}

const GROUPS: { key: Strength; title: string; blurb: string }[] = [
  // a group sits in the band of its best sighting, so the copy says "hold", not "are"
  { key: "strong", title: "Very likely", blurb: "Start here — these groups hold your closest matches." },
  { key: "moderate", title: "Possible", blurb: "Part of your description fits these groups." },
  { key: "weak", title: "Less likely", blurb: "Shown anyway — keep scrolling if you haven’t found the one yet." },
];

/* --------------------------- grouping the results ---------------------------
   Two levels, both read off data we actually store — nothing is inferred.

   • A STACK is one physical person or vehicle seen more than once: rows the
     pipeline already joined (global_id) or rows carrying the same registration.
     Nothing else counts as "the same one" — two white hatchbacks a minute apart
     are two results, not one, and this tool never claims otherwise.
   • A BOX holds look-alikes: same colour and body type for vehicles, same
     colour worn on top for people. This is what the operator scans. Boxes run
     best-first and the weaker ones sit further down the page.               */
type Stack = { key: string; lead: Result; others: Result[] };
type Box = { key: string; label: string; strength: Strength; stacks: Stack[]; count: number };

const capitalise = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const plural = (s: string) => (/(s|sh|ch|x)$/.test(s) ? `${s}es` : `${s}s`);
const an = (w: string) => (/^[aeiou]/i.test(w) ? `an ${w}` : `a ${w}`);
const isPerson = (r: Result) => (r.entity_type ?? (r.subtype === "person" ? "person" : "vehicle")) === "person";

// the only two things that prove two sightings are the same object
function sameObjectKey(r: Result): string | null {
  if (r.global_id != null) return `id:${r.scene}:${r.global_id}`;
  const plate = r.plate?.toUpperCase().replace(/[^A-Z0-9]/g, "");
  return plate && plate.length >= 4 ? `plate:${plate}` : null;
}

function stackResults(rows: Result[]): Stack[] {
  const stacks: Stack[] = [];
  const byKey = new Map<string, Stack>();
  for (const r of rows) {
    const key = sameObjectKey(r);
    const hit = key ? byKey.get(key) : undefined;
    if (hit) { hit.others.push(r); continue; }
    const stack: Stack = { key: key ?? `one:${r.tracklet_id}`, lead: r, others: [] };
    if (key) byKey.set(key, stack);
    stacks.push(stack);
  }
  return stacks;
}

const HEADWEAR = new Set(["helmet", "cap", "hat", "turban", "no helmet"]);

/* What this result looks like, in the words the box will be labelled with.
   Two keys, because one is never right for both queries: search "red shirt" and
   every person is wearing a red top, so the fine key (top AND bottom) is what
   actually separates them — but on a query where the fine key leaves boxes of
   one, those fold back up into the coarse key rather than littering the page. */
type BoxKeys = { key: string; label: string; coarseKey: string; coarseLabel: string };

function boxOf(r: Result): BoxKeys {
  if (isPerson(r)) {
    const top = r.attrs?.find((a) => a.endsWith(" top"))?.slice(0, -4);
    const bottom = r.attrs?.find((a) => a.endsWith(" bottom"))?.slice(0, -7);
    if (top) {
      const coarseKey = `pc:${top}`;
      const coarseLabel = `People in ${an(top)} top`;
      return bottom
        ? { key: `pf:${top}:${bottom}`, label: `${coarseLabel} and ${bottom} bottom`, coarseKey, coarseLabel }
        : { key: coarseKey, label: coarseLabel, coarseKey, coarseLabel };
    }
    const head = r.attrs?.find((a) => HEADWEAR.has(a));
    const other = { coarseKey: "pc:any", coarseLabel: "Other people" };
    return head
      ? { key: `pf:head:${head}`, label: `People wearing ${an(head)}`, ...other }
      : { key: other.coarseKey, label: other.coarseLabel, ...other };
  }
  const kind = plural(titleise(r.subtype));
  const coarseKey = `vc:${r.subtype}`;
  const coarseLabel = capitalise(kind);
  return r.color
    ? { key: `vf:${r.color}:${r.subtype}`, label: capitalise(`${r.color} ${kind}`), coarseKey, coarseLabel }
    : { key: coarseKey, label: coarseLabel, coarseKey, coarseLabel };
}

// routes a typed query to plate-search when it looks like a registration, else semantic.
// runs BEFORE any translate/rewrite — translating "GJ05JP4237" would be nonsense.
function detectMode(raw: string): "plate" | "describe" {
  const t = raw.trim();
  if (!t) return "describe";
  const norm = t.toUpperCase().replace(/[^A-Z0-9]/g, "");   // strip separators first (spaced plates)
  if (norm.length < 2 || norm.length > 12) return "describe";
  if (!/[0-9]/.test(norm)) return "describe";               // a plate always has digits
  return /^[A-Z]{0,3}[0-9]{1,4}[A-Z]{0,3}[0-9]{0,4}$/.test(norm) ? "plate" : "describe";
}

// hand-annotated camera pixel positions on cam_loc/<scene>.png (only S01 mapped;
// SUR01 cameras are non-overlapping, so trace shows a per-camera timeline instead).
const CAM_POS: Record<string, Record<string, [number, number]>> = {
  S01: { c001: [0.61, 0.67], c002: [0.33, 0.38], c003: [0.68, 0.4], c004: [0.33, 0.62], c005: [0.53, 0.32] },
};

export default function Home() {
  const [scene, setScene] = useState("SUR01");
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [total, setTotal] = useState(0);

  const [mode, setMode] = useState<"search" | "photo">("search");
  const [q, setQ] = useState("man in an orange shirt");
  const [forceMode, setForceMode] = useState<null | "plate" | "describe">(null);

  const [cameraId, setCameraId] = useState("");
  const [lookingFor, setLookingFor] = useState<"" | "person" | "vehicle">("");
  const [color, setColor] = useState("");
  const [timeSet, setTimeSet] = useState(false);
  const [tFrom, setTFrom] = useState("06:10");
  const [tTo, setTTo] = useState("06:15");

  const [results, setResults] = useState<Result[]>([]);
  const [fellBack, setFellBack] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [active, setActive] = useState<Result | null>(null);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  const [context, setContext] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchEvidence, setSearchEvidence] = useState<SearchEvidenceContext | null>(null);
  const [verification, setVerification] = useState<VerificationResult | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [caseId, setCaseId] = useState(`CASE-${new Date().toISOString().slice(0, 10).replaceAll("-", "")}`);
  const [officer, setOfficer] = useState("");
  const [caseOpen, setCaseOpen] = useState(false);
  const [boardOpen, setBoardOpen] = useState(false);
  const [board, setBoard] = useState<CaseBoardData | null>(null);
  const [timeline, setTimeline] = useState<AuditEvent[]>([]);
  const [boardMessage, setBoardMessage] = useState<string | null>(null);
  const [sortMode, setSortMode] = useState<"match" | "matchflat" | "earliest" | "latest" | "camera">("match");
  const [density, setDensity] = useState<"comfortable" | "compact">("comfortable");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const queryRef = useRef<HTMLInputElement>(null);

  const auto = detectMode(q);
  const effMode: "plate" | "describe" = forceMode ?? auto;
  const identified = Boolean(caseId.trim() && officer.trim());

  // colour is a vehicle-level column — the backend rejects person+colour outright
  useEffect(() => { if (lookingFor === "person" && color) setColor(""); }, [lookingFor, color]);

  // load the scene's cameras (labels + counts + coverage) for the location picker
  useEffect(() => {
    let ok = true;
    fetch(`${API}/scene-cameras/${scene}`)
      .then((r) => r.json())
      .then((d) => { if (ok) { setCameras(d.cameras ?? []); setTotal(d.total ?? 0); } })
      .catch(() => ok && setCameras([]));
    setCameraId("");
    return () => { ok = false; };
  }, [scene]);

  const requireIdentity = useCallback(() => {
    setCaseOpen(true);
    setError("Add a case number and your name before searching — every search is written to the case file.");
  }, []);

  const runSearch = useCallback(async (queryArg?: string, modeOverride?: "plate" | "describe") => {
    const query = queryArg ?? q;
    if (!query.trim()) return;
    if (!caseId.trim() || !officer.trim()) { requireIdentity(); return; }
    const em = modeOverride ?? forceMode ?? detectMode(query);
    setLoading(true); setSearched(true); setContext(null); setError(null);
    const startedAt = performance.now();
    try {
      const p = new URLSearchParams({ limit: "24" });
      if (em === "plate") p.set("plate", query.trim());
      else p.set("q", query);
      if (scene) p.set("scene", scene);
      if (cameraId) p.set("camera_id", cameraId);
      if (lookingFor) p.set("type", lookingFor);
      if (color && lookingFor !== "person") p.set("color", color);
      if (timeSet) { p.set("t0", String(hhmmToSec(tFrom))); p.set("t1", String(hhmmToSec(tTo))); }
      p.set("case_id", caseId.trim()); p.set("officer", officer.trim());
      const r = await fetch(`${API}/search?${p}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setResults(data.results ?? []);
      setFellBack(Boolean(data.fell_back));
      setSuggestions(data.suggestions ?? []);
      setSearchEvidence({
        original_query: query,
        rewritten_query: data.rewritten_query ?? null,
        timestamp_utc: data.searched_at_utc ?? new Date().toISOString(),
        filters: {
          scene, camera_id: cameraId || null,
          type: lookingFor || null,
          color: color || null,
          t0: timeSet ? hhmmToSec(tFrom) : null,
          t1: timeSet ? hhmmToSec(tTo) : null,
          query_mode: em,
        },
        method: em === "plate" ? "layered_plate_lookup" : (data.search_mode ?? "semantic_search"),
        aggregation: data.aggregation ?? null,
        composition: data.composition ?? "weighted",
        search_captions: data.search_captions ?? [],
      });
    } catch (e) {
      setResults([]);
      setSuggestions([]);
      setError(`Search failed (${e instanceof Error ? e.message : "network"}) — is the backend running?`);
    } finally {
      setLatencyMs(performance.now() - startedAt);
      setLoading(false);
    }
  }, [q, forceMode, scene, cameraId, lookingFor, color, timeSet, tFrom, tTo, caseId, officer, requireIdentity]);

  const runImageSearch = useCallback(async (file: File) => {
    if (!caseId.trim() || !officer.trim()) { requireIdentity(); return; }
    setLoading(true); setSearched(true); setContext("matches for the uploaded photo"); setError(null);
    const startedAt = performance.now();
    try {
      const fd = new FormData();
      fd.append("image", file);
      if (scene) fd.append("scene", scene);
      if (cameraId) fd.append("camera_id", cameraId);
      if (lookingFor) fd.append("type", lookingFor);
      if (color && lookingFor !== "person") fd.append("color", color);
      if (timeSet) { fd.append("t0", String(hhmmToSec(tFrom))); fd.append("t1", String(hhmmToSec(tTo))); }
      fd.append("limit", "24");
      fd.append("case_id", caseId.trim()); fd.append("officer", officer.trim());
      const r = await fetch(`${API}/search/image`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setResults(data.results ?? []);
      setFellBack(Boolean(data.fell_back));
      setSuggestions(data.suggestions ?? []);
      setSearchEvidence({
        original_query: "[reference image search]",
        timestamp_utc: data.searched_at_utc ?? new Date().toISOString(),
        filters: {
          scene, camera_id: cameraId || null,
          type: lookingFor || null, color: color || null,
          t0: timeSet ? hhmmToSec(tFrom) : null,
          t1: timeSet ? hhmmToSec(tTo) : null,
        },
        method: data.search_mode ?? "siglip_image_search",
        aggregation: data.aggregation ?? null,
      });
    } catch (e) {
      setResults([]);
      setSuggestions([]);
      setError(`Image search failed (${e instanceof Error ? e.message : "network"}).`);
    } finally {
      setLatencyMs(performance.now() - startedAt);
      setLoading(false);
    }
  }, [scene, cameraId, lookingFor, color, timeSet, tFrom, tTo, caseId, officer, requireIdentity]);

  const runSimilar = useCallback(async (r: Result, vec: "semantic" | "reid") => {
    if (!caseId.trim() || !officer.trim()) { requireIdentity(); return; }
    setActive(null); setLoading(true); setSearched(true);
    setContext(vec === "reid"
      ? `other sightings of this same ${titleise(r.subtype)}`
      : `objects that look like this ${titleise(r.subtype)}`);
    setError(null);
    try {
      const p = new URLSearchParams({ tracklet_id: r.tracklet_id, vec, limit: "24", case_id: caseId.trim(), officer: officer.trim() });
      const res = await fetch(`${API}/search/similar?${p}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResults(data.results ?? []); setFellBack(false); setSuggestions([]);
      setSearchEvidence({
        original_query: `${vec} similarity from ${r.tracklet_id}`,
        timestamp_utc: new Date().toISOString(),
        filters: { source_tracklet_id: r.tracklet_id, entity_type: r.subtype },
        method: vec === "reid" ? "reid_similarity" : "semantic_similarity",
      });
      if (data.fell_back && vec === "reid")
        setContext(`look-alikes of this ${titleise(r.subtype)} (no re-ID fingerprint stored — visual similarity only)`);
    } catch (e) {
      setResults([]); setSuggestions([]); setError(`Similar search failed (${e instanceof Error ? e.message : "network"}).`);
    } finally { setLoading(false); }
  }, [caseId, officer, requireIdentity]);

  const loadBoard = useCallback(async () => {
    if (!caseId.trim() || !officer.trim()) return;
    await fetch(`${API}/cases`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_id: caseId.trim(), officer: officer.trim() }),
    });
    const [boardResponse, timelineResponse] = await Promise.all([
      fetch(`${API}/cases/${encodeURIComponent(caseId.trim())}`),
      fetch(`${API}/cases/${encodeURIComponent(caseId.trim())}/timeline`),
    ]);
    if (!boardResponse.ok) throw new Error(`HTTP ${boardResponse.status}`);
    setBoard(await boardResponse.json());
    setTimeline(timelineResponse.ok ? (await timelineResponse.json()).events ?? [] : []);
  }, [caseId, officer]);

  const setCaseItem = useCallback(async (result: Result, status: "pinned" | "excluded", note?: string) => {
    if (!caseId.trim() || !officer.trim()) { requireIdentity(); return; }
    setBoardMessage(status === "pinned" ? "Pinning evidence…" : "Recording exclusion…");
    const response = await fetch(`${API}/cases/${encodeURIComponent(caseId.trim())}/items/${encodeURIComponent(result.tracklet_id)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        officer: officer.trim(), status, note: note ?? null,
        result_snapshot: {
          ...result,
          search: searchEvidence,
          retrieval: {
            method: searchEvidence?.method, aggregation: searchEvidence?.aggregation,
            composition: searchEvidence?.composition, search_captions: searchEvidence?.search_captions,
            result_score: result.score, component_scores: result.component_scores ?? [],
          },
        },
      }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      setBoardMessage(body?.detail ?? `Could not update case (${response.status})`); return;
    }
    await loadBoard();
    setBoardMessage(status === "pinned" ? "Pinned to the case file." : "Ruled out, with the reason on record.");
  }, [caseId, officer, searchEvidence, loadBoard, requireIdentity]);

  useEffect(() => {
    if (!boardOpen) return;
    loadBoard().catch((e) => setBoardMessage(`Could not load case: ${e instanceof Error ? e.message : "unknown error"}`));
  }, [boardOpen, loadBoard]);

  const onPickImage = useCallback((f: File | null) => {
    if (!f) return;
    setImgFile(f);
    setImgPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(f); });
    runImageSearch(f);
  }, [runImageSearch]);

  // After the first explicit search, re-run whenever a committed filter changes.
  useEffect(() => {
    if (!searched) return;
    if (mode === "photo" && imgFile) runImageSearch(imgFile);
    else runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId, lookingFor, color, timeSet, tFrom, tTo, scene]);

  const selectedCam = cameras.find((c) => c.camera_id === cameraId);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.tagName === "SELECT";
      if (event.key === "/" && !isTyping && mode === "search") {
        event.preventDefault();
        queryRef.current?.focus();
      }
      if (event.key === "Escape") {
        if (active) setActive(null);
        else if (boardOpen) setBoardOpen(false);
        else if (caseOpen) setCaseOpen(false);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [mode, active, boardOpen, caseOpen]);

  const verifyPackage = useCallback(async (file: File | null) => {
    if (!file) return;
    setVerifying(true); setVerification(null);
    try {
      const body = new FormData(); body.append("package", file);
      const response = await fetch(`${API}/forensics/verify`, { method: "POST", body });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setVerification(await response.json());
    } catch (e) {
      setVerification({ valid: false, status: "TAMPERED", errors: [e instanceof Error ? e.message : "verification failed"], warnings: [] });
    } finally { setVerifying(false); }
  }, []);

  const activeChips = useMemo(() => {
    const chips: { label: string; clear: () => void }[] = [];
    if (lookingFor) chips.push({ label: LOOKING_FOR.find((o) => o.value === lookingFor)!.label, clear: () => setLookingFor("") });
    if (selectedCam) chips.push({ label: selectedCam.camera_label, clear: () => setCameraId("") });
    if (timeSet) chips.push({ label: `${tFrom}–${tTo}`, clear: () => setTimeSet(false) });
    if (color) chips.push({ label: `${color} vehicles`, clear: () => setColor("") });
    return chips;
  }, [lookingFor, selectedCam, timeSet, tFrom, tTo, color]);

  const clearFilters = useCallback(() => { setLookingFor(""); setCameraId(""); setTimeSet(false); setColor(""); }, []);

  // rank + strength are computed once from the backend order, so they stay stable
  // no matter how the operator re-sorts the cards afterwards.
  const strengthByTracklet = useMemo(() => {
    const bucket = bucketer(results.map((r) => r.score));
    return new Map(results.map((r) => [r.tracklet_id, strengthOf(r, bucket)]));
  }, [results]);
  const rankByTracklet = useMemo(
    () => new Map(results.map((r, i) => [r.tracklet_id, i + 1])), [results]);

  const displayedResults = useMemo(() => {
    const copy = [...results];
    if (sortMode === "earliest") copy.sort((a, b) => a.ts_start_s - b.ts_start_s);
    if (sortMode === "latest") copy.sort((a, b) => b.ts_start_s - a.ts_start_s);
    if (sortMode === "camera") copy.sort((a, b) => (a.camera_label ?? a.camera_id).localeCompare(b.camera_label ?? b.camera_id));
    return copy;
  }, [results, sortMode]);

  const grouped = useMemo(() => GROUPS.map((g) => ({
    ...g, rows: displayedResults.filter((r) => strengthByTracklet.get(r.tracklet_id) === g.key),
  })).filter((g) => g.rows.length > 0), [displayedResults, strengthByTracklet]);

  // the same bands, but each one holding boxes of look-alikes instead of loose cards
  const banded = useMemo(() => {
    const rankOf = (r: Result) => rankByTracklet.get(r.tracklet_id) ?? Number.MAX_SAFE_INTEGER;
    const strongestOf = (a: Strength, b: Strength) =>
      (GROUPS.findIndex((g) => g.key === a) <= GROUPS.findIndex((g) => g.key === b) ? a : b);

    type Draft = BoxKeys & { stacks: Stack[] };
    const fine = new Map<string, Draft>();
    for (const stack of stackResults(results)) {
      const keys = boxOf(stack.lead);
      const draft = fine.get(keys.key);
      if (draft) draft.stacks.push(stack);
      else fine.set(keys.key, { ...keys, stacks: [stack] });
    }
    // a box of one is not a group — fold it in with its nearest kin
    const merged = new Map<string, Draft>();
    for (const draft of fine.values()) {
      const keepFine = draft.stacks.length > 1;
      const key = keepFine ? draft.key : draft.coarseKey;
      const hit = merged.get(key);
      if (hit) hit.stacks.push(...draft.stacks);
      else merged.set(key, { ...draft, key, label: keepFine ? draft.label : draft.coarseLabel });
    }

    const ordered = [...merged.values()].map<Box>((draft) => {
      const stacks = [...draft.stacks].sort((a, b) => rankOf(a.lead) - rankOf(b.lead));
      return {
        key: draft.key,
        label: draft.label,
        strength: stacks.reduce<Strength>(
          (best, s) => strongestOf(best, strengthByTracklet.get(s.lead.tracklet_id) ?? "moderate"), "weak"),
        stacks,
        count: stacks.reduce((n, s) => n + 1 + s.others.length, 0),
      };
    }).sort((a, b) => rankOf(a.stacks[0]!.lead) - rankOf(b.stacks[0]!.lead));
    return GROUPS.map((g) => ({ ...g, boxes: ordered.filter((box) => box.strength === g.key) }))
      .filter((g) => g.boxes.length > 0);
  }, [results, strengthByTracklet, rankByTracklet]);

  const boxCount = useMemo(() => banded.reduce((n, g) => n + g.boxes.length, 0), [banded]);

  const pinnedIds = useMemo(() => new Set(board?.items.filter((item) => item.status === "pinned").map((item) => item.tracklet_id) ?? []), [board]);

  const applySuggestion = useCallback((action: Suggestion["action"]) => {
    if (action === "expand_time") setTimeSet(false);
    if (action === "remove_color") setColor("");
    if (action === "all_entity_types") setLookingFor("");
  }, []);

  // one click from a card's plate to every sighting of that registration
  const searchPlate = useCallback((plate: string) => {
    setMode("search"); setQ(plate); setForceMode("plate");
    window.scrollTo({ top: 0, behavior: "smooth" });
    runSearch(plate, "plate");
  }, [runSearch]);

  const excludeWithReason = useCallback((r: Result) => {
    const reason = window.prompt("Why is this not the one you want? (kept in the case record)");
    if (reason !== null) setCaseItem(r, "excluded", reason);
  }, [setCaseItem]);

  const renderCard = (r: Result, others: Result[] = []) => (
    <ResultCard
      key={r.tracklet_id} r={r} others={others}
      rank={rankByTracklet.get(r.tracklet_id) ?? 0} outOf={results.length}
      strength={strengthByTracklet.get(r.tracklet_id) ?? "moderate"}
      pinned={pinnedIds.has(r.tracklet_id)}
      onOpen={() => setActive(r)}
      onOpenOther={setActive}
      onPin={() => setCaseItem(r, "pinned")}
      onExclude={() => excludeWithReason(r)}
      onPlate={searchPlate}
    />
  );

  return (
    <main className={styles.main}>
      {/* ===== top bar: context and identity, never search controls ===== */}
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <div className={styles.mark}>
            <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" /></svg>
          </div>
          <div><h1>Drishti</h1><div className={styles.sub}>CCTV search</div></div>
        </div>
        <div className={styles.systemState}>
          <span className={styles.liveDot} />
          <b className={styles.mono}>{total.toLocaleString()}</b> people &amp; vehicles found in
          <b className={styles.mono}>{cameras.length}</b> cameras
        </div>
        <div className={styles.spacer} />
        <div className={styles.scenePick}>
          <span className={styles.sceneIcon}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 7h16v12H4z"/><path d="m8 7 1-3h6l1 3"/><circle cx="12" cy="13" r="3"/></svg></span>
          <span className={styles.headerCopy}>Footage</span>
          <select value={scene} onChange={(e) => setScene(e.target.value)} aria-label="Which footage to search">
            <option value="SUR01">SUR01 · Surat</option>
            <option value="surat-live">surat-live · Surat (live)</option>
            <option value="S01">S01 · CityFlow</option>
          </select>
        </div>
        <CaseChip
          caseId={caseId} officer={officer} pins={pinnedIds.size} open={caseOpen}
          setOpen={setCaseOpen} onCaseId={setCaseId} onOfficer={setOfficer}
          onBoard={() => { setCaseOpen(false); setBoardOpen(true); }}
        />
        <label className={styles.verifyBtn} title="Check that an exported evidence ZIP has not been altered">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l7 3v6c0 4.4-3 7.7-7 9-4-1.3-7-4.6-7-9V6l7-3Z" /><path d="m9 12 2 2 4-4" /></svg>
          {verifying ? "Checking…" : "Check an export"}
          <input type="file" accept=".zip,application/zip" hidden disabled={verifying}
            onChange={(e) => { verifyPackage(e.target.files?.[0] ?? null); e.currentTarget.value = ""; }} />
        </label>
        {verification && (
          <span className={cx(styles.verifyStatus, verification.valid ? styles.valid : styles.tampered)}
            title={[...verification.errors, ...verification.warnings].join("\n") || "All hashes and signature verified"}>
            {verification.valid ? "Untouched" : "Altered"}
          </span>
        )}
      </header>

      <div className={styles.workspace}>
        <section className={styles.intro}>
          <div>
            <h2>What are you looking for?</h2>
            <p>Describe the person or vehicle in your own words, or type a number plate. हिंदी and ગુજરાતી work too.</p>
          </div>
          <div className={styles.kbdHint}>Press <kbd>/</kbd> to start typing</div>
        </section>

        <form
          className={styles.searchrow}
          onSubmit={(e) => { e.preventDefault(); if (mode === "photo") { if (imgFile) runImageSearch(imgFile); } else runSearch(); }}
        >
          <div className={styles.segmented} role="tablist" aria-label="How to search">
            <button type="button" role="tab" aria-selected={mode === "search"} className={cx(mode === "search" && styles.on)} onClick={() => setMode("search")}>Describe in words</button>
            <button type="button" role="tab" aria-selected={mode === "photo"} className={cx(mode === "photo" && styles.on)} onClick={() => setMode("photo")}>Use a photo</button>
          </div>

          <div className={styles.searchline}>
            {mode === "photo" ? (
              <label className={styles.photofield}>
                {imgPreview && <img src={imgPreview} alt="the photo you are searching with" />}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 16l4.6-4.6a2 2 0 0 1 2.8 0L16 16m-2-2 1.6-1.6a2 2 0 0 1 2.8 0L20 14M14 8h.01M6 20h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2Z" /></svg>
                <span>{imgFile ? imgFile.name : "Choose a photo of the person or vehicle"}</span>
                <input type="file" accept="image/*" hidden onChange={(e) => onPickImage(e.target.files?.[0] ?? null)} />
              </label>
            ) : (
              <div className={styles.searchfield}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
                <input
                  ref={queryRef}
                  value={q}
                  onChange={(e) => { setQ(e.target.value); setForceMode(null); }}
                  placeholder="man in an orange shirt"
                  aria-label="What you are looking for"
                  spellCheck={false}
                />
              </div>
            )}
            <button className={styles.go} type="submit" disabled={loading}
              title={identified ? undefined : "You will be asked for a case number and your name first"}>
              {loading ? "Searching…" : "Search"}
            </button>
          </div>

          <div className={styles.filterRow}>
            <span className={styles.filterLabel}>Narrow it down</span>
            <div className={styles.pillGroup} role="group" aria-label="What to look for">
              {LOOKING_FOR.map((o) => (
                <button type="button" key={o.value || "any"} aria-pressed={lookingFor === o.value}
                  className={cx(lookingFor === o.value && styles.on)}
                  onClick={() => setLookingFor(o.value)}>{o.label}</button>
              ))}
            </div>
            <LocationCombo cameras={cameras} total={total} value={cameraId} onChange={setCameraId} />
            <TimeSelect cameras={cameras} from={tFrom} to={tTo} set={timeSet}
              onApply={(f, t) => { setTFrom(f); setTTo(t); setTimeSet(true); }} onClear={() => setTimeSet(false)} />
            <ColorSelect value={color} onChange={setColor} disabled={lookingFor === "person"} />
            {activeChips.length > 0 && <button type="button" className={styles.clearFilters} onClick={clearFilters}>Clear all</button>}
          </div>

          {mode === "search" && (
            <div className={styles.assist}>
              {effMode === "plate" ? (
                <>
                  <span className={styles.plChip}><span className={styles.plTag}>PLATE</span>Reading this as a number plate<span className={styles.faint}> · camera filter applies, time and colour do not</span></span>
                  <button type="button" className={styles.override} onClick={() => setForceMode("describe")}>treat it as a description</button>
                </>
              ) : auto === "plate" && forceMode === "describe" ? (
                <>
                  <span className={cx(styles.plChip, styles.alt)}><span className={styles.plTag}>WORDS</span>Reading this as a description</span>
                  <button type="button" className={styles.override} onClick={() => setForceMode("plate")}>treat it as a number plate</button>
                </>
              ) : searchEvidence?.rewritten_query && searchEvidence.rewritten_query !== searchEvidence.original_query ? (
                <span className={styles.xlate}>
                  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M4 5h7M9 3v2c0 4-2.5 7-5 8M5 9c0 2.5 2.2 4.4 5 5M12 20l4-9 4 9M13.5 17h5" /></svg>
                  Searched as “{searchEvidence.rewritten_query}”
                </span>
              ) : (
                <span className={styles.hintText}>Plain words work best — colour, clothing, vehicle type, “without helmet”.</span>
              )}
              <span className={styles.tryLbl}>Try:</span>
              <div className={styles.examples}>
                {EXAMPLES.map((ex) => (
                  <button type="button" key={ex} className={styles.ex} onClick={() => { setQ(ex); setForceMode(null); runSearch(ex); }}>{ex}</button>
                ))}
              </div>
            </div>
          )}
        </form>

        {/* ===== results ===== */}
        <div className={styles.content}>
          <div className={styles.resultsHead}>
            <div className={styles.resultCount}>
              {loading
                ? "Looking through the footage…"
                : <>
                    <b className={styles.mono}>{results.length}</b> {results.length === 1 ? "result" : "results"}
                    {sortMode === "match" && results.length > 0 &&
                      <span className={styles.faint}> in {boxCount} {boxCount === 1 ? "group" : "groups"}</span>}
                  </>}
            </div>
            {!loading && latencyMs != null && searched && <div className={styles.searchSummary}>found in {(latencyMs / 1000).toFixed(1)}s</div>}
            <div className={styles.activeFilters}>
              {activeChips.map((c) => (
                <span key={c.label} className={styles.afilter}>{c.label} <button onClick={c.clear} aria-label={`stop filtering by ${c.label}`}>✕</button></span>
              ))}
            </div>
            <label className={styles.sortLabel}>Order
              <select value={sortMode} onChange={(event) => setSortMode(event.target.value as typeof sortMode)}>
                <option value="match">Best match, grouped</option>
                <option value="matchflat">Best match, one by one</option>
                <option value="earliest">Earliest time first</option>
                <option value="latest">Latest time first</option>
                <option value="camera">By camera</option>
              </select>
            </label>
            <div className={styles.density} role="group" aria-label="Card size">
              <button type="button" aria-pressed={density === "comfortable"} className={cx(density === "comfortable" && styles.on)} onClick={() => setDensity("comfortable")} title="Big cards">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="3" y="4" width="8" height="7" rx="1.5"/><rect x="13" y="4" width="8" height="7" rx="1.5"/><rect x="3" y="13" width="8" height="7" rx="1.5"/><rect x="13" y="13" width="8" height="7" rx="1.5"/></svg>
              </button>
              <button type="button" aria-pressed={density === "compact"} className={cx(density === "compact" && styles.on)} onClick={() => setDensity("compact")} title="Small cards, more at once">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="3" y="4" width="5" height="4.5" rx="1"/><rect x="9.5" y="4" width="5" height="4.5" rx="1"/><rect x="16" y="4" width="5" height="4.5" rx="1"/><rect x="3" y="9.8" width="5" height="4.5" rx="1"/><rect x="9.5" y="9.8" width="5" height="4.5" rx="1"/><rect x="16" y="9.8" width="5" height="4.5" rx="1"/><rect x="3" y="15.6" width="5" height="4.5" rx="1"/><rect x="9.5" y="15.6" width="5" height="4.5" rx="1"/><rect x="16" y="15.6" width="5" height="4.5" rx="1"/></svg>
              </button>
            </div>
          </div>

          {fellBack && <div className={styles.notice}>Nothing matched those filters exactly, so these are the closest results instead.</div>}
          {context && !loading && !error && <div className={styles.notice}>Showing {context}.</div>}
          {error && !loading && <div className={cx(styles.notice, styles.noticeBad)}>{error}</div>}

          {loading && (
            <div className={cx(styles.grid, density === "compact" && styles.compact)} aria-hidden="true">
              {Array.from({ length: density === "compact" ? 12 : 6 }).map((_, i) => <div key={i} className={styles.skeleton} />)}
            </div>
          )}

          {!loading && !searched && (
            <div className={styles.emptyState}>
              <strong>Nothing searched yet</strong>
              <div>Type what you saw above and press Search. Every result comes with the camera, the time, and the clip.</div>
            </div>
          )}

          {!loading && searched && results.length === 0 && !error && (
            <div className={styles.emptyState}>
              <strong>No results for this search</strong>
              <div>Try fewer words, or loosen a filter:</div>
              {suggestions.length > 0 && (
                <div className={styles.suggestions}>
                  {suggestions.map((s) => (
                    <button type="button" key={s.action} onClick={() => applySuggestion(s.action)}>{s.label}</button>
                  ))}
                </div>
              )}
            </div>
          )}

          {!loading && results.length > 0 && (
            sortMode === "match" ? (
              banded.map((g) => (
                <section key={g.key} className={styles.group}>
                  <div className={cx(styles.groupHead, styles[`gh_${g.key}`])}>
                    <h3><Meter strength={g.key} />{g.title}
                      <span className={styles.groupCount}>{g.boxes.reduce((n, b) => n + b.count, 0)}</span></h3>
                    <p>{g.blurb} <span className={styles.faint}>{g.boxes.length} {g.boxes.length === 1 ? "group" : "groups"} below.</span></p>
                  </div>
                  {g.boxes.map((box) => (
                    <section key={box.key} className={styles.pack}>
                      <div className={styles.packHead}>
                        <span className={styles.packTitle}>{box.label}</span>
                        <span className={styles.packCount}>{box.count}</span>
                        <span className={styles.packNote}>
                          {box.count === 1 ? "one sighting" : `${box.count} sightings that look alike`}
                        </span>
                      </div>
                      <div className={cx(styles.packGrid, styles.grid, density === "compact" && styles.compact)}>
                        {box.stacks.map((stack) => renderCard(stack.lead, stack.others))}
                      </div>
                    </section>
                  ))}
                </section>
              ))
            ) : sortMode === "matchflat" ? (
              grouped.map((g) => (
                <section key={g.key} className={styles.group}>
                  <div className={cx(styles.groupHead, styles[`gh_${g.key}`])}>
                    <h3><Meter strength={g.key} />{g.title}<span className={styles.groupCount}>{g.rows.length}</span></h3>
                    <p>{g.blurb}</p>
                  </div>
                  <div className={cx(styles.grid, density === "compact" && styles.compact)}>
                    {g.rows.map((r) => renderCard(r))}
                  </div>
                </section>
              ))
            ) : (
              <div className={cx(styles.grid, density === "compact" && styles.compact)}>
                {displayedResults.map((r) => renderCard(r))}
              </div>
            )
          )}
        </div>
      </div>

      {active && <Player result={active} search={searchEvidence} caseId={caseId} officer={officer}
        strength={strengthByTracklet.get(active.tracklet_id) ?? "moderate"}
        onClose={() => setActive(null)} onSimilar={runSimilar} onCaseItem={setCaseItem} />}
      {boardOpen && <CaseBoard board={board} timeline={timeline} caseId={caseId} officer={officer}
        message={boardMessage} onClose={() => setBoardOpen(false)} onRefresh={loadBoard}
        onOpen={(item) => { setBoardOpen(false); setActive({ ...(item.result_snapshot ?? {}), ...item } as Result); }}
        onUpdate={setCaseItem} />}
    </main>
  );
}

/* ============================ result card ============================ */
function Meter({ strength }: { strength: Strength }) {
  const filled = strength === "strong" ? 3 : strength === "moderate" ? 2 : 1;
  return (
    <span className={styles.meter} aria-hidden="true">
      {[0, 1, 2].map((i) => <i key={i} className={cx(i < filled && styles.lit)} />)}
    </span>
  );
}

function ResultCard({ r, others, rank, outOf, strength, pinned, onOpen, onOpenOther, onPin, onExclude, onPlate }: {
  r: Result; others: Result[]; rank: number; outOf: number; strength: Strength; pinned: boolean;
  onOpen: () => void; onOpenOther: (r: Result) => void;
  onPin: () => void; onExclude: () => void; onPlate: (plate: string) => void;
}) {
  // people have no stored colour, so the clothing the VLM committed to names them instead
  const worn = isPerson(r) ? r.attrs?.find((a) => a.endsWith(" top")) : undefined;
  const title = capitalise(worn ? `person in ${an(worn)}` : [r.color, titleise(r.subtype)].filter(Boolean).join(" "));
  const verdict = strengthCopy(r, strength);
  // only things that carry a registration get a plate line — not people, not bicycles
  const platable = !isPerson(r) && r.subtype !== "bicycle";
  const plateTitle = r.plate_conf != null
    ? `Number plate read by OCR · confidence ${Math.round(r.plate_conf * 100)}%`
    : "Number plate read by OCR";
  return (
    <article className={cx(styles.card, pinned && styles.pinned)}>
      <button type="button" className={styles.cardOpen} onClick={onOpen} aria-label={`Watch the clip for this ${title}`}>
        <div className={styles.shot}>
          {r.crop_url ? (
            <>
              <img className={styles.shotBack} src={`${API}${r.crop_url}`} alt="" aria-hidden="true" loading="lazy" />
              <img className={styles.shotImg} src={`${API}${r.crop_url}`} alt={`${title} seen on ${r.camera_label ?? r.camera_id}`} loading="lazy" />
            </>
          ) : <div className={styles.shotEmpty}>No still image</div>}
          <div className={styles.shotBadges}>
            <span className={cx(styles.verdict, styles[strength])} title={verdict}><Meter strength={strength} />{strengthShort(r, strength)}</span>
            {r.plate && <span className={styles.plateBadge} title={plateTitle}>{r.plate}</span>}
          </div>
          <span className={styles.playHint}>
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>Watch clip
          </span>
          <span className={styles.osd}>{fmt(r.ts_start_s)}</span>
        </div>
      </button>

      <div className={styles.cardBody}>
        <h4 className={styles.cardTitle}>{title}</h4>
        <dl className={styles.facts}>
          {platable && (
            <div><dt>Plate</dt><dd>
              {r.plate
                ? <button type="button" className={styles.plateFind} title={`${plateTitle} — click to find every sighting of it`}
                    onClick={() => onPlate(r.plate!)}>
                    <span className={styles.plateBadge}>{r.plate}</span>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
                    find all sightings
                  </button>
                : <span className={styles.faint}>not readable in this footage</span>}
            </dd></div>
          )}
          <div><dt>Time</dt><dd className={styles.mono}>{fmt(r.ts_start_s)} – {fmt(r.ts_end_s)}<span className={styles.faint}> · {fmtSpan(r.ts_start_s, r.ts_end_s)}</span></dd></div>
          <div><dt>Place</dt><dd>{r.camera_label ?? r.camera_id}</dd></div>
          <div className={styles.detailFact}><dt>Camera</dt><dd className={styles.mono}>{r.camera_id}<span className={styles.faint}> · {r.scene}</span></dd></div>
          <div className={styles.detailFact}><dt>Match</dt><dd className={styles[`t_${strength}`]}>{verdict}{rank > 0 && <span className={styles.faint}> · #{rank} of {outOf}</span>}</dd></div>
        </dl>
        {others.length > 0 && (
          <div className={styles.strip}>
            <span className={styles.stripLbl}>Same one, also at</span>
            {others.map((o) => (
              <button type="button" key={o.tracklet_id} className={styles.stripBtn}
                onClick={() => onOpenOther(o)}
                title={`Watch this sighting · ${o.camera_label ?? o.camera_id} · ${fmt(o.ts_start_s)}`}>
                {o.crop_url && <img src={`${API}${o.crop_url}`} alt="" loading="lazy" />}
                <span className={styles.mono}>{fmt(o.ts_start_s)}</span>
              </button>
            ))}
          </div>
        )}
        {((r.sightings ?? 0) > 1 || (r.attrs && r.attrs.length > 0)) && (
          <div className={styles.tagrow}>
            {(r.sightings ?? 0) > 1 && <span className={styles.tag}>seen {r.sightings} times</span>}
            {r.attrs?.slice(0, 5).map((a) => <span key={a} className={styles.tag}>{a}</span>)}
          </div>
        )}
      </div>

      <div className={styles.cardActions}>
        <button type="button" className={styles.primaryAction} onClick={onOpen}>Watch clip</button>
        <button type="button" className={pinned ? styles.pinnedAction : undefined} onClick={onPin}>{pinned ? "✓ In case file" : "Add to case"}</button>
        <button type="button" onClick={onExclude}>Not this one</button>
      </div>
    </article>
  );
}

/* ============================ case identity ============================ */
function CaseChip({ caseId, officer, pins, open, setOpen, onCaseId, onOfficer, onBoard }: {
  caseId: string; officer: string; pins: number; open: boolean;
  setOpen: (v: boolean) => void; onCaseId: (v: string) => void; onOfficer: (v: string) => void;
  onBoard: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const ready = Boolean(caseId.trim() && officer.trim());

  useEffect(() => {
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [setOpen]);
  useEffect(() => { if (open && !officer.trim()) nameRef.current?.focus(); }, [open, officer]);

  return (
    <div className={styles.caseWrap} ref={ref}>
      <button type="button" className={cx(styles.caseChip, !ready && styles.caseChipWarn)}
        onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={styles.caseAvatar}>{(officer.trim()[0] ?? "?").toUpperCase()}</span>
        <span className={styles.caseText}>
          <b>{caseId.trim() || "No case number"}</b>
          <i>{officer.trim() || "Add your name to search"}</i>
        </span>
        {pins > 0 && <span className={styles.casePins}>{pins}</span>}
      </button>
      {open && (
        <div className={styles.casePop}>
          <h4>Who is searching?</h4>
          <p>Searches, pins and exports are written to the case file, so both of these are required.</p>
          <label>Case number<input value={caseId} maxLength={120} onChange={(e) => onCaseId(e.target.value)} /></label>
          <label>Your name or badge<input ref={nameRef} value={officer} maxLength={120} placeholder="e.g. PSI Patel · 4471" onChange={(e) => onOfficer(e.target.value)} /></label>
          <div className={styles.casePopActions}>
            <button type="button" onClick={onBoard}>Open case file{pins > 0 ? ` · ${pins}` : ""}</button>
            <button type="button" className={styles.apply} disabled={!ready} onClick={() => setOpen(false)}>Done</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================ colour picker ============================ */
function ColorSelect({ value, onChange, disabled }: {
  value: string; onChange: (v: string) => void; disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);
  useEffect(() => { if (disabled) setOpen(false); }, [disabled]);

  const current = COLORS.find((c) => c.name === value);
  return (
    <div className={cx(styles.pill, styles.colorPill, open && styles.open, disabled && styles.pillOff)} ref={ref}>
      <button type="button" className={styles.pillBtn} disabled={disabled} onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        title={disabled ? "Colour is stored for vehicles. For clothing, describe it instead — “orange shirt”." : undefined}>
        <span className={cx(styles.swatch, !current && styles.swatchAny)} style={current ? { background: current.swatch } : undefined} />
        <span className={styles.pillValue}>{current ? `${current.name} vehicles` : "Any colour"}</span>
        <svg className={styles.chev} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
      </button>
      {open && (
        <div className={styles.colorPop}>
          <h4>Vehicle colour</h4>
          <div className={styles.swatches}>
            <button type="button" className={cx(styles.swatchOpt, !value && styles.sel)} onClick={() => { onChange(""); setOpen(false); }}>
              <span className={cx(styles.swatch, styles.swatchAny)} />Any
            </button>
            {COLORS.map((c) => (
              <button type="button" key={c.name} className={cx(styles.swatchOpt, value === c.name && styles.sel)}
                onClick={() => { onChange(c.name); setOpen(false); }}>
                <span className={styles.swatch} style={{ background: c.swatch }} />{c.name}
              </button>
            ))}
          </div>
          <p className={styles.popNote}>Colour is recorded for vehicles only. For clothing, put it in your description.</p>
        </div>
      )}
    </div>
  );
}

/* ============================ location combobox ============================ */
function LocationCombo({ cameras, total, value, onChange }: {
  cameras: Camera[]; total: number; value: string; onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [typing, setTyping] = useState(false);
  const [hi, setHi] = useState(-1);
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selected = cameras.find((c) => c.camera_id === value);
  useEffect(() => { if (!open) { setText(selected?.camera_label ?? ""); setTyping(false); } }, [value, open, selected]);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const opts = useMemo(() => {
    const cov = cameras.length ? `${secToHHMM(Math.min(...cameras.map((c) => c.ts_start_s)))}–${secToHHMM(Math.max(...cameras.map((c) => c.ts_end_s)))}` : "";
    const all = [
      { camera_id: "", camera_label: "All cameras", count: total, cov, whole: true },
      ...cameras.map((c) => ({ camera_id: c.camera_id, camera_label: c.camera_label, count: c.count, cov: `${secToHHMM(c.ts_start_s)}–${secToHHMM(c.ts_end_s)}`, whole: false })),
    ];
    const q = typing ? text.trim().toLowerCase() : "";
    return q ? all.filter((o) => o.camera_label.toLowerCase().includes(q)) : all;
  }, [cameras, total, typing, text]);

  const highlight = (name: string) => {
    const q = typing ? text.trim().toLowerCase() : "";
    if (!q) return name;
    const i = name.toLowerCase().indexOf(q);
    return i < 0 ? name : <>{name.slice(0, i)}<mark>{name.slice(i, i + q.length)}</mark>{name.slice(i + q.length)}</>;
  };

  const pick = (id: string) => { onChange(id); setOpen(false); setTyping(false); };

  return (
    <div className={cx(styles.pill, styles.combo, open && styles.open, value && styles.pillSet)} ref={ref}>
      <div className={styles.comboField} onClick={() => { if (!open) { setOpen(true); setHi(-1); } inputRef.current?.focus(); }}>
        <svg className={styles.pin} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z" /><circle cx="12" cy="10" r="3" /></svg>
        <input
          ref={inputRef} value={text} placeholder="All cameras" autoComplete="off" spellCheck={false}
          aria-label="Which camera to search"
          onChange={(e) => { setText(e.target.value); setTyping(true); setHi(-1); setOpen(true); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setHi((h) => Math.min(h + 1, opts.length - 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setHi((h) => Math.max(h - 1, 0)); }
            else if (e.key === "Enter") { e.preventDefault(); const o = opts[hi] ?? opts[0]; if (o) pick(o.camera_id); }
            else if (e.key === "Escape") { setOpen(false); inputRef.current?.blur(); }
          }}
        />
        <svg className={styles.chev} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}><path d="m6 9 6 6 6-6" /></svg>
      </div>
      {open && (
        <div className={styles.comboMenu}>
          {opts.length === 0 ? (
            <div className={styles.comboEmpty}>No camera called “{text.trim()}”</div>
          ) : opts.map((o, i) => (
            <div key={o.camera_id || "all"} className={cx(styles.comboOpt, o.camera_id === value && styles.sel, i === hi && styles.hi)}
              onMouseDown={(e) => { e.preventDefault(); pick(o.camera_id); }}>
              <svg className={styles.ico} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z" /><circle cx="12" cy="10" r="3" /></svg>
              <span className={styles.nm}>{highlight(o.camera_label)}</span>
              <span className={styles.cov}>{o.cov}{o.whole ? " · whole scene" : ""}</span>
              <span className={styles.cnt}>{o.count.toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ============================ time-of-day select ============================ */
function TimeSelect({ cameras, from, to, set, onApply, onClear }: {
  cameras: Camera[]; from: string; to: string; set: boolean;
  onApply: (f: string, t: string) => void; onClear: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [f, setF] = useState(from);
  const [t, setT] = useState(to);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { setF(from); setT(to); }, [from, to, open]);
  useEffect(() => {
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const span0 = cameras.length ? Math.min(...cameras.map((c) => c.ts_start_s)) : 0;
  const span1 = cameras.length ? Math.max(...cameras.map((c) => c.ts_end_s)) : 86400;
  const pos = (s: number) => Math.max(0, Math.min(1, (s - span0) / (span1 - span0 || 1)));
  const bandL = pos(hhmmToSec(f)), bandR = pos(hhmmToSec(t));

  return (
    <div className={cx(styles.pill, styles.timeselect, open && styles.open, set && styles.pillSet)} ref={ref}>
      <button type="button" className={styles.pillBtn} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>
        <span className={cx(styles.pillValue, set && styles.mono)}>{set ? `${from} – ${to}` : "Any time"}</span>
        <svg className={styles.chev} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
      </button>
      {open && (
        <div className={styles.timePop}>
          <h4>Time of day</h4>
          <div className={styles.fromto}>
            <input value={f} spellCheck={false} aria-label="from time" onChange={(e) => setF(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); onApply(f, t); setOpen(false); } }} />
            <span>to</span>
            <input value={t} spellCheck={false} aria-label="to time" onChange={(e) => setT(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); onApply(f, t); setOpen(false); } }} />
          </div>
          <div className={styles.cov2}>
            {cameras.map((c) => (
              <div key={c.camera_id} className={styles.cluster} style={{ left: `${pos(c.ts_start_s) * 100}%`, width: `${Math.max(1, (pos(c.ts_end_s) - pos(c.ts_start_s)) * 100)}%` }} />
            ))}
            <div className={styles.selband} style={{ left: `${bandL * 100}%`, width: `${Math.max(1.5, (bandR - bandL) * 100)}%` }} />
            <div className={styles.covTick} style={{ left: "2%" }}>{secToHHMM(span0)}</div>
            <div className={styles.covTick} style={{ right: "2%" }}>{secToHHMM(span1)}</div>
          </div>
          <div className={styles.covHint}>There is only footage inside the lit bands — pick a time inside one.</div>
          <div className={styles.presets}>
            <button type="button" className={styles.preset} onClick={() => { setF("00:00"); setT("23:59"); }}><span>Whole day</span><span className={styles.w}>any</span></button>
            {cameras.map((c) => (
              <button type="button" key={c.camera_id} className={styles.preset}
                onClick={() => { setF(secToHHMM(c.ts_start_s)); setT(secToHHMM(c.ts_end_s)); }}>
                <span>{c.camera_label.split(" – ")[0]}</span>
                <span className={styles.w}>{secToHHMM(c.ts_start_s)}–{secToHHMM(c.ts_end_s)}</span>
              </button>
            ))}
          </div>
          <div className={styles.popActions}>
            <button type="button" className={styles.clear} onClick={() => { onClear(); setOpen(false); }}>Any time</button>
            <button type="button" className={styles.apply} onClick={() => { onApply(f, t); setOpen(false); }}>Use this time</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================ player + box overlay ============================ */
type BoxRow = [number, number, number, number, number];
function boxIndexAt(boxes: BoxRow[], t: number): number {
  let lo = 0, hi = boxes.length - 1, ans = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (boxes[mid]![0] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
  return ans;
}

function Player({ result, search, caseId, officer, strength, onClose, onSimilar, onCaseItem }: {
  result: Result; search: SearchEvidenceContext | null; caseId: string; officer: string;
  strength: Strength; onClose: () => void;
  onSimilar: (r: Result, vec: "semantic" | "reid") => void;
  onCaseItem: (r: Result, status: "pinned" | "excluded", note?: string) => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const boxRef = useRef<SVGRectElement>(null);
  const [boxes, setBoxes] = useState<BoxRow[] | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [explainCrop, setExplainCrop] = useState<string | null>(result.crop_url);
  const src = result.video_url ? `${API}${result.video_url}` : null;
  const seek0 = result.video_start_s ?? result.ts_start_s;
  const seek1 = result.video_end_s ?? result.ts_end_s;

  useEffect(() => {
    let ok = true; setBoxes(null);
    fetch(`${API}/tracklets/${result.tracklet_id}/boxes`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => ok && setBoxes(d?.boxes ?? null))
      .catch(() => ok && setBoxes(null));
    return () => { ok = false; };
  }, [result.tracklet_id]);

  useEffect(() => {
    if (!boxes || boxes.length === 0) return;
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const v = ref.current, rect = boxRef.current;
      if (!v || !rect || !v.videoWidth) return;
      const t = v.currentTime, i = boxIndexAt(boxes, t), cur = i >= 0 ? boxes[i] : undefined;
      const GAP = 0.5;
      if (!cur || t - cur[0] > GAP) { rect.setAttribute("visibility", "hidden"); return; }
      const t0 = cur[0];
      let [, x1, y1, x2, y2] = cur;
      const nxt = boxes[i + 1];
      if (nxt && nxt[0] > t0 && nxt[0] - t0 <= GAP) {
        const a = (t - t0) / (nxt[0] - t0);
        x1 += (nxt[1] - x1) * a; y1 += (nxt[2] - y1) * a; x2 += (nxt[3] - x2) * a; y2 += (nxt[4] - y2) * a;
      }
      const scale = Math.min(v.clientWidth / v.videoWidth, v.clientHeight / v.videoHeight);
      const ox = (v.clientWidth - v.videoWidth * scale) / 2, oy = (v.clientHeight - v.videoHeight * scale) / 2;
      rect.setAttribute("x", String(ox + x1 * scale)); rect.setAttribute("y", String(oy + y1 * scale));
      rect.setAttribute("width", String(Math.max(0, (x2 - x1) * scale))); rect.setAttribute("height", String(Math.max(0, (y2 - y1) * scale)));
      rect.setAttribute("visibility", "visible");
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [boxes]);

  const title = [result.color, titleise(result.subtype)].filter(Boolean).join(" ");

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.player} onClick={(e) => e.stopPropagation()} role="dialog" aria-label={`Clip for ${title}`}>
        <div className={styles.playerHead}>
          <strong className={styles.playerTitle}>{title}</strong>
          {result.plate && <span className={styles.plateBadge}
            title={result.plate_conf != null ? `Number plate read by OCR · confidence ${Math.round(result.plate_conf * 100)}%` : "Number plate read by OCR"}>{result.plate}</span>}
          <span>
            {result.camera_label ?? result.camera_id} · {fmt(result.ts_start_s)}–{fmt(result.ts_end_s)}
            {result.video_start_s != null && ` · ${fmtClip(result.video_start_s)} into the clip`}
            <span className={styles.faint}> · {result.tracklet_id}</span>
          </span>
          <button className={styles.simBtn} onClick={() => onSimilar(result, "semantic")} title="objects that look like this one">Find look-alikes</button>
          <button className={styles.simBtn} onClick={() => onSimilar(result, "reid")} title="other sightings of this exact object">Find this one again</button>
          <button className={styles.simBtn} onClick={() => onCaseItem(result, "pinned")}>Add to case</button>
          <button className={styles.exportBtn} onClick={() => setExportOpen((v) => !v)}>Export evidence</button>
          <button className={styles.close} onClick={onClose} aria-label="Close">✕</button>
        </div>
        {exportOpen && <ExportPanel result={result} search={search} caseId={caseId} officer={officer} />}
        {/* the clip is what the operator opened this for — it goes above the reasoning,
            so the box tracking the object is on screen without scrolling */}
        {src ? (
          <div className={styles.videoWrap}>
            <video ref={ref} className={styles.video} src={src} controls autoPlay
              onLoadedMetadata={() => { if (ref.current) ref.current.currentTime = seek0; }}
              onTimeUpdate={() => { const v = ref.current; if (v && v.currentTime >= seek1) v.currentTime = seek0; }} />
            {boxes && boxes.length > 0 && <svg className={styles.boxSvg}><rect ref={boxRef} className={styles.boxRect} visibility="hidden" rx={3} /></svg>}
          </div>
        ) : <div className={styles.status}>There is no clip stored for this sighting.</div>}
        <div className={styles.whyPanel}>
          <div className={styles.whyCopy}>
            <strong>Why this came up</strong>
            <span>Match strength is relative to the other results of this search — not a probability.</span>
            <div className={styles.whyFacts}>
              <span className={styles[`t_${strength}`]}><Meter strength={strength} /> {strengthCopy(result, strength)}</span>
              <span>✓ {result.entity_type ?? result.subtype} · confirmed by the detector</span>
              <span>📍 {result.camera_label ?? result.camera_id}</span>
              <span>◷ {fmt(result.ts_start_s)}–{fmt(result.ts_end_s)}</span>
            </div>
            <div className={styles.components}>
              {result.component_scores?.map((component, index) => (
                <button type="button" key={`${component.caption}-${index}`}
                  className={cx(component.supporting_crop_url === explainCrop && styles.selected)}
                  onClick={() => setExplainCrop(component.supporting_crop_url ?? result.crop_url)}>
                  <span>{component.kind === "overall" ? "Whole description" : `✓ ${component.caption}`}</span>
                  <b className={styles[component.match_strength ?? "moderate"]}>{component.match_strength ?? "ranked"}</b>
                  <code>{component.score.toFixed(3)}</code>
                </button>
              )) ?? <span>Matched on the number plate, not on a description.</span>}
            </div>
          </div>
          {explainCrop && <img src={`${API}${explainCrop}`} alt="the still image behind the selected part of the match" />}
        </div>
        {result.global_id != null && <TraceView scene={result.scene} globalId={result.global_id} />}
      </div>
    </div>
  );
}

function ExportPanel({ result, search, caseId, officer }: {
  result: Result; search: SearchEvidenceContext | null; caseId: string; officer: string;
}) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const create = async () => {
    if (!caseId.trim() || !officer.trim()) {
      setMessage("A case number and your name are needed for the chain of custody.");
      return;
    }
    setBusy(true); setMessage("Copying the source clip, building the labelled files, signing the package…");
    try {
      const response = await fetch(`${API}/forensics/export`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tracklet_id: result.tracklet_id,
          case_id: caseId,
          officer,
          selected_crop_ref: result.matched_crop_ref,
          search: search ? {
            original_query: search.original_query,
            rewritten_query: search.rewritten_query,
            timestamp_utc: search.timestamp_utc,
            filters: search.filters,
          } : {},
          retrieval: {
            method: search?.method ?? "unspecified",
            aggregation: search?.aggregation,
            composition: search?.composition,
            search_captions: search?.search_captions ?? [],
            result_score: result.score,
            component_scores: result.component_scores ?? [],
          },
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? `HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") ?? "";
      const filename = disposition.match(/filename="?([^";]+)"?/)?.[1] ?? `${caseId || "case-export"}.zip`;
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a"); anchor.href = href; anchor.download = filename;
      document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(href);
      setMessage(`Signed and downloaded · ${response.headers.get("X-Forensic-Export-ID") ?? filename}`);
    } catch (e) {
      setMessage(`Export failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally { setBusy(false); }
  };

  return (
    <div className={styles.exportPanel}>
      <div><strong>Forensic export</strong><span>The source recording stays unmarked; the copies are labelled and signed.</span></div>
      <label>Case number<input value={caseId} readOnly /></label>
      <label>Officer<input value={officer} readOnly /></label>
      <button type="button" disabled={busy} onClick={create}>{busy ? "Building package…" : "Create signed ZIP"}</button>
      {message && <p>{message}</p>}
    </div>
  );
}

function CaseBoard({ board, timeline, caseId, officer, message, onClose, onRefresh, onOpen, onUpdate }: {
  board: CaseBoardData | null; timeline: AuditEvent[]; caseId: string; officer: string;
  message: string | null; onClose: () => void; onRefresh: () => Promise<void>;
  onOpen: (item: CaseItem) => void;
  onUpdate: (result: Result, status: "pinned" | "excluded", note?: string) => Promise<void>;
}) {
  const [tab, setTab] = useState<"evidence" | "timeline">("evidence");
  const [exporting, setExporting] = useState(false);
  const pinned = board?.items.filter((item) => item.status === "pinned") ?? [];
  const excluded = board?.items.filter((item) => item.status === "excluded") ?? [];

  const exportPinned = async () => {
    setExporting(true);
    try {
      const response = await fetch(`${API}/cases/${encodeURIComponent(caseId)}/export`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ officer }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? `HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") ?? "";
      const filename = disposition.match(/filename="?([^";]+)"?/)?.[1] ?? `${caseId}-bundle.zip`;
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a"); anchor.href = href; anchor.download = filename;
      document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(href);
      await onRefresh();
    } catch (error) {
      window.alert(`Case export failed: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally { setExporting(false); }
  };

  return (
    <div className={styles.boardOverlay} onClick={onClose}>
      <aside className={styles.board} onClick={(event) => event.stopPropagation()} role="dialog" aria-label="Case file">
        <header>
          <div><strong>{caseId}</strong><span>{officer || "Officer not set"}</span></div>
          <button type="button" onClick={onClose} aria-label="Close">✕</button>
        </header>
        <div className={styles.boardTabs}>
          <button className={cx(tab === "evidence" && styles.on)} onClick={() => setTab("evidence")}>Evidence {pinned.length}</button>
          <button className={cx(tab === "timeline" && styles.on)} onClick={() => setTab("timeline")}>What was done {timeline.length}</button>
          <button className={styles.bundleBtn} disabled={exporting || pinned.length === 0} onClick={exportPinned}>
            {exporting ? "Signing…" : "Export all"}
          </button>
        </div>
        {message && <div className={styles.boardMessage}>{message}</div>}
        {tab === "evidence" ? (
          <div className={styles.boardList}>
            {!board && <div className={styles.status}>Loading case…</div>}
            {pinned.map((item) => <CaseItemRow key={item.tracklet_id} item={item} onOpen={onOpen} onUpdate={onUpdate} />)}
            {excluded.length > 0 && <h4>Ruled out · kept on record</h4>}
            {excluded.map((item) => <CaseItemRow key={item.tracklet_id} item={item} onOpen={onOpen} onUpdate={onUpdate} />)}
            {board && board.items.length === 0 && <div className={styles.status}>Add a result to the case file and it will appear here.</div>}
          </div>
        ) : (
          <div className={styles.auditList}>
            {[...timeline].reverse().map((event) => (
              <div key={event.event_id} className={styles.auditEvent}>
                <i /><div><strong>{event.event_type.replaceAll("_", " ")}</strong>
                  <span>{new Date(event.occurred_at).toLocaleString()} · {event.officer}</span>
                  {event.original_query && <p>“{event.original_query}”</p>}
                  {event.latency_ms != null && <code>{Math.round(event.latency_ms)} ms</code>}
                </div>
              </div>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}

function CaseItemRow({ item, onOpen, onUpdate }: {
  item: CaseItem; onOpen: (item: CaseItem) => void;
  onUpdate: (result: Result, status: "pinned" | "excluded", note?: string) => Promise<void>;
}) {
  const [note, setNote] = useState(item.note ?? "");
  useEffect(() => setNote(item.note ?? ""), [item.note]);
  return (
    <div className={cx(styles.caseItem, item.status === "excluded" && styles.excluded)}>
      <button type="button" className={styles.caseThumb} onClick={() => onOpen(item)}>
        {item.crop_url && <img src={`${API}${item.crop_url}`} alt={item.subtype} />}
      </button>
      <div className={styles.caseMeta}>
        <strong>{titleise(item.subtype)} · {item.camera_label ?? item.camera_id}</strong>
        {item.plate && <span className={styles.plateBadge}>{item.plate}</span>}
        <span>{fmt(item.ts_start_s)}–{fmt(item.ts_end_s)} · {item.tracklet_id}</span>
        <textarea value={note} placeholder={item.status === "excluded" ? "Why it was ruled out" : "Your note"}
          onChange={(event) => setNote(event.target.value)} />
        <div>
          <button type="button" onClick={() => onUpdate(item, item.status, note)}>Save note</button>
          {item.status === "excluded" && <button type="button" onClick={() => onUpdate(item, "pinned", note)}>Put back in evidence</button>}
          {item.status === "pinned" && <button type="button" onClick={() => onUpdate(item, "excluded", note || "Ruled out during case review")}>Rule out</button>}
        </div>
      </div>
    </div>
  );
}

function TraceView({ scene, globalId }: { scene: string; globalId: number }) {
  const [hops, setHops] = useState<Hop[] | null>(null);
  useEffect(() => {
    let ok = true;
    fetch(`${API}/trace/${scene}/${globalId}`).then((r) => r.json()).then((d) => ok && setHops(d.hops ?? [])).catch(() => ok && setHops([]));
    return () => { ok = false; };
  }, [scene, globalId]);
  if (!hops) return <div className={styles.status} style={{ padding: "10px 16px" }}>Loading path…</div>;

  const positions = CAM_POS[scene] ?? {};
  const pts: [number, number][] = [];
  for (const h of hops) { const p = positions[h.camera_id]; if (!p) continue; const last = pts[pts.length - 1]; if (!last || last[0] !== p[0] || last[1] !== p[1]) pts.push(p); }
  const camsInPath = new Set(hops.map((h) => h.camera_id));
  const single = hops.length <= 1;

  return (
    <div className={styles.trace}>
      <div className={styles.traceHead}>
        {single ? "Seen on one camera only — no path across cameras."
          : camsInPath.size === 1 ? `${hops.length} sightings on this camera (joined up by number plate)`
            : `Path across cameras · ${hops.length} sightings on ${camsInPath.size} cameras`}
      </div>
      <div className={styles.traceBody} style={Object.keys(positions).length === 0 ? { gridTemplateColumns: "1fr" } : undefined}>
        {Object.keys(positions).length > 0 && (
          <div className={styles.mapWrap}>
            <img className={styles.mapImg} src={`${API}/scene-map/${scene}`} alt="map of the cameras in this scene" />
            <svg className={styles.mapSvg} viewBox="0 0 100 100" preserveAspectRatio="none">
              {pts.length > 1 && <polyline points={pts.map(([x, y]) => `${x * 100},${y * 100}`).join(" ")} fill="none" stroke="#4c8bf5" strokeWidth={1.1} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />}
              {Object.entries(positions).map(([cam, [x, y]]) => (
                <circle key={cam} cx={x * 100} cy={y * 100} r={1.8} fill={camsInPath.has(cam) ? "#4c8bf5" : "#555"} stroke="#0b0c0f" strokeWidth={0.5} vectorEffect="non-scaling-stroke" />
              ))}
            </svg>
          </div>
        )}
        <div className={styles.timeline}>
          {hops.map((h, i) => (
            <div key={`${h.tracklet_id}-${i}`} className={styles.hop}>
              {h.crop_url && <img className={styles.hopCrop} src={`${API}${h.crop_url}`} alt={h.camera_id} />}
              <div className={styles.hopMeta}>
                <strong>{(h.camera_label ?? h.camera_id).slice(0, 9)}</strong>
                <span>{fmt(h.ts_start_s)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
