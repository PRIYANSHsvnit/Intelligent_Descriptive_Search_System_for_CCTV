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
  component_scores?: { caption: string; score: number }[];
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

// describe/plate examples share one bar — clicking a plate one flips the detected mode
const EXAMPLES = ["person with a backpack", "woman in a red saree", "rider without helmet", "GJ05JP4237", "4237"];

// scene-clock seconds -> time of day h:mm:ss (SUR01 offsets are seconds since midnight)
function fmt(t: number): string {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}
function fmtClip(t: number): string {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}
const pad2 = (n: number) => String(n).padStart(2, "0");
const secToHHMM = (s: number) => `${pad2(Math.floor(s / 3600) % 24)}:${pad2(Math.floor((s % 3600) / 60))}`;
const hhmmToSec = (s: string) => { const [h, m] = s.split(":").map(Number); return (h || 0) * 3600 + (m || 0) * 60; };

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

  const auto = detectMode(q);
  const effMode: "plate" | "describe" = forceMode ?? auto;

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

  const runSearch = useCallback(async (queryArg?: string) => {
    const query = queryArg ?? q;
    if (!query.trim()) return;
    const em = forceMode ?? detectMode(query);
    setLoading(true); setSearched(true); setContext(null); setError(null);
    try {
      const p = new URLSearchParams({ limit: "24" });
      if (em === "plate") p.set("plate", query.trim());
      else p.set("q", query);
      if (scene) p.set("scene", scene);
      if (cameraId) p.set("camera_id", cameraId);
      if (timeSet) { p.set("t0", String(hhmmToSec(tFrom))); p.set("t1", String(hhmmToSec(tTo))); }
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
      setLoading(false);
    }
  }, [q, forceMode, scene, cameraId, timeSet, tFrom, tTo]);

  const runImageSearch = useCallback(async (file: File) => {
    setLoading(true); setSearched(true); setContext("matches for the uploaded photo"); setError(null);
    try {
      const fd = new FormData();
      fd.append("image", file);
      if (scene) fd.append("scene", scene);
      if (cameraId) fd.append("camera_id", cameraId);
      if (timeSet) { fd.append("t0", String(hhmmToSec(tFrom))); fd.append("t1", String(hhmmToSec(tTo))); }
      fd.append("limit", "24");
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
      setLoading(false);
    }
  }, [scene, cameraId, timeSet, tFrom, tTo]);

  const runSimilar = useCallback(async (r: Result, vec: "semantic" | "reid") => {
    setActive(null); setLoading(true); setSearched(true);
    setContext(vec === "reid"
      ? `re-ID matches for ${r.subtype} ${r.tracklet_id} (same object, other sightings)`
      : `tracklets that look similar to ${r.subtype} ${r.tracklet_id}`);
    setError(null);
    try {
      const p = new URLSearchParams({ tracklet_id: r.tracklet_id, vec, limit: "24" });
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
        setContext(`look-alikes of ${r.subtype} ${r.tracklet_id} (no re-ID vector stored — visual similarity only)`);
    } catch (e) {
      setResults([]); setSuggestions([]); setError(`Similar search failed (${e instanceof Error ? e.message : "network"}).`);
    } finally { setLoading(false); }
  }, []);

  const onPickImage = useCallback((f: File | null) => {
    if (!f) return;
    setImgFile(f);
    setImgPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(f); });
    runImageSearch(f);
  }, [runImageSearch]);

  // re-run search whenever a committed FILTER changes (not while typing q); also on mount.
  useEffect(() => {
    if (mode === "photo" && imgFile) runImageSearch(imgFile);
    else runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId, timeSet, tFrom, tTo, scene]);

  const selectedCam = cameras.find((c) => c.camera_id === cameraId);

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
    if (selectedCam) chips.push({ label: selectedCam.camera_label, clear: () => setCameraId("") });
    if (timeSet) chips.push({ label: `${tFrom}–${tTo}`, clear: () => setTimeSet(false) });
    return chips;
  }, [selectedCam, timeSet, tFrom, tTo]);

  return (
    <main className={styles.main}>
      {/* ===== top bar ===== */}
      <div className={styles.topbar}>
        <div className={styles.brand}>
          <div className={styles.mark}>
            <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" /></svg>
          </div>
          <div>
            <h1>Descriptive Search</h1>
            <div className={styles.sub}>Surat City CCTV · Forensic Console</div>
          </div>
        </div>
        <div className={styles.spacer} />
        <label className={styles.verifyBtn}>
          {verifying ? "Verifying…" : "Verify export"}
          <input type="file" accept=".zip,application/zip" hidden disabled={verifying}
            onChange={(e) => { verifyPackage(e.target.files?.[0] ?? null); e.currentTarget.value = ""; }} />
        </label>
        {verification && (
          <span className={cx(styles.verifyStatus, verification.valid ? styles.valid : styles.tampered)}
            title={[...verification.errors, ...verification.warnings].join("\n") || "All hashes and signature verified"}>
            {verification.status}
          </span>
        )}
        <div className={styles.scenePick}>
          <span className={styles.dot} />
          <select value={scene} onChange={(e) => setScene(e.target.value)}>
            <option value="SUR01">SUR01 · Surat</option>
            <option value="surat-live">surat-live · Surat (live)</option>
            <option value="S01">S01 · CityFlow</option>
          </select>
        </div>
        <div className={styles.statusChip}>
          <b className="mono">{total.toLocaleString()}</b> entities · <b className="mono">{cameras.length}</b> cameras
        </div>
      </div>

      {/* ===== search row ===== */}
      <form
        className={styles.searchrow}
        onSubmit={(e) => { e.preventDefault(); if (mode === "photo") { if (imgFile) runImageSearch(imgFile); } else runSearch(); }}
      >
        <div className={styles.searchline}>
          <div className={styles.segmented}>
            <button type="button" className={cx(mode === "search" && styles.on)} onClick={() => setMode("search")}>Search</button>
            <button type="button" className={cx(mode === "photo" && styles.on)} onClick={() => setMode("photo")}>Photo</button>
          </div>

          {mode === "photo" ? (
            <label className={styles.photofield}>
              {imgPreview && <img src={imgPreview} alt="query" />}
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 16l4.6-4.6a2 2 0 0 1 2.8 0L16 16m-2-2 1.6-1.6a2 2 0 0 1 2.8 0L20 14M14 8h.01M6 20h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2Z" /></svg>
              <span>{imgFile ? imgFile.name : "Upload a cropped photo of the vehicle or person…"}</span>
              <input type="file" accept="image/*" hidden onChange={(e) => onPickImage(e.target.files?.[0] ?? null)} />
            </label>
          ) : (
            <div className={styles.searchfield}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
              <input
                value={q}
                onChange={(e) => { setQ(e.target.value); setForceMode(null); }}
                placeholder="Describe a person or vehicle — or type a number plate"
                spellCheck={false}
              />
            </div>
          )}

          <LocationCombo cameras={cameras} total={total} value={cameraId} onChange={setCameraId} />

          <TimeSelect
            cameras={cameras} from={tFrom} to={tTo} set={timeSet}
            onApply={(f, t) => { setTFrom(f); setTTo(t); setTimeSet(true); }}
            onClear={() => setTimeSet(false)}
          />

          <button className={styles.go} type="submit">Search</button>
        </div>

        {mode === "search" && (
          <div className={styles.assist}>
            {effMode === "plate" ? (
              <>
                <span className={styles.plChip}><span className={styles.plTag}>PLATE</span>Searching by number plate</span>
                <button type="button" className={styles.override} onClick={() => setForceMode("describe")}>search as description instead</button>
              </>
            ) : auto === "plate" && forceMode === "describe" ? (
              <>
                <span className={cx(styles.plChip, styles.alt)}><span className={styles.plTag}>TEXT</span>Searching by description</span>
                <button type="button" className={styles.override} onClick={() => setForceMode("plate")}>use number-plate search instead</button>
              </>
            ) : (
              <span className={styles.xlate}>
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M4 5h7M9 3v2c0 4-2.5 7-5 8M5 9c0 2.5 2.2 4.4 5 5M12 20l4-9 4 9M13.5 17h5" /></svg>
                <span className="mono">नारंगी शर्ट में आदमी</span>&nbsp;→ translated &amp; searched
              </span>
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
          <div className="count" style={{ fontSize: 15, fontWeight: 600 }}>
            {loading ? "Searching…" : <><b style={{ color: "var(--accent-2)" }} className="mono">{results.length}</b> {results.length === 1 ? "match" : "matches"}</>}
          </div>
          <div className={styles.activeFilters}>
            {activeChips.map((c) => (
              <span key={c.label} className={styles.afilter}>{c.label} <button className="x" onClick={c.clear} aria-label={`remove ${c.label}`}>✕</button></span>
            ))}
          </div>
        </div>

        {fellBack && <div className={styles.notice}>No matches for those filters — showing the closest results instead.</div>}
        {context && !loading && !error && <div className={styles.notice}>Showing {context}.</div>}
        {error && !loading && <div className={styles.notice}>{error}</div>}
        {!loading && searched && results.length === 0 && !error && (
          <div className={styles.emptyState}>
            <div>No results with the selected filters.</div>
            {suggestions.length > 0 && (
              <div className={styles.suggestions}>
                {suggestions.map((s) => (
                  <button
                    type="button"
                    key={s.action}
                    onClick={() => {
                      if (s.action === "expand_time") setTimeSet(false);
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        <div className={styles.grid}>
          {results.map((r) => (
            <button key={r.tracklet_id} className={styles.card} onClick={() => setActive(r)}>
              <div className={styles.thumbWrap}>
                {r.crop_url
                  ? <img className={styles.thumb} src={`${API}${r.crop_url}`} alt={r.subtype} loading="lazy" />
                  : <div className={styles.thumb} />}
                <div className={styles.scan} />
                <div className={styles.osd}>{(r.camera_label ?? r.camera_id).toUpperCase().slice(0, 12)} · {fmt(r.ts_start_s)}</div>
                <div className={styles.score}>{(r.score * 100).toFixed(0)}</div>
              </div>
              <div className={styles.cardBody}>
                <div className={styles.cardTitle}>
                  <span className="st" style={{ fontSize: 13, fontWeight: 600, textTransform: "capitalize" }}>{r.subtype}</span>
                  {r.color && <span className="cl" style={{ fontSize: 11, color: "var(--txt-dim)", textTransform: "capitalize" }}>{r.color}</span>}
                </div>
                <div className={styles.cardSub}>{r.camera_label ?? r.camera_id} · <span className="mono">{fmt(r.ts_start_s)}</span></div>
                {(r.plate || (r.attrs && r.attrs.length > 0)) && (
                  <div className={styles.tagrow}>
                    {r.plate && <span className={cx(styles.tag, styles.plate)}>{r.plate}</span>}
                    {r.matched_on && r.matched_on !== "exact" && <span className={cx(styles.tag, styles.warn)}>{r.matched_on}</span>}
                    {(r.sightings ?? 0) > 1 && <span className={styles.tag}>{r.sightings} sightings</span>}
                    {r.attrs?.slice(0, 4).map((a) => <span key={a} className={styles.tag}>{a}</span>)}
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>

      {active && <Player result={active} search={searchEvidence} onClose={() => setActive(null)} onSimilar={runSimilar} />}
    </main>
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
      { camera_id: "", camera_label: "All locations", count: total, cov, whole: true },
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
    <div className={cx(styles.combo, open && styles.open)} ref={ref}>
      <span className={styles.fieldLbl}>Location</span>
      <div className={styles.comboField} onClick={() => { if (!open) { setOpen(true); setHi(-1); } inputRef.current?.focus(); }}>
        <svg className={styles.pin} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z" /><circle cx="12" cy="10" r="3" /></svg>
        <input
          ref={inputRef} value={text} placeholder="All locations" autoComplete="off" spellCheck={false}
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
            <div className={styles.comboEmpty}>No camera matches “{text.trim()}”</div>
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
    <div className={cx(styles.timeselect, open && styles.open)} ref={ref}>
      <span className={styles.fieldLbl}>Time</span>
      <div className={cx(styles.timeTrigger, set && styles.set)} onClick={() => setOpen((o) => !o)}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>
        <span className={styles.rng}>{set ? `${from} – ${to}` : "Any time"}</span>
        <svg className={styles.chev} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
      </div>
      {open && (
        <div className={styles.timePop}>
          <h4>Time of day</h4>
          <div className={styles.fromto}>
            <input value={f} spellCheck={false} onChange={(e) => setF(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); onApply(f, t); setOpen(false); } }} />
            <span>to</span>
            <input value={t} spellCheck={false} onChange={(e) => setT(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); onApply(f, t); setOpen(false); } }} />
          </div>
          <div className={styles.cov2}>
            {cameras.map((c) => (
              <div key={c.camera_id} className={styles.cluster} style={{ left: `${pos(c.ts_start_s) * 100}%`, width: `${Math.max(1, (pos(c.ts_end_s) - pos(c.ts_start_s)) * 100)}%` }} />
            ))}
            <div className={styles.selband} style={{ left: `${bandL * 100}%`, width: `${Math.max(1.5, (bandR - bandL) * 100)}%` }} />
            <div className={styles.covTick} style={{ left: "2%" }}>{secToHHMM(span0)}</div>
            <div className={styles.covTick} style={{ right: "2%" }}>{secToHHMM(span1)}</div>
          </div>
          <div className={styles.covHint}>Footage exists only in the lit bands — pick within one.</div>
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
            <button type="button" className={styles.apply} onClick={() => { onApply(f, t); setOpen(false); }}>Apply</button>
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

function Player({ result, search, onClose, onSimilar }: {
  result: Result; search: SearchEvidenceContext | null; onClose: () => void;
  onSimilar: (r: Result, vec: "semantic" | "reid") => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const boxRef = useRef<SVGRectElement>(null);
  const [boxes, setBoxes] = useState<BoxRow[] | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
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

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.player} onClick={(e) => e.stopPropagation()}>
        <div className={styles.playerHead}>
          <strong>{result.tracklet_id}</strong>
          <span>
            {result.subtype}{result.color ? ` · ${result.color}` : ""}{result.plate ? ` · ${result.plate}` : ""} · {result.camera_label ?? result.camera_id} · {fmt(result.ts_start_s)}–{fmt(result.ts_end_s)}
            {result.video_start_s != null && ` (${fmtClip(result.video_start_s)} in clip)`}
          </span>
          <button className={styles.simBtn} onClick={() => onSimilar(result, "semantic")} title="objects that look like this one">similar look</button>
          <button className={styles.simBtn} onClick={() => onSimilar(result, "reid")} title="other sightings of this exact object">same object</button>
          <button className={styles.exportBtn} onClick={() => setExportOpen((v) => !v)}>Export evidence</button>
          <button className={styles.close} onClick={onClose}>✕</button>
        </div>
        {exportOpen && <ExportPanel result={result} search={search} />}
        {src ? (
          <div className={styles.videoWrap}>
            <video ref={ref} className={styles.video} src={src} controls autoPlay
              onLoadedMetadata={() => { if (ref.current) ref.current.currentTime = seek0; }}
              onTimeUpdate={() => { const v = ref.current; if (v && v.currentTime >= seek1) v.currentTime = seek0; }} />
            {boxes && boxes.length > 0 && <svg className={styles.boxSvg}><rect ref={boxRef} className={styles.boxRect} visibility="hidden" rx={3} /></svg>}
          </div>
        ) : <div className={styles.status}>No video for this tracklet.</div>}
        {result.global_id != null && <TraceView scene={result.scene} globalId={result.global_id} />}
      </div>
    </div>
  );
}

function ExportPanel({ result, search }: { result: Result; search: SearchEvidenceContext | null }) {
  const [caseId, setCaseId] = useState(`CASE-${new Date().toISOString().slice(0, 10).replaceAll("-", "")}`);
  const [officer, setOfficer] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const create = async () => {
    if (!caseId.trim() || !officer.trim()) {
      setMessage("Case ID and officer / user are required for chain-of-custody metadata.");
      return;
    }
    setBusy(true); setMessage("Copying source evidence, creating derived files, and signing package…");
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
      setMessage(`Export signed and downloaded · ${response.headers.get("X-Forensic-Export-ID") ?? filename}`);
    } catch (e) {
      setMessage(`Export failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally { setBusy(false); }
  };

  return (
    <div className={styles.exportPanel}>
      <div><strong>Forensic export</strong><span>Source recording remains unannotated; derived files are labelled and signed.</span></div>
      <label>Case ID<input value={caseId} maxLength={120} onChange={(e) => setCaseId(e.target.value)} /></label>
      <label>Officer / user<input value={officer} maxLength={120} placeholder="Name or badge ID" onChange={(e) => setOfficer(e.target.value)} /></label>
      <button type="button" disabled={busy} onClick={create}>{busy ? "Building package…" : "Create signed ZIP"}</button>
      {message && <p>{message}</p>}
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
        {single ? "Seen in a single camera (no cross-camera path)."
          : camsInPath.size === 1 ? `${hops.length} sightings in this camera (fragments stitched by plate)`
            : `Cross-camera path · ${hops.length} sightings across ${camsInPath.size} cameras`}
      </div>
      <div className={styles.traceBody} style={Object.keys(positions).length === 0 ? { gridTemplateColumns: "1fr" } : undefined}>
        {Object.keys(positions).length > 0 && (
          <div className={styles.mapWrap}>
            <img className={styles.mapImg} src={`${API}/scene-map/${scene}`} alt="scene map" />
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
