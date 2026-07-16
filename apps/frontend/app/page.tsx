"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

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
};

const EXAMPLES = [
  "man wearing a cap",
  "person in a blue shirt",
  "woman in a red saree",
  "man with a backpack",
  "motorcycle",
  "auto rickshaw",
];

// registration queries: full plate or any fragment (last-4 digits is the realistic case)
const PLATE_EXAMPLES = ["GJ05JP4237", "GJ05CU6121", "GJ18Z8995", "4237", "GJ08"];

// scene-clock seconds -> time of day h:mm:ss (offsets are seconds since midnight)
function fmt(t: number): string {
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = Math.floor(t % 60);
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

// video-local seconds -> m:ss position inside the clip
function fmtClip(t: number): string {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}

// Hand-annotated camera pixel positions (normalized 0..1) on cam_loc/<scene>.png.
const CAM_POS: Record<string, Record<string, [number, number]>> = {
  S01: {
    c001: [0.61, 0.67],
    c002: [0.33, 0.38],
    c003: [0.68, 0.4],
    c004: [0.33, 0.62],
    c005: [0.53, 0.32],
  },
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

export default function Home() {
  const [q, setQ] = useState("man wearing a cap");
  const [mode, setMode] = useState<"desc" | "plate" | "photo">("desc");
  const [type, setType] = useState("");
  const [scene, setScene] = useState("SUR01");
  const [results, setResults] = useState<Result[]>([]);
  const [fellBack, setFellBack] = useState(false);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [active, setActive] = useState<Result | null>(null);
  const [traceGid, setTraceGid] = useState<number | null>(null);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  // what a photo / find-similar result set is relative to (shown as a notice)
  const [context, setContext] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSearch = useCallback(
    async (query: string) => {
      if (!query.trim()) return;
      setLoading(true);
      setSearched(true);
      setContext(null);
      setError(null);
      try {
        const p = new URLSearchParams({ limit: "24" });
        if (mode === "plate") {
          p.set("plate", query);
        } else {
          p.set("q", query);
          if (type) p.set("type", type);
        }
        if (scene) p.set("scene", scene);
        const r = await fetch(`${API}/search?${p}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        setResults(data.results ?? []);
        setFellBack(Boolean(data.fell_back));
      } catch (e) {
        setResults([]);
        setError(`Search failed (${e instanceof Error ? e.message : "network"}) — is the backend up to date?`);
      } finally {
        setLoading(false);
      }
    },
    [mode, type, scene],
  );

  // reference-image search: multipart POST, same response shape as text search
  const runImageSearch = useCallback(
    async (file: File) => {
      setLoading(true);
      setSearched(true);
      setContext("matches for the uploaded photo");
      setError(null);
      try {
        const fd = new FormData();
        fd.append("image", file);
        if (type) fd.append("type", type);
        if (scene) fd.append("scene", scene);
        fd.append("limit", "24");
        const r = await fetch(`${API}/search/image`, { method: "POST", body: fd });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        setResults(data.results ?? []);
        setFellBack(Boolean(data.fell_back));
      } catch (e) {
        setResults([]);
        setError(`Image search failed (${e instanceof Error ? e.message : "network"}) — is the backend up to date?`);
      } finally {
        setLoading(false);
      }
    },
    [type, scene],
  );

  // more-like-this from a stored tracklet (no upload; server reuses stored vectors)
  const runSimilar = useCallback(async (r: Result, vec: "semantic" | "reid") => {
    setActive(null);
    setLoading(true);
    setSearched(true);
    setContext(
      vec === "reid"
        ? `re-ID matches for ${r.subtype} ${r.tracklet_id} (same object, other sightings)`
        : `tracklets that look similar to ${r.subtype} ${r.tracklet_id}`,
    );
    setError(null);
    try {
      const p = new URLSearchParams({ tracklet_id: r.tracklet_id, vec, limit: "24" });
      const res = await fetch(`${API}/search/similar?${p}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResults(data.results ?? []);
      setFellBack(false);
      if (data.fell_back && vec === "reid") {
        setContext(
          `look-alikes of ${r.subtype} ${r.tracklet_id} (no re-ID vector stored — visual similarity only)`,
        );
      }
    } catch (e) {
      setResults([]);
      setError(`Similar search failed (${e instanceof Error ? e.message : "network"}) — is the backend up to date?`);
    } finally {
      setLoading(false);
    }
  }, []);

  const onPickImage = useCallback(
    (f: File | null) => {
      if (!f) return;
      setImgFile(f);
      setImgPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(f);
      });
      runImageSearch(f); // search immediately on pick
    },
    [runImageSearch],
  );

  // run the default query once on mount; support ?trace=<gid> deep-link to a path
  useEffect(() => {
    runSearch("man wearing a cap");
    const g = new URLSearchParams(window.location.search).get("trace");
    if (g) setTraceGid(Number(g));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1 className={styles.title}>CCTV Descriptive Search</h1>
        <p className={styles.subtitle}>
          Describe a vehicle in plain words — or upload a photo — get every matching clip
          across cameras.
        </p>
      </header>

      <form
        className={styles.searchBar}
        onSubmit={(e) => {
          e.preventDefault();
          if (mode === "photo") {
            if (imgFile) runImageSearch(imgFile);
          } else {
            runSearch(q);
          }
        }}
      >
        <select
          className={styles.select}
          value={mode}
          onChange={(e) => setMode(e.target.value as "desc" | "plate" | "photo")}
        >
          <option value="desc">describe</option>
          <option value="plate">plate no.</option>
          <option value="photo">photo</option>
        </select>
        {mode === "photo" ? (
          <label className={styles.fileLabel}>
            {imgPreview && (
              // eslint-disable-next-line @next/next/no-img-element
              <img className={styles.filePreview} src={imgPreview} alt="query" />
            )}
            <span>
              {imgFile ? imgFile.name : "upload a cropped photo of the vehicle/person…"}
            </span>
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => onPickImage(e.target.files?.[0] ?? null)}
            />
          </label>
        ) : (
          <input
            className={styles.input}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={mode === "plate" ? "e.g. GJ05JP4237 or just 4237" : "e.g. white pickup truck"}
          />
        )}
        {mode !== "plate" && (
          <select className={styles.select} value={type} onChange={(e) => setType(e.target.value)}>
            <option value="">any type</option>
            <option value="vehicle">vehicle</option>
            <option value="person">person</option>
          </select>
        )}
        <select className={styles.select} value={scene} onChange={(e) => setScene(e.target.value)}>
          <option value="">all scenes</option>
          <option value="SUR01">Surat (SUR01)</option>
          <option value="S01">S01</option>
        </select>
        <button className={styles.button} type="submit">
          Search
        </button>
      </form>

      <div className={styles.examples}>
        {(mode === "photo" ? [] : mode === "plate" ? PLATE_EXAMPLES : EXAMPLES).map((ex) => (
          <button
            key={ex}
            className={styles.chip}
            onClick={() => {
              setQ(ex);
              runSearch(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {fellBack && (
        <div className={styles.notice}>
          No matches for those filters — showing the closest results instead.
        </div>
      )}
      {context && !loading && !error && (
        <div className={styles.notice}>Showing {context}.</div>
      )}
      {error && !loading && <div className={styles.notice}>{error}</div>}

      {loading && <div className={styles.status}>Searching…</div>}
      {!loading && searched && results.length === 0 && (
        <div className={styles.status}>No results.</div>
      )}

      <div className={styles.grid}>
        {results.map((r) => (
          <button key={r.tracklet_id} className={styles.card} onClick={() => setActive(r)}>
            {r.crop_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img className={styles.thumb} src={`${API}${r.crop_url}`} alt={r.subtype} />
            ) : (
              <div className={styles.thumb} />
            )}
            <div className={styles.meta}>
              <span className={styles.subtype}>{r.subtype}</span>
              {r.color && <span className={styles.color}>{r.color}</span>}
              <span className={styles.score}>{(r.score * 100).toFixed(0)}</span>
            </div>
            <div className={styles.metaSub}>
              {r.camera_label ?? r.camera_id} · {fmt(r.ts_start_s)}–{fmt(r.ts_end_s)}
              {r.video_start_s != null && ` · ${fmtClip(r.video_start_s)} in clip`}
            </div>
            {(r.plate || r.matched_on) && (
              <div className={styles.plateRow}>
                {r.plate && (
                  <span className={styles.plate}>
                    {r.plate}
                    <span className={styles.plateConf}>
                      {(r.plate_conf ?? 0) >= 0.8 ? "confirmed" : "probable"}
                    </span>
                  </span>
                )}
                {(r.sightings ?? 0) > 1 && (
                  <span className={styles.matchTag}>{r.sightings} sightings</span>
                )}
                {r.matched_on && r.matched_on !== "exact" && (
                  <span className={styles.matchTag}>
                    {r.matched_on === "partial" ? "partial match" : "fuzzy match — verify in clip"}
                  </span>
                )}
              </div>
            )}
            {r.attrs && r.attrs.length > 0 && (
              <div className={styles.attrs}>
                {r.attrs.map((a) => (
                  <span key={a} className={styles.attr}>
                    {a}
                  </span>
                ))}
              </div>
            )}
          </button>
        ))}
      </div>

      {active && (
        <Player result={active} onClose={() => setActive(null)} onSimilar={runSimilar} />
      )}
      {traceGid != null && (
        <div className={styles.overlay} onClick={() => setTraceGid(null)}>
          <div className={styles.player} onClick={(e) => e.stopPropagation()}>
            <div className={styles.playerHead}>
              <strong>{scene || "S01"} · global_id {traceGid}</strong>
              <button className={styles.close} onClick={() => setTraceGid(null)}>
                ✕
              </button>
            </div>
            <TraceView scene={scene || "S01"} globalId={traceGid} />
          </div>
        </div>
      )}
    </main>
  );
}

// per-frame box timeline from /tracklets/{id}/boxes: [video-local s, x1,y1,x2,y2]
// in source-video pixels (media is never rescaled, so videoWidth/Height match)
type BoxRow = [number, number, number, number, number];

// largest i with boxes[i].t <= t (binary search; -1 if t precedes the track)
function boxIndexAt(boxes: BoxRow[], t: number): number {
  let lo = 0,
    hi = boxes.length - 1,
    ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (boxes[mid]![0] <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

function Player({
  result,
  onClose,
  onSimilar,
}: {
  result: Result;
  onClose: () => void;
  onSimilar: (r: Result, vec: "semantic" | "reid") => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const boxRef = useRef<SVGRectElement>(null);
  const [boxes, setBoxes] = useState<BoxRow[] | null>(null);
  const src = result.video_url ? `${API}${result.video_url}` : null;
  // seek in video-file time (scene-clock minus camera offset); older rows without it
  // fall back to the scene timestamps (S01, where offsets are ~0)
  const seek0 = result.video_start_s ?? result.ts_start_s;
  const seek1 = result.video_end_s ?? result.ts_end_s;

  useEffect(() => {
    let ok = true;
    setBoxes(null);
    fetch(`${API}/tracklets/${result.tracklet_id}/boxes`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => ok && setBoxes(d?.boxes ?? null))
      .catch(() => ok && setBoxes(null)); // overlay is best-effort; video still plays
    return () => {
      ok = false;
    };
  }, [result.tracklet_id]);

  // follow the object: every animation frame, place the rect for currentTime
  // (timeupdate alone fires ~4 Hz and would lag the video)
  useEffect(() => {
    if (!boxes || boxes.length === 0) return;
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const v = ref.current;
      const rect = boxRef.current;
      if (!v || !rect || !v.videoWidth) return;
      const t = v.currentTime;
      const i = boxIndexAt(boxes, t);
      const cur = i >= 0 ? boxes[i] : undefined;
      // hide before the track starts, after it ends, or across detection gaps
      const GAP = 0.5;
      if (!cur || t - cur[0] > GAP) {
        rect.setAttribute("visibility", "hidden");
        return;
      }
      let [t0, x1, y1, x2, y2] = cur;
      const nxt = boxes[i + 1];
      if (nxt && nxt[0] > t0 && nxt[0] - t0 <= GAP) {
        const a = (t - t0) / (nxt[0] - t0);
        x1 += (nxt[1] - x1) * a;
        y1 += (nxt[2] - y1) * a;
        x2 += (nxt[3] - x2) * a;
        y2 += (nxt[4] - y2) * a;
      }
      // map source pixels -> element pixels through the video's contain-fit
      // (max-height can pillarbox the content inside the element box)
      const scale = Math.min(v.clientWidth / v.videoWidth, v.clientHeight / v.videoHeight);
      const ox = (v.clientWidth - v.videoWidth * scale) / 2;
      const oy = (v.clientHeight - v.videoHeight * scale) / 2;
      rect.setAttribute("x", String(ox + x1 * scale));
      rect.setAttribute("y", String(oy + y1 * scale));
      rect.setAttribute("width", String(Math.max(0, (x2 - x1) * scale)));
      rect.setAttribute("height", String(Math.max(0, (y2 - y1) * scale)));
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
            {result.subtype}
            {result.color ? ` · ${result.color}` : ""}
            {result.plate ? ` · ${result.plate}` : ""} · {result.camera_label ?? result.camera_id} ·{" "}
            {fmt(result.ts_start_s)}–{fmt(result.ts_end_s)}
            {result.video_start_s != null && ` (${fmtClip(result.video_start_s)} in clip)`}
          </span>
          <button
            className={styles.simBtn}
            title="other objects that look like this one (SigLIP)"
            onClick={() => onSimilar(result, "semantic")}
          >
            similar look
          </button>
          <button
            className={styles.simBtn}
            title="other sightings of this exact object (re-ID appearance)"
            onClick={() => onSimilar(result, "reid")}
          >
            same object
          </button>
          <button className={styles.close} onClick={onClose}>
            ✕
          </button>
        </div>
        {src ? (
          <div className={styles.videoWrap}>
            <video
              ref={ref}
              className={styles.video}
              src={src}
              controls
              autoPlay
              onLoadedMetadata={() => {
                if (ref.current) ref.current.currentTime = seek0;
              }}
              onTimeUpdate={() => {
                const v = ref.current;
                if (v && v.currentTime >= seek1) {
                  v.currentTime = seek0; // loop the object's window
                }
              }}
            />
            {boxes && boxes.length > 0 && (
              <svg className={styles.boxSvg}>
                <rect ref={boxRef} className={styles.boxRect} visibility="hidden" rx={3} />
              </svg>
            )}
          </div>
        ) : (
          <div className={styles.status}>No video for this tracklet.</div>
        )}
        {result.global_id != null && (
          <TraceView scene={result.scene} globalId={result.global_id} />
        )}
      </div>
    </div>
  );
}

function TraceView({ scene, globalId }: { scene: string; globalId: number }) {
  const [hops, setHops] = useState<Hop[] | null>(null);
  useEffect(() => {
    let ok = true;
    fetch(`${API}/trace/${scene}/${globalId}`)
      .then((r) => r.json())
      .then((d) => ok && setHops(d.hops ?? []))
      .catch(() => ok && setHops([]));
    return () => {
      ok = false;
    };
  }, [scene, globalId]);

  if (!hops) return <div className={styles.status}>Loading path…</div>;

  const positions = CAM_POS[scene] ?? {};
  // path points through hop cameras in time order (drop consecutive duplicates)
  const pts: [number, number][] = [];
  for (const h of hops) {
    const p = positions[h.camera_id];
    if (!p) continue;
    const last = pts[pts.length - 1];
    if (!last || last[0] !== p[0] || last[1] !== p[1]) pts.push(p);
  }
  const camsInPath = new Set(hops.map((h) => h.camera_id));
  const single = hops.length <= 1;

  return (
    <div className={styles.trace}>
      <div className={styles.traceHead}>
        {single
          ? "Seen in a single camera (no cross-camera path)."
          : camsInPath.size === 1
            ? `${hops.length} sightings in this camera (fragments stitched by plate)`
            : `Cross-camera path · ${hops.length} sightings across ${camsInPath.size} cameras`}
      </div>
      <div className={styles.traceBody} style={Object.keys(positions).length === 0 ? { gridTemplateColumns: "1fr" } : undefined}>
        {Object.keys(positions).length > 0 && (
        <div className={styles.mapWrap}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className={styles.mapImg} src={`${API}/scene-map/${scene}`} alt="scene map" />
          <svg className={styles.mapSvg} viewBox="0 0 100 100" preserveAspectRatio="none">
            {pts.length > 1 && (
              <polyline
                points={pts.map(([x, y]) => `${x * 100},${y * 100}`).join(" ")}
                fill="none"
                stroke="#4c8bf5"
                strokeWidth={1.1}
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {Object.entries(positions).map(([cam, [x, y]]) => (
              <circle
                key={cam}
                cx={x * 100}
                cy={y * 100}
                r={1.8}
                fill={camsInPath.has(cam) ? "#4c8bf5" : "#555"}
                stroke="#0b0c0f"
                strokeWidth={0.5}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>
        </div>
        )}
        <div className={styles.timeline}>
          {hops.map((h, i) => (
            <div key={`${h.tracklet_id}-${i}`} className={styles.hop}>
              {h.crop_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img className={styles.hopCrop} src={`${API}${h.crop_url}`} alt={h.camera_id} />
              )}
              <div className={styles.hopMeta}>
                <strong>{h.camera_label ?? h.camera_id}</strong>
                <span title={h.video_start_s != null ? `${fmtClip(h.video_start_s)} into the clip` : undefined}>
                  {fmt(h.ts_start_s)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
