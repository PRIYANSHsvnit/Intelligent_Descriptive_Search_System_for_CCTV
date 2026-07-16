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
  score: number;
  crop_url: string | null;
  video_url: string | null;
  global_id: number | null;
  attrs?: string[];
  plate?: string | null;
  plate_conf?: number | null;
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
  crop_url: string | null;
  video_url: string | null;
};

export default function Home() {
  const [q, setQ] = useState("man wearing a cap");
  const [mode, setMode] = useState<"desc" | "plate">("desc");
  const [type, setType] = useState("");
  const [scene, setScene] = useState("SUR01");
  const [results, setResults] = useState<Result[]>([]);
  const [fellBack, setFellBack] = useState(false);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [active, setActive] = useState<Result | null>(null);
  const [traceGid, setTraceGid] = useState<number | null>(null);

  const runSearch = useCallback(
    async (query: string) => {
      if (!query.trim()) return;
      setLoading(true);
      setSearched(true);
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
        const data = await r.json();
        setResults(data.results ?? []);
        setFellBack(Boolean(data.fell_back));
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    },
    [mode, type, scene],
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
          Describe a vehicle in plain words — get every matching clip across cameras.
        </p>
      </header>

      <form
        className={styles.searchBar}
        onSubmit={(e) => {
          e.preventDefault();
          runSearch(q);
        }}
      >
        <select
          className={styles.select}
          value={mode}
          onChange={(e) => setMode(e.target.value as "desc" | "plate")}
        >
          <option value="desc">describe</option>
          <option value="plate">plate no.</option>
        </select>
        <input
          className={styles.input}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={mode === "plate" ? "e.g. GJ05JP4237 or just 4237" : "e.g. white pickup truck"}
        />
        {mode === "desc" && (
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
        {(mode === "plate" ? PLATE_EXAMPLES : EXAMPLES).map((ex) => (
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
              {r.camera_label ?? r.camera_id} · {r.ts_start_s.toFixed(1)}–{r.ts_end_s.toFixed(1)}s
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

      {active && <Player result={active} onClose={() => setActive(null)} />}
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

function Player({ result, onClose }: { result: Result; onClose: () => void }) {
  const ref = useRef<HTMLVideoElement>(null);
  const src = result.video_url ? `${API}${result.video_url}` : null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.player} onClick={(e) => e.stopPropagation()}>
        <div className={styles.playerHead}>
          <strong>{result.tracklet_id}</strong>
          <span>
            {result.subtype}
            {result.color ? ` · ${result.color}` : ""}
            {result.plate ? ` · ${result.plate}` : ""} · {result.camera_label ?? result.camera_id} ·{" "}
            {result.ts_start_s.toFixed(1)}–{result.ts_end_s.toFixed(1)}s
          </span>
          <button className={styles.close} onClick={onClose}>
            ✕
          </button>
        </div>
        {src ? (
          <video
            ref={ref}
            className={styles.video}
            src={src}
            controls
            autoPlay
            onLoadedMetadata={() => {
              if (ref.current) ref.current.currentTime = result.ts_start_s;
            }}
            onTimeUpdate={() => {
              const v = ref.current;
              if (v && v.currentTime >= result.ts_end_s) {
                v.currentTime = result.ts_start_s; // loop the object's window
              }
            }}
          />
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
          : `Cross-camera path · ${hops.length} sightings across ${camsInPath.size} cameras`}
      </div>
      <div className={styles.traceBody}>
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
        <div className={styles.timeline}>
          {hops.map((h, i) => (
            <div key={`${h.tracklet_id}-${i}`} className={styles.hop}>
              {h.crop_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img className={styles.hopCrop} src={`${API}${h.crop_url}`} alt={h.camera_id} />
              )}
              <div className={styles.hopMeta}>
                <strong>{h.camera_label ?? h.camera_id}</strong>
                <span>{h.ts_start_s.toFixed(1)}s</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
