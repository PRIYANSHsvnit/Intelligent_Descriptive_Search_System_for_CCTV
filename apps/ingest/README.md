# apps/ingest — offline CV pipeline

Detect → track → crop → attributes → embed → store, per camera. GPU-first
(YOLO + SigLIP on the RTX 4050); the re-ID pass runs on CPU by design.

## Setup

```bash
cd apps/ingest
uv sync
uv run python verify_gpu.py     # must print "OK: GPU is live." (cuda:0)
```

On this NixOS box `gpu_setup.ensure_gpu_libs()` must run **before** torch is
imported (it prepends the Nix driver dir to `LD_LIBRARY_PATH` and re-execs once).
Every ML entry point imports and calls it first. See `gpu_setup.py`.

## Locked decisions (Phase 0 — do not change)

Frozen in `constants.py`; the Postgres schema hardcodes the dims. Changing any of
these forces re-embedding every vector.

| Vector | Model | Dim |
|---|---|---|
| `semantic_vector` (search, GPU) | `google/siglip2-so400m-patch14-224` | **1152** |
| `reid_appearance` (tracing, CPU) | VeRi FastReID ResNet50-IBN (ONNX) | 2048 |
| `reid_color` (tracing) | HSV signature | 56 |

Detector: `yolo11m.pt`, `imgsz=1280`, `conf=0.3`, COCO classes {0,1,2,3,5,7}.

## Database

Postgres 16 + pgvector runs via Docker Compose in `../../infra/`:

```bash
cd ../../infra && docker compose up -d
# DATABASE_URL=postgresql://cctv:cctv@localhost:5432/cctv
```

The schema (`infra/schema.sql`) applies automatically on first init. The
`tracklets` table is one row per tracklet; every not-yet-computed field is
nullable so people/plates/global_id are additive later.
