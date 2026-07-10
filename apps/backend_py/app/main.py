"""CCTV descriptive-search API (Phase 2).

Endpoints (see plan.md API contract):
  GET /search?q=&type=&scene=&t0=&t1=&limit=  -> ranked, de-duplicated tracklets
  GET /media/{scene}/{camera}                 -> per-camera H.264/MP4 (HTTP range)
  GET /trace/{scene}/{global_id}              -> cross-camera hops (Phase 3 data)
  /files/<crop_ref>                           -> crop thumbnails (static)

Auth (hackathon posture): none — one shared local DB. Don't build accounts.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()  # load SigLIP once at startup
    yield


app = FastAPI(title="CCTV Descriptive Search", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon: frontend on :3000 → backend on :8000
    allow_methods=["*"],
    allow_headers=["*"],
)

# crop thumbnails (StaticFiles supports range too); created lazily by ingest
config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(config.OUTPUT_ROOT)), name="files")


@app.get("/health")
def health():
    return {"ok": True, "model": config.SIGLIP_MODEL, "output_root": str(config.OUTPUT_ROOT)}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    type: str | None = Query(None, pattern="^(vehicle|person)$"),
    scene: str | None = None,
    t0: float | None = None,
    t1: float | None = None,
    limit: int = Query(20, ge=1, le=100),
):
    return engine.search(q, type, scene, t0, t1, limit)


@app.get("/media/{scene}/{camera}")
def media(scene: str, camera: str):
    path = config.OUTPUT_ROOT / scene / camera / "media" / f"{camera}.mp4"
    if not path.exists():
        raise HTTPException(404, f"no media for {scene}/{camera}")
    return FileResponse(str(path), media_type="video/mp4")  # starlette handles Range


@app.get("/trace/{scene}/{global_id}")
def trace(scene: str, global_id: int):
    return engine.trace(scene, global_id)
