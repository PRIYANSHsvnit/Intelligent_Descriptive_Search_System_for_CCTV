"""CCTV descriptive-search API (Phase 2).

Endpoints (see plan.md API contract):
  GET /search?q=&type=&scene=&t0=&t1=&limit=  -> ranked, de-duplicated tracklets
  GET /search?plate=&scene=&limit=            -> layered plate lookup (exact/partial/fuzzy)
  POST /search/image (multipart)              -> reference-image search (SigLIP image tower)
  GET /search/similar?tracklet_id=&vec=       -> more-like-this from a stored tracklet
  GET /media/{scene}/{camera}                 -> per-camera H.264/MP4 (HTTP range)
  GET /tracklets/{tracklet_id}/boxes          -> per-frame boxes for the player overlay
  GET /trace/{scene}/{global_id}              -> cross-camera hops (Phase 3 data)
  /files/<crop_ref>                           -> crop thumbnails (static)

Auth (hackathon posture): none — one shared local DB. Don't build accounts.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import boxes, config, engine


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
    q: str | None = Query(None, min_length=1),
    plate: str | None = Query(None, min_length=2, max_length=16,
                              description="registration query, full or partial (e.g. 'GJ05AY1139' or '1139')"),
    type: str | None = Query(None, pattern="^(vehicle|person)$"),
    scene: str | None = None,
    camera_id: str | None = Query(None, description="restrict to one camera/location"),
    color: str | None = Query(None, description="filter by stored colour name (vehicle-level)"),
    t0: float | None = None,
    t1: float | None = None,
    limit: int = Query(20, ge=1, le=100),
):
    if plate:
        return engine.search_plate(plate, scene, limit, camera_id=camera_id)
    if not q:
        raise HTTPException(422, "provide q (description) or plate")
    return engine.search(q, type, scene, t0, t1, limit, camera_id=camera_id, color=color)


@app.get("/scene-cameras/{scene}")
def scene_cameras(scene: str):
    """Cameras + labels + counts + coverage for the location picker."""
    return engine.scene_cameras(scene)


@app.post("/search/image")
async def search_image(
    image: UploadFile = File(..., description="tight crop of the vehicle/person"),
    type: str | None = Form(None),
    scene: str | None = Form(None),
    camera_id: str | None = Form(None),
    color: str | None = Form(None),
    t0: float | None = Form(None),
    t1: float | None = Form(None),
    limit: int = Form(20),
):
    if type and type not in ("vehicle", "person"):
        raise HTTPException(422, "type must be 'vehicle' or 'person'")
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(422, "could not decode image")
    return engine.search_image(img, type or None, scene or None, t0, t1,
                               min(max(limit, 1), 100),
                               camera_id=camera_id or None, color=color or None)


@app.get("/search/similar")
def search_similar(
    tracklet_id: str,
    vec: str = Query("semantic", pattern="^(semantic|reid)$",
                     description="semantic = looks similar, reid = same instance"),
    limit: int = Query(20, ge=1, le=100),
):
    out = engine.search_similar(tracklet_id, vec, limit)
    if out is None:
        raise HTTPException(404, f"no tracklet {tracklet_id}")
    return out


@app.get("/media/{scene}/{camera}")
def media(scene: str, camera: str):
    path = config.OUTPUT_ROOT / scene / camera / "media" / f"{camera}.mp4"
    if not path.exists():
        raise HTTPException(404, f"no media for {scene}/{camera}")
    return FileResponse(str(path), media_type="video/mp4")  # starlette handles Range


@app.get("/tracklets/{tracklet_id}/boxes")
def tracklet_boxes(tracklet_id: str):
    out = boxes.tracklet_boxes(tracklet_id)
    if out is None:
        raise HTTPException(404, f"no boxes for {tracklet_id}")
    # detections.npy is immutable per ingest run — let the browser keep it 6h
    return JSONResponse(out, headers={"Cache-Control": "public, max-age=21600"})


@app.get("/trace/{scene}/{global_id}")
def trace(scene: str, global_id: int):
    return engine.trace(scene, global_id)


@app.get("/scene-map/{scene}")
def scene_map(scene: str):
    path = config.scene_map_file(scene)
    if not path.exists():
        raise HTTPException(404, f"no map for {scene}")
    return FileResponse(str(path), media_type="image/png")
