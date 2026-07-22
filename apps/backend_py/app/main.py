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
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import boxes, config, engine, forensics, investigations


class ForensicExportRequest(BaseModel):
    tracklet_id: str = Field(min_length=1, max_length=200)
    case_id: str = Field(min_length=1, max_length=120)
    officer: str = Field(min_length=1, max_length=120)
    selected_crop_ref: str | None = Field(default=None, max_length=500)
    search: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)


class CaseRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=120)
    officer: str = Field(min_length=1, max_length=120)
    title: str | None = Field(default=None, max_length=240)


class CaseItemRequest(BaseModel):
    officer: str = Field(min_length=1, max_length=120)
    status: str = Field(pattern="^(pinned|excluded)$")
    note: str | None = Field(default=None, max_length=2000)
    result_snapshot: dict[str, Any] = Field(default_factory=dict)


class CaseExportRequest(BaseModel):
    officer: str = Field(min_length=1, max_length=120)


def _audit_context(case_id: str | None, officer: str | None) -> tuple[str, str] | None:
    if bool(case_id) != bool(officer):
        raise HTTPException(422, "case_id and officer must be supplied together for audit logging")
    return (case_id, officer) if case_id and officer else None


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
    expose_headers=["Content-Disposition", "X-Forensic-Export-ID", "X-Forensic-Key-ID"],
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
    aggregation: str | None = Query(
        None, pattern="^(legacy_mean|mean|best_single|max|top2_mean)$"),
    compositional: bool = True,
    composition: str | None = Query(None, pattern="^(weighted|soft_and)$"),
    limit: int = Query(20, ge=1, le=100),
    case_id: str | None = Query(None, max_length=120),
    officer: str | None = Query(None, max_length=120),
):
    audit = _audit_context(case_id, officer)
    started = time.perf_counter()
    if plate:
        out = engine.search_plate(plate, scene, limit, camera_id=camera_id)
        query, mode = plate, "layered_plate_lookup"
    else:
        if not q:
            raise HTTPException(422, "provide q (description) or plate")
        if type == "person" and color:
            raise HTTPException(422, "color is vehicle-level; describe person clothing in q")
        out = engine.search(q, type, scene, t0, t1, limit, camera_id=camera_id, color=color,
                            aggregation=aggregation, compositional=compositional,
                            composition=composition)
        query, mode = q, out.get("search_mode", "semantic_search")
    if audit:
        filters = {"scene": scene, "camera_id": camera_id, "type": type, "color": color,
                   "t0": t0, "t1": t1, "limit": limit}
        out["audit_event_id"] = investigations.record_search(
            out, case_id=audit[0], officer=audit[1], query=query, filters=filters,
            search_mode=mode, latency_ms=(time.perf_counter() - started) * 1000,
        )
    return out


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
    aggregation: str | None = Form(None),
    limit: int = Form(20),
    case_id: str | None = Form(None),
    officer: str | None = Form(None),
):
    audit = _audit_context(case_id, officer)
    started = time.perf_counter()
    if type and type not in ("vehicle", "person"):
        raise HTTPException(422, "type must be 'vehicle' or 'person'")
    if type == "person" and color:
        raise HTTPException(422, "color is vehicle-level; person clothing is semantic")
    if aggregation and aggregation not in (
            "legacy_mean", "mean", "best_single", "max", "top2_mean"):
        raise HTTPException(422, "invalid aggregation")
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(422, "could not decode image")
    out = engine.search_image(img, type or None, scene or None, t0, t1,
                              min(max(limit, 1), 100),
                              camera_id=camera_id or None, color=color or None,
                              aggregation=aggregation or None)
    if audit:
        filters = {"scene": scene, "camera_id": camera_id, "type": type, "color": color,
                   "t0": t0, "t1": t1, "limit": limit}
        out["audit_event_id"] = investigations.record_search(
            out, case_id=audit[0], officer=audit[1], query="[reference image search]",
            filters=filters, search_mode=out.get("search_mode", "siglip_image_search"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    return out


@app.get("/search/similar")
def search_similar(
    tracklet_id: str,
    vec: str = Query("semantic", pattern="^(semantic|reid)$",
                     description="semantic = looks similar, reid = same instance"),
    limit: int = Query(20, ge=1, le=100),
    case_id: str | None = Query(None, max_length=120),
    officer: str | None = Query(None, max_length=120),
):
    audit = _audit_context(case_id, officer)
    started = time.perf_counter()
    out = engine.search_similar(tracklet_id, vec, limit)
    if out is None:
        raise HTTPException(404, f"no tracklet {tracklet_id}")
    if audit:
        out["audit_event_id"] = investigations.record_search(
            out, case_id=audit[0], officer=audit[1],
            query=f"{vec} similarity from {tracklet_id}",
            filters={"source_tracklet_id": tracklet_id, "vec": vec, "limit": limit},
            search_mode=f"{vec}_similarity",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    return out


@app.post("/cases")
def create_case(request: CaseRequest):
    try:
        return investigations.ensure_case(request.case_id, request.officer, request.title)
    except investigations.InvestigationError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/cases/{case_id}")
def case_board(case_id: str):
    out = investigations.get_case(case_id)
    if out is None:
        raise HTTPException(404, f"case {case_id} does not exist")
    return out


@app.get("/cases/{case_id}/timeline")
def case_timeline(case_id: str):
    if investigations.get_case(case_id) is None:
        raise HTTPException(404, f"case {case_id} does not exist")
    return {"case_id": case_id, "events": investigations.get_timeline(case_id)}


@app.put("/cases/{case_id}/items/{tracklet_id}")
def update_case_item(case_id: str, tracklet_id: str, request: CaseItemRequest):
    try:
        return investigations.set_case_item(
            case_id, tracklet_id, request.officer, request.status,
            request.note, request.result_snapshot,
        )
    except investigations.InvestigationError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/cases/{case_id}/export")
def export_case_board(case_id: str, request: CaseExportRequest):
    items = investigations.pinned_items(case_id)
    if not items:
        raise HTTPException(422, "case has no pinned evidence")
    try:
        result = forensics.create_case_bundle(case_id, request.officer, items)
        investigations.record_event(
            case_id, request.officer, "case_export_created",
            metadata={
                "bundle_id": result.bundle_id,
                "export_ids": result.export_ids,
                "tracklet_ids": [item["tracklet_id"] for item in items],
                "signing_key_id": result.signing_key_id,
            },
        )
    except forensics.ForensicExportError as exc:
        raise HTTPException(422, str(exc)) from exc
    return FileResponse(
        str(result.package_path), media_type="application/zip",
        filename=result.package_path.name,
        headers={"X-Forensic-Export-ID": result.bundle_id,
                 "X-Forensic-Key-ID": result.signing_key_id},
    )


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


@app.post("/forensics/export")
def forensic_export(request: ForensicExportRequest):
    """Build a signed ZIP containing source evidence and explicitly derived artifacts."""
    try:
        investigations.ensure_case(request.case_id, request.officer)
        result = forensics.create_export(request.model_dump())
        investigations.record_event(
            request.case_id, request.officer, "export_created",
            original_query=str(request.search.get("original_query") or "") or None,
            filters=request.search.get("filters") if isinstance(request.search.get("filters"), dict) else {},
            search_mode=str(request.retrieval.get("method") or "unspecified"),
            metadata={"export_id": result.export_id, "tracklet_id": request.tracklet_id,
                      "signing_key_id": result.signing_key_id},
        )
    except forensics.ForensicExportError as exc:
        status = 404 if "tracklet does not exist" in str(exc) else 422
        raise HTTPException(status, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "forensic export failed; no package was recorded") from exc
    return FileResponse(
        str(result.package_path),
        media_type="application/zip",
        filename=result.package_path.name,
        headers={
            "X-Forensic-Export-ID": result.export_id,
            "X-Forensic-Key-ID": result.signing_key_id,
        },
    )


@app.post("/forensics/verify")
def forensic_verify(package: UploadFile = File(..., description="forensic export ZIP")):
    """Verify hashes and signature against this deployment's trusted public key."""
    suffix = Path(package.filename or "package.zip").suffix
    if suffix.lower() != ".zip":
        raise HTTPException(422, "verification requires a ZIP package")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="forensic-verify-", suffix=".zip",
                                         delete=False) as temp:
            temp_path = Path(temp.name)
            total = 0
            while chunk := package.file.read(1024 * 1024):
                total += len(chunk)
                if total > 8 * 1024 * 1024 * 1024:
                    raise HTTPException(413, "package exceeds the 8 GiB verification limit")
                temp.write(chunk)
        _, public_key = forensics.ensure_signing_key()
        return forensics.verify_package(temp_path, trusted_public_key=public_key)
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
