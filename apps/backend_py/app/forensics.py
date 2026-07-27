"""Tamper-evident CCTV evidence export and verification.

Package integrity has two layers:

1. SHA256SUMS binds every evidence/metadata artifact (except itself and the signature).
2. An Ed25519 signature binds SHA256SUMS to this deployment's signing key.

The source recording is copied byte-for-byte and never annotated. The selected clip,
annotated frame, report, and manifest are explicitly marked as derived artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from fpdf import FPDF
from PIL import Image, ImageDraw

from . import boxes, config, i18n

PACKAGE_ROOT = "case-export"
BUNDLE_ROOT = "case-bundle"
REQUIRED_FILES = {
    "original_or_source_clip.mp4",
    "selected_clip.mp4",
    "annotated_frame.jpg",
    "manifest.json",
    "report.pdf",
    "SHA256SUMS",
    "signature.sig",
    "public_key.pem",
}
HASHED_FILES = REQUIRED_FILES - {"SHA256SUMS", "signature.sig"}
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class ForensicExportError(RuntimeError):
    """Safe, user-facing export failure."""


@dataclass(frozen=True)
class ExportResult:
    export_id: str
    case_id: str
    package_path: Path
    manifest_sha256: str
    source_sha256: str
    signing_key_id: str


@dataclass(frozen=True)
class BundleResult:
    bundle_id: str
    case_id: str
    package_path: Path
    signing_key_id: str
    export_ids: list[str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_identifier(value: str, fallback: str) -> str:
    cleaned = _SAFE_ID.sub("-", value.strip()).strip(".-_")
    return cleaned[:80] or fallback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_hash(source: Path, target: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as src, target.open("xb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
            dst.write(chunk)
    shutil.copystat(source, target)
    return digest.hexdigest()


def _key_paths(key_dir: Path | None = None) -> tuple[Path, Path]:
    root = key_dir or config.FORENSIC_KEY_DIR
    return root / "ed25519-private.pem", root / "ed25519-public.pem"


def ensure_signing_key(key_dir: Path | None = None) -> tuple[Path, Path]:
    """Create deployment key material once; never rotate it implicitly."""
    private_path, public_path = _key_paths(key_dir)
    private_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if private_path.exists() and public_path.exists():
        return private_path, public_path
    if private_path.exists() != public_path.exists():
        raise ForensicExportError("incomplete forensic signing keypair")

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Exclusive creation avoids two workers silently replacing one another's key.
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(private_bytes)
    except Exception:
        private_path.unlink(missing_ok=True)
        raise
    public_path.write_bytes(public_bytes)
    public_path.chmod(0o644)
    return private_path, public_path


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ForensicExportError("forensic private key is not Ed25519")
    return key


def _load_public_key(data: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise ForensicExportError("forensic public key is not Ed25519")
    return key


def _key_id(public_bytes: bytes) -> str:
    return f"sha256:{hashlib.sha256(public_bytes).hexdigest()}"


def _load_evidence(tracklet_id: str) -> dict[str, Any] | None:
    sql = """
        SELECT t.tracklet_id, t.scene, t.camera_id, t.entity_type, t.subtype,
               t.color, t.ts_start_s, t.ts_end_s, t.video_ref, t.crop_refs,
               t.plate_text, t.plate_conf, t.global_id,
               COALESCE(
                 jsonb_agg(
                   jsonb_build_object(
                     'crop_index', c.crop_index, 'crop_ref', c.crop_ref,
                     'frame_no', c.frame_no, 'quality', c.quality
                   ) ORDER BY c.crop_index
                 ) FILTER (WHERE c.tracklet_id IS NOT NULL), '[]'::jsonb
               ) AS crops
        FROM tracklets t
        LEFT JOIN tracklet_crops c ON c.tracklet_id = t.tracklet_id
        WHERE t.tracklet_id = %s
        GROUP BY t.tracklet_id
    """
    with psycopg.connect(config.DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, (tracklet_id,))
            return cur.fetchone()


def _source_path(evidence: dict[str, Any]) -> Path:
    video_ref = evidence.get("video_ref")
    path = config.OUTPUT_ROOT / str(video_ref) if video_ref else (
        config.OUTPUT_ROOT / evidence["scene"] / evidence["camera_id"]
        / "media" / f"{evidence['camera_id']}.mp4"
    )
    try:
        path = path.resolve(strict=True)
        path.relative_to(config.OUTPUT_ROOT.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise ForensicExportError("source recording is missing or outside the evidence root") from exc
    if not path.is_file():
        raise ForensicExportError("source recording is not a regular file")
    return path


def _run_ffmpeg(args: list[str], operation: str) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *args],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        detail = stderr.decode("utf-8", "replace")[-500:].strip()
        raise ForensicExportError(f"could not {operation}: {detail or type(exc).__name__}") from exc


def _create_selected_clip(source: Path, target: Path, start_s: float, end_s: float) -> None:
    duration = max(0.25, end_s - start_s)
    _run_ffmpeg(
        ["-y", "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
         "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
         str(target)],
        "create selected evidence clip",
    )


def _extract_frame(source: Path, at_s: float) -> Image.Image:
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss",
             f"{at_s:.3f}", "-i", str(source), "-frames:v", "1", "-f", "image2pipe",
             "-vcodec", "png", "pipe:1"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        raise ForensicExportError("could not extract annotated evidence frame") from exc


def _nearest_box(tracklet_id: str, at_s: float) -> list[float] | None:
    timeline = boxes.tracklet_boxes(tracklet_id)
    if not timeline or not timeline.get("boxes"):
        return None
    return min(timeline["boxes"], key=lambda item: abs(float(item[0]) - at_s))


def _create_annotated_frame(
    source: Path, target: Path, evidence: dict[str, Any], at_s: float,
) -> dict[str, Any]:
    image = _extract_frame(source, at_s)
    draw = ImageDraw.Draw(image)
    box = _nearest_box(evidence["tracklet_id"], at_s)
    if box:
        _, x1, y1, x2, y2 = box
        width = max(3, round(min(image.size) / 240))
        draw.rectangle((x1, y1, x2, y2), outline=(255, 55, 55), width=width)
        label_y = max(0, int(y1) - 22)
        label = f"{evidence['tracklet_id']}  {evidence['subtype']}"
        label_w = max(180, 7 * len(label))
        draw.rectangle((x1, label_y, min(image.width, x1 + label_w), label_y + 22),
                       fill=(190, 20, 20))
        draw.text((x1 + 5, label_y + 5), label, fill="white")
    image.save(target, "JPEG", quality=95, optimize=True)
    return {
        "video_local_timestamp_s": round(at_s, 3),
        "bounding_box_xyxy": [round(float(v), 1) for v in box[1:]] if box else None,
        "annotation": "derived overlay; source recording remains unmodified",
    }


FONT_DIR = Path(__file__).resolve().parent / "fonts"
# Pillow on this deployment is built without libraqm, so it cannot shape Devanagari or
# Gujarati at all (no matra reordering, no conjunct formation). fpdf2 + uharfbuzz does
# real HarfBuzz shaping and embeds subset fonts, so the report is selectable text too.
_SCRIPT_FONTS = {
    "latin": ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
    "devanagari": ("NotoSansDevanagari-Regular.ttf", "NotoSansDevanagari-Bold.ttf"),
    "gujarati": ("NotoSansGujarati-Regular.ttf", "NotoSansGujarati-Bold.ttf"),
}
_INK = (28, 34, 44)
_HEADING = (18, 55, 105)
_BAND = (18, 28, 44)
_MUTED = (98, 108, 124)


def _wall_clock(scene: str, ts_s: float) -> str:
    """Scene-clock seconds -> 'YYYY-MM-DD HH:MM:SS.mmm'.

    Time-of-day is the real camera clock recorded at ingest; the date comes from
    `config.recording_date`. Seconds past 24h roll the date forward correctly.
    """
    date_iso, _ = config.recording_date(scene)
    try:
        base = datetime.strptime(date_iso, "%Y-%m-%d")
    except ValueError:
        base = datetime.strptime(config.INGEST_BASE_DATE, "%Y-%m-%d")
    stamp = base + timedelta(seconds=float(ts_s))
    return f"{stamp:%Y-%m-%d %H:%M:%S}.{stamp.microsecond // 1000:03d}"


class _ReportPDF(FPDF):
    """A4 report whose footer stays in the language of the page it sits on."""

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.case_id = ""
        self.lang = "en"
        # A section's first page is pinned here; continuation pages produced by an
        # auto page break are not, and fall back to the section language in self.lang.
        self.page_lang: dict[int, str] = {}
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)
        for family, (regular, bold) in _SCRIPT_FONTS.items():
            self.add_font(family, "", FONT_DIR / regular)
            self.add_font(family, "B", FONT_DIR / bold)
        # Every face must fall back to every other one: Indic Noto faces lack some Latin
        # punctuation (Gujarati has no em dash), and the English page has to render an
        # operator query that may itself be typed in Gujarati or Hindi. Without this,
        # HarfBuzz drops the unsupported runs silently.
        self.set_fallback_fonts(list(_SCRIPT_FONTS), exact_match=False)
        self.set_text_shaping(True)

    def footer(self) -> None:
        lang = self.page_lang.get(self.page_no(), self.lang)
        self.set_y(-14)
        self.set_font(i18n.LANGUAGE_SCRIPT[lang], "", 7.5)
        self.set_text_color(*_MUTED)
        self.cell(0, 6, f"{self.case_id}  ·  {i18n.LANGUAGE_NAMES[lang]}", align="L")
        self.cell(0, 6, f"{i18n.t('page_of', lang)} {self.page_no()}", align="R")


def _report_rows(summary: dict[str, Any], lang: str) -> list[tuple[str, str, str]]:
    """(kind, label, value) rows. Facts stay verbatim; only labels are translated."""
    rows: list[tuple[str, str, str]] = [
        ("kv", i18n.t("case_id", lang), summary["case_id"]),
        ("kv", i18n.t("export_id", lang), summary["export_id"]),
        ("kv", i18n.t("officer", lang), summary["officer"]),
        ("kv", i18n.t("report_generated", lang), summary["created_at"]),
        ("head", i18n.t("section_search", lang), ""),
        ("kv", i18n.t("query", lang), summary["query"] or i18n.t("query_absent", lang)),
        ("kv", i18n.t("filters", lang),
         json.dumps(summary["filters"], ensure_ascii=False, sort_keys=True)),
        ("kv", i18n.t("retrieval_method", lang), summary["retrieval_method"]),
    ]
    if summary.get("searched_at"):
        rows.append(("kv", i18n.t("searched_at", lang), summary["searched_at"]))
    rows += [
        ("head", i18n.t("section_evidence", lang), ""),
        ("kv", i18n.t("tracklet", lang), summary["tracklet_id"]),
        ("kv", i18n.t("camera", lang), f"{summary['camera_label']} ({summary['camera_id']})"),
        ("kv", i18n.t("scene", lang), summary["scene"]),
        ("kv", i18n.t("sighting_start", lang), summary["wall_start"]),
        ("kv", i18n.t("sighting_end", lang), summary["wall_end"]),
        ("kv", i18n.t("duration", lang),
         f"{summary['ts_end_s'] - summary['ts_start_s']:.3f} {i18n.t('seconds_short', lang)}"),
        ("kv", i18n.t("video_offset", lang),
         (f"{summary['video_start_s']:.3f} – {summary['video_end_s']:.3f} "
          f"{i18n.t('seconds_short', lang)}")),
        ("note", i18n.t("clock_note", lang).format(tz=config.RECORDING_TIMEZONE), ""),
        ("kv", i18n.t("entity", lang), i18n.entity_phrase(
            summary.get("color"), summary["subtype"], summary["entity_type"], lang)),
    ]
    if summary.get("plate"):
        rows.append(("kv", i18n.t("plate", lang), summary["plate"]))
    rows += [
        ("kv", i18n.t("source_sha256", lang), summary["source_sha256"]),
        ("head", i18n.t("section_integrity", lang), ""),
        ("body", i18n.t("integrity_source", lang), ""),
        ("body", i18n.t("integrity_derived", lang), ""),
        ("body", i18n.t("integrity_verify", lang), ""),
        ("kv", i18n.t("signing_key", lang), summary["signing_key_id"]),
    ]
    return rows


def _render_language_page(pdf: _ReportPDF, summary: dict[str, Any], lang: str) -> None:
    family = i18n.LANGUAGE_SCRIPT[lang]
    # add_page() flushes the previous page's footer, so pdf.lang may only advance after it.
    pdf.add_page()
    pdf.lang = lang
    pdf.page_lang[pdf.page_no()] = lang

    pdf.set_fill_color(*_BAND)
    pdf.rect(0, 0, 210, 34, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(family, "B", 15)
    pdf.set_xy(18, 8)
    pdf.cell(174, 8, i18n.t("report_title", lang))
    pdf.set_font(family, "", 9)
    pdf.set_text_color(170, 200, 245)
    pdf.set_xy(18, 18)
    pdf.cell(174, 5, i18n.t("report_subtitle", lang))
    pdf.set_xy(18, 24)
    pdf.cell(174, 5, i18n.LANGUAGE_NAMES[lang])

    pdf.set_xy(18, 41)
    pdf.set_text_color(*_MUTED)
    pdf.set_font(family, "", 7.8)
    pdf.multi_cell(174, 4, i18n.t("translation_note", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    label_w, value_w, line_h = 48.0, 126.0, 5.2
    for kind, left, right in _report_rows(summary, lang):
        if kind == "head":
            pdf.ln(2.5)
            pdf.set_font(family, "B", 11)
            pdf.set_text_color(*_HEADING)
            pdf.cell(0, 6.5, left, new_x="LMARGIN", new_y="NEXT")
            y = pdf.get_y()
            pdf.set_draw_color(*_HEADING)
            pdf.set_line_width(0.3)
            pdf.line(18, y, 192, y)
            pdf.ln(2)
            continue
        if kind in ("body", "note"):
            if kind == "note":
                pdf.ln(1.5)
            pdf.set_font(family, "", 8.2 if kind == "note" else 8.8)
            pdf.set_text_color(*(_MUTED if kind == "note" else _INK))
            pdf.multi_cell(174, line_h, left, new_x="LMARGIN", new_y="NEXT")
            continue

        # Two-column key/value: measure both sides first, so a wrapped value (a 64-char
        # hash, a long query) cannot desynchronise the next row's baseline.
        pdf.set_font(family, "B", 8.8)
        label_lines = pdf.multi_cell(label_w, line_h, left, dry_run=True, output="LINES")
        pdf.set_font(family, "", 8.8)
        value_lines = pdf.multi_cell(value_w, line_h, right, dry_run=True, output="LINES")
        height = line_h * max(len(label_lines), len(value_lines), 1)
        if pdf.get_y() + height > pdf.page_break_trigger:
            pdf.add_page()
        top = pdf.get_y()
        pdf.set_font(family, "B", 8.8)
        pdf.set_text_color(*_HEADING)
        pdf.set_xy(18, top)
        pdf.multi_cell(label_w, line_h, left, new_x="RIGHT", new_y="TOP")
        pdf.set_font(family, "", 8.8)
        pdf.set_text_color(*_INK)
        pdf.set_xy(18 + label_w, top)
        pdf.multi_cell(value_w, line_h, right, new_x="LMARGIN", new_y="TOP")
        pdf.set_xy(18, top + height)


def _create_report(path: Path, summary: dict[str, Any]) -> None:
    """Trilingual (English / Hindi / Gujarati) PDF report, one page per language.

    English is rendered first as the authoritative text; every language section carries
    the identical set of recorded facts.
    """
    pdf = _ReportPDF()
    pdf.case_id = str(summary["case_id"])
    pdf.set_title(f"CCTV forensic export report - {summary['case_id']}")
    pdf.set_lang("en")
    for lang in i18n.LANGUAGES:
        _render_language_page(pdf, summary, lang)
    pdf.output(str(path))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _record_export(result: ExportResult, manifest: dict[str, Any]) -> None:
    sql = """
        INSERT INTO forensic_exports (
          export_id, case_id, officer, tracklet_id, created_at, package_path,
          manifest_sha256, source_sha256, signing_key_id, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with psycopg.connect(config.DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                result.export_id, result.case_id, manifest["officer"], manifest["evidence"]["tracklet_id"],
                manifest["created_at_utc"], str(result.package_path), result.manifest_sha256,
                result.source_sha256, result.signing_key_id, Jsonb(manifest),
            ))


def create_export(request: dict[str, Any]) -> ExportResult:
    evidence = _load_evidence(str(request.get("tracklet_id", "")))
    if not evidence:
        raise ForensicExportError("tracklet does not exist")
    source = _source_path(evidence)
    export_id = str(uuid.uuid4())
    case_id = _safe_identifier(str(request.get("case_id", "")), f"CASE-{export_id[:8]}")
    officer = str(request.get("officer", "")).strip()[:120] or "Unspecified officer"
    created_at = _utc_now()
    export_root = config.FORENSIC_EXPORT_ROOT
    export_root.mkdir(parents=True, exist_ok=True)
    private_path, public_path = ensure_signing_key()
    private_key = _load_private_key(private_path)
    public_bytes = public_path.read_bytes()
    signing_key_id = _key_id(public_bytes)

    with tempfile.TemporaryDirectory(prefix="forensic-", dir=export_root) as temp_name:
        stage = Path(temp_name) / PACKAGE_ROOT
        stage.mkdir()
        source_copy = stage / "original_or_source_clip.mp4"
        source_sha = _copy_and_hash(source, source_copy)

        video_start = max(0.0, float(evidence["ts_start_s"])
                          - config.camera_offset(evidence["scene"], evidence["camera_id"]))
        video_end = max(video_start, float(evidence["ts_end_s"])
                        - config.camera_offset(evidence["scene"], evidence["camera_id"]))
        clip_start = max(0.0, video_start - config.FORENSIC_CLIP_PADDING_S)
        clip_end = video_end + config.FORENSIC_CLIP_PADDING_S
        _create_selected_clip(source, stage / "selected_clip.mp4", clip_start, clip_end)
        frame_at = (video_start + video_end) / 2.0
        frame_meta = _create_annotated_frame(
            source, stage / "annotated_frame.jpg", evidence, frame_at)

        search = request.get("search") or {}
        retrieval = request.get("retrieval") or {}
        recording_date, date_configured = config.recording_date(evidence["scene"])
        summary = {
            "case_id": case_id,
            "export_id": export_id,
            "officer": officer,
            "created_at": _iso(created_at),
            "query": str(search.get("original_query", ""))[:1000],
            "filters": search.get("filters") if isinstance(search.get("filters"), dict) else {},
            "retrieval_method": str(retrieval.get("method", "unspecified"))[:120],
            "searched_at": search.get("timestamp_utc"),
            "tracklet_id": evidence["tracklet_id"],
            "camera_label": config.camera_label(evidence["scene"], evidence["camera_id"]),
            "camera_id": evidence["camera_id"],
            "scene": evidence["scene"],
            "ts_start_s": float(evidence["ts_start_s"]),
            "ts_end_s": float(evidence["ts_end_s"]),
            "wall_start": _wall_clock(evidence["scene"], evidence["ts_start_s"]),
            "wall_end": _wall_clock(evidence["scene"], evidence["ts_end_s"]),
            "video_start_s": video_start,
            "video_end_s": video_end,
            "entity_type": evidence["entity_type"],
            "subtype": evidence["subtype"],
            "color": evidence.get("color"),
            "plate": evidence.get("plate_text"),
            "source_sha256": source_sha,
            "signing_key_id": signing_key_id,
        }
        _create_report(stage / "report.pdf", summary)
        (stage / "public_key.pem").write_bytes(public_bytes)

        artifact_roles = {
            "original_or_source_clip.mp4": "source",
            "selected_clip.mp4": "derived",
            "annotated_frame.jpg": "derived",
            "report.pdf": "derived",
            "public_key.pem": "verification_material",
        }
        artifact_hashes = {
            name: {"sha256": _sha256_file(stage / name), "role": role,
                   "size_bytes": (stage / name).stat().st_size}
            for name, role in artifact_roles.items()
        }
        crops = list(evidence.get("crops") or [])
        selected_crop_ref = str(request.get("selected_crop_ref", ""))
        valid_crop_refs = {str(c.get("crop_ref")) for c in crops}
        if selected_crop_ref not in valid_crop_refs:
            selected_crop_ref = str(crops[0].get("crop_ref")) if crops else None

        manifest = {
            "schema": "surat-cctv-forensic-export/v1",
            "case_id": case_id,
            "export_id": export_id,
            "created_at_utc": _iso(created_at),
            "officer": officer,
            "search": {
                "original_query": summary["query"],
                "rewritten_query": str(search.get("rewritten_query", ""))[:1000] or None,
                "search_timestamp_utc": search.get("timestamp_utc"),
                "filters": summary["filters"],
            },
            "evidence": {
                "tracklet_id": evidence["tracklet_id"],
                "crop_ids": crops,
                "selected_crop_ref": selected_crop_ref,
                "scene": evidence["scene"],
                "camera_id": evidence["camera_id"],
                "location": summary["camera_label"],
                "source_timestamp_start_s": summary["ts_start_s"],
                "source_timestamp_end_s": summary["ts_end_s"],
                "sighting_start_wall_clock": summary["wall_start"],
                "sighting_end_wall_clock": summary["wall_end"],
                "wall_clock": {
                    "timezone": config.RECORDING_TIMEZONE,
                    "recording_date": recording_date,
                    "recording_date_source": (
                        "deployment configuration (RECORDING_DATES)" if date_configured
                        else "ingest base date placeholder; not read from source metadata"
                    ),
                    "time_of_day_source": "camera scene clock recorded at ingest",
                },
                "video_local_timestamp_start_s": round(video_start, 3),
                "video_local_timestamp_end_s": round(video_end, 3),
                "selected_clip_start_s": round(clip_start, 3),
                "selected_clip_end_s": round(clip_end, 3),
                "source_video_ref": evidence.get("video_ref"),
                "source_recording_scope": (
                    "byte-for-byte copy of the indexed camera media referenced by video_ref; "
                    "the indexed media may have been normalized during ingestion"
                ),
                "entity_type": evidence["entity_type"],
                "subtype": evidence["subtype"],
                "color": evidence.get("color"),
                "plate": evidence.get("plate_text"),
                "global_id": evidence.get("global_id"),
                "annotated_frame": frame_meta,
            },
            "report": {
                "file": "report.pdf",
                "languages": list(i18n.LANGUAGES),
                "authoritative_language": "en",
                "translation_scope": (
                    "fixed labels and detector vocabulary only; identifiers, hashes, "
                    "filters, timestamps and the operator query are verbatim in every language"
                ),
            },
            "models": config.forensic_model_inventory(evidence["scene"]),
            "retrieval": {
                "method": summary["retrieval_method"],
                "aggregation": retrieval.get("aggregation"),
                "composition": retrieval.get("composition"),
                "search_captions": retrieval.get("search_captions") or [],
                "result_score": retrieval.get("result_score"),
                "component_scores": retrieval.get("component_scores") or [],
            },
            "integrity": {
                "hash_algorithm": "SHA-256",
                "signature_algorithm": "Ed25519",
                "signing_key_id": signing_key_id,
                "signed_file": "SHA256SUMS",
                "manifest_integrity": "manifest.json is hashed by SHA256SUMS",
                "source_recording_sha256": source_sha,
            },
            "artifacts": artifact_hashes,
        }
        _write_json(stage / "manifest.json", manifest)

        sums_names = sorted(HASHED_FILES)
        sums = "".join(f"{_sha256_file(stage / name)}  {name}\n" for name in sums_names)
        sums_bytes = sums.encode("utf-8")
        (stage / "SHA256SUMS").write_bytes(sums_bytes)
        signature = private_key.sign(sums_bytes)
        (stage / "signature.sig").write_text(
            base64.b64encode(signature).decode("ascii") + "\n", encoding="ascii")

        package_name = f"{case_id}_{export_id}.zip"
        package_path = export_root / package_name
        with zipfile.ZipFile(package_path, "x", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for path in sorted(stage.iterdir()):
                archive.write(path, f"{PACKAGE_ROOT}/{path.name}")

    result = ExportResult(
        export_id=export_id,
        case_id=case_id,
        package_path=package_path,
        manifest_sha256=_sha256_file_from_zip(package_path, "manifest.json"),
        source_sha256=source_sha,
        signing_key_id=signing_key_id,
    )
    try:
        _record_export(result, manifest)
    except Exception:
        package_path.unlink(missing_ok=True)
        raise
    return result


def create_case_bundle(
    case_id: str, officer: str, items: list[dict[str, Any]],
) -> BundleResult:
    """Create a signed collection whose children remain independently verifiable exports."""
    if not items:
        raise ForensicExportError("case has no pinned evidence")
    bundle_id = str(uuid.uuid4())
    safe_case_id = _safe_identifier(case_id, f"CASE-{bundle_id[:8]}")
    private_path, public_path = ensure_signing_key()
    private_key = _load_private_key(private_path)
    public_bytes = public_path.read_bytes()
    key_id = _key_id(public_bytes)
    export_root = config.FORENSIC_EXPORT_ROOT
    export_root.mkdir(parents=True, exist_ok=True)
    children: list[tuple[str, ExportResult, dict[str, Any]]] = []
    for index, item in enumerate(items, start=1):
        snapshot = item.get("result_snapshot") or {}
        request = {
            "tracklet_id": item["tracklet_id"],
            "case_id": safe_case_id,
            "officer": officer,
            "selected_crop_ref": snapshot.get("matched_crop_ref"),
            "search": snapshot.get("search") or {},
            "retrieval": snapshot.get("retrieval") or {},
        }
        child = create_export(request)
        child_name = f"evidence-{index:03d}-{_safe_identifier(item['tracklet_id'], 'tracklet')}.zip"
        children.append((child_name, child, item))

    with tempfile.TemporaryDirectory(prefix="case-bundle-", dir=export_root) as temp_name:
        stage = Path(temp_name) / BUNDLE_ROOT
        stage.mkdir()
        (stage / "public_key.pem").write_bytes(public_bytes)
        evidence_inventory = []
        for name, child, item in children:
            shutil.copy2(child.package_path, stage / name)
            evidence_inventory.append({
                "tracklet_id": item["tracklet_id"],
                "status": "pinned",
                "note": item.get("note"),
                "child_export_id": child.export_id,
                "file": name,
                "sha256": _sha256_file(stage / name),
                "source_sha256": child.source_sha256,
            })
        manifest = {
            "schema": "surat-cctv-forensic-case-bundle/v1",
            "bundle_id": bundle_id,
            "case_id": safe_case_id,
            "officer": officer,
            "created_at_utc": _iso(_utc_now()),
            "selection_policy": "only case items whose current status was pinned",
            "evidence": evidence_inventory,
            "integrity": {
                "hash_algorithm": "SHA-256",
                "signature_algorithm": "Ed25519",
                "signing_key_id": key_id,
                "signed_file": "SHA256SUMS",
                "child_packages": "each child is also an independently signed case export",
            },
        }
        _write_json(stage / "manifest.json", manifest)
        hashed = ["manifest.json", "public_key.pem", *(name for name, _, _ in children)]
        sums_bytes = "".join(
            f"{_sha256_file(stage / name)}  {name}\n" for name in sorted(hashed)
        ).encode("utf-8")
        (stage / "SHA256SUMS").write_bytes(sums_bytes)
        (stage / "signature.sig").write_text(
            base64.b64encode(private_key.sign(sums_bytes)).decode("ascii") + "\n",
            encoding="ascii",
        )
        package_path = export_root / f"{safe_case_id}_bundle_{bundle_id}.zip"
        with zipfile.ZipFile(package_path, "x", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for path in sorted(stage.iterdir()):
                archive.write(path, f"{BUNDLE_ROOT}/{path.name}")
    return BundleResult(
        bundle_id=bundle_id, case_id=safe_case_id, package_path=package_path,
        signing_key_id=key_id, export_ids=[child.export_id for _, child, _ in children],
    )


def _entry_name(name: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        return None
    if path.parts[0] != PACKAGE_ROOT:
        return None
    return path.parts[1]


def _sha256_zip_entry(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_from_zip(package: Path, name: str) -> str:
    with zipfile.ZipFile(package) as archive:
        return _sha256_zip_entry(archive, archive.getinfo(f"{PACKAGE_ROOT}/{name}"))


def _parse_sums(data: bytes) -> tuple[dict[str, str], list[str]]:
    parsed: dict[str, str] = {}
    errors: list[str] = []
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return {}, ["SHA256SUMS is not UTF-8 text"]
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if not match:
            errors.append("SHA256SUMS contains an invalid line")
            continue
        digest, name = match.groups()
        if name in parsed:
            errors.append(f"duplicate checksum entry: {name}")
        parsed[name] = digest
    return parsed, errors


def verify_package(package: Path, trusted_public_key: Path | None = None) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(package) as archive:
            roots = {PurePosixPath(member.filename).parts[0] for member in archive.infolist()
                     if not member.is_dir() and PurePosixPath(member.filename).parts}
        if roots == {BUNDLE_ROOT}:
            return _verify_case_bundle(package, trusted_public_key)
    except (FileNotFoundError, zipfile.BadZipFile, OSError):
        pass
    return _verify_single_export(package, trusted_public_key)


def _verify_single_export(package: Path, trusted_public_key: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    export_id = None
    key_id = None
    try:
        with zipfile.ZipFile(package) as archive:
            members: dict[str, zipfile.ZipInfo] = {}
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = _entry_name(member.filename)
                if name is None:
                    errors.append(f"unsafe or unexpected ZIP path: {member.filename}")
                    continue
                if name in members:
                    errors.append(f"duplicate ZIP entry: {name}")
                if member.compress_type != zipfile.ZIP_STORED:
                    errors.append(f"compressed entries are not accepted: {name}")
                members[name] = member
            if sum(member.file_size for member in members.values()) > 8 * 1024 * 1024 * 1024:
                errors.append("uncompressed package content exceeds the 8 GiB limit")
            missing = sorted(REQUIRED_FILES - members.keys())
            if missing:
                errors.append(f"missing required files: {', '.join(missing)}")
            extra = sorted(members.keys() - REQUIRED_FILES)
            if extra:
                errors.append(f"unexpected package files: {', '.join(extra)}")
            if errors:
                return _verification_result(errors, warnings, export_id, key_id)

            sums_bytes = archive.read(members["SHA256SUMS"])
            expected, parse_errors = _parse_sums(sums_bytes)
            errors.extend(parse_errors)
            missing_sums = sorted(HASHED_FILES - expected.keys())
            extra_sums = sorted(expected.keys() - HASHED_FILES)
            if missing_sums:
                errors.append(f"missing checksum entries: {', '.join(missing_sums)}")
            if extra_sums:
                errors.append(f"unexpected checksum entries: {', '.join(extra_sums)}")
            for name in sorted(HASHED_FILES & expected.keys()):
                actual = _sha256_zip_entry(archive, members[name])
                if actual != expected[name]:
                    errors.append(f"checksum mismatch: {name}")

            embedded_public = archive.read(members["public_key.pem"])
            trusted_bytes = trusted_public_key.read_bytes() if trusted_public_key else embedded_public
            if trusted_public_key and trusted_bytes != embedded_public:
                errors.append("embedded public key does not match the trusted deployment key")
            key_id = _key_id(trusted_bytes)
            try:
                signature = base64.b64decode(
                    archive.read(members["signature.sig"]).strip(), validate=True)
                _load_public_key(trusted_bytes).verify(signature, sums_bytes)
            except (ValueError, InvalidSignature, ForensicExportError):
                errors.append("Ed25519 signature verification failed")
            if not trusted_public_key:
                warnings.append("signer identity was not pinned; integrity is valid against the embedded key")

            try:
                manifest = json.loads(archive.read(members["manifest.json"]))
                export_id = manifest.get("export_id")
                if manifest.get("integrity", {}).get("signing_key_id") != _key_id(embedded_public):
                    errors.append("manifest signing key ID does not match public_key.pem")
                if manifest.get("integrity", {}).get("source_recording_sha256") != expected.get(
                        "original_or_source_clip.mp4"):
                    errors.append("manifest source hash does not match SHA256SUMS")
                for name, metadata in manifest.get("artifacts", {}).items():
                    if name not in expected or metadata.get("sha256") != expected[name]:
                        errors.append(f"manifest artifact hash mismatch: {name}")
                declared = set(manifest.get("artifacts", {}))
                expected_artifacts = HASHED_FILES - {"manifest.json"}
                if declared != expected_artifacts:
                    errors.append("manifest artifact inventory is incomplete or unexpected")
            except (json.JSONDecodeError, AttributeError, TypeError):
                errors.append("manifest.json is invalid")
    except (FileNotFoundError, zipfile.BadZipFile, OSError) as exc:
        errors.append(f"could not read package: {exc}")
    return _verification_result(errors, warnings, export_id, key_id)


def _verify_case_bundle(package: Path, trusted_public_key: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    bundle_id = None
    key_id = None
    try:
        with zipfile.ZipFile(package) as archive:
            members: dict[str, zipfile.ZipInfo] = {}
            for member in archive.infolist():
                if member.is_dir():
                    continue
                path = PurePosixPath(member.filename)
                if (path.is_absolute() or ".." in path.parts or len(path.parts) != 2
                        or path.parts[0] != BUNDLE_ROOT):
                    errors.append(f"unsafe or unexpected ZIP path: {member.filename}")
                    continue
                name = path.parts[1]
                if name in members:
                    errors.append(f"duplicate ZIP entry: {name}")
                if member.compress_type != zipfile.ZIP_STORED:
                    errors.append(f"compressed entries are not accepted: {name}")
                members[name] = member
            core = {"manifest.json", "SHA256SUMS", "signature.sig", "public_key.pem"}
            children = {name for name in members if re.fullmatch(r"evidence-\d{3}-.+\.zip", name)}
            unexpected = set(members) - core - children
            if core - set(members):
                errors.append("case bundle is missing integrity files")
            if unexpected:
                errors.append(f"unexpected bundle files: {', '.join(sorted(unexpected))}")
            if not children:
                errors.append("case bundle contains no evidence packages")
            if sum(member.file_size for member in members.values()) > 8 * 1024 * 1024 * 1024:
                errors.append("uncompressed package content exceeds the 8 GiB limit")
            if errors:
                return _verification_result(errors, warnings, bundle_id, key_id)

            sums_bytes = archive.read(members["SHA256SUMS"])
            expected, parse_errors = _parse_sums(sums_bytes)
            errors.extend(parse_errors)
            hashed = {"manifest.json", "public_key.pem", *children}
            if set(expected) != hashed:
                errors.append("bundle checksum inventory is incomplete or unexpected")
            for name in sorted(hashed & set(expected)):
                if _sha256_zip_entry(archive, members[name]) != expected[name]:
                    errors.append(f"checksum mismatch: {name}")

            embedded_public = archive.read(members["public_key.pem"])
            trusted_bytes = trusted_public_key.read_bytes() if trusted_public_key else embedded_public
            if trusted_public_key and trusted_bytes != embedded_public:
                errors.append("embedded public key does not match the trusted deployment key")
            key_id = _key_id(trusted_bytes)
            try:
                signature = base64.b64decode(
                    archive.read(members["signature.sig"]).strip(), validate=True)
                _load_public_key(trusted_bytes).verify(signature, sums_bytes)
            except (ValueError, InvalidSignature, ForensicExportError):
                errors.append("Ed25519 signature verification failed")
            if not trusted_public_key:
                warnings.append("signer identity was not pinned; integrity is valid against the embedded key")

            try:
                manifest = json.loads(archive.read(members["manifest.json"]))
                bundle_id = manifest.get("bundle_id")
                declared = {item["file"]: item for item in manifest.get("evidence", [])}
                if set(declared) != children:
                    errors.append("bundle manifest evidence inventory is incomplete or unexpected")
                if manifest.get("integrity", {}).get("signing_key_id") != _key_id(embedded_public):
                    errors.append("manifest signing key ID does not match public_key.pem")
                for name, item in declared.items():
                    if item.get("sha256") != expected.get(name):
                        errors.append(f"manifest artifact hash mismatch: {name}")
            except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
                errors.append("manifest.json is invalid")

            # Verify every nested evidence export, not just its parent hash.
            for name in sorted(children):
                with tempfile.NamedTemporaryFile(prefix="nested-evidence-", suffix=".zip") as child:
                    with archive.open(members[name]) as nested:
                        shutil.copyfileobj(nested, child, length=1024 * 1024)
                    child.flush()
                    child_result = _verify_single_export(Path(child.name), trusted_public_key)
                if not child_result["valid"]:
                    errors.append(f"nested export failed verification: {name}")
    except (FileNotFoundError, zipfile.BadZipFile, OSError) as exc:
        errors.append(f"could not read package: {exc}")
    return _verification_result(errors, warnings, bundle_id, key_id)


def _verification_result(
    errors: list[str], warnings: list[str], export_id: str | None, key_id: str | None,
) -> dict[str, Any]:
    valid = not errors
    return {
        "valid": valid,
        "status": "VALID" if valid else "TAMPERED",
        "export_id": export_id,
        "signing_key_id": key_id,
        "errors": errors,
        "warnings": warnings,
        "verified_at_utc": _iso(_utc_now()),
    }


def create_tampered_demo(source: Path, target: Path) -> Path:
    """Create a separate deliberately altered ZIP for a safe live verification demo."""
    if source.resolve() == target.resolve():
        raise ForensicExportError("tamper demo output must not overwrite the original package")
    if target.exists():
        raise ForensicExportError("tamper demo output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(
                target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as altered:
            for member in original.infolist():
                with original.open(member) as src, altered.open(member, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                    if member.filename == f"{PACKAGE_ROOT}/selected_clip.mp4":
                        dst.write(b"TAMPER-DEMO")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Verify a CCTV forensic export ZIP")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="recompute hashes and verify the Ed25519 signature")
    verify.add_argument("package", type=Path)
    verify.add_argument("--public-key", type=Path, help="trusted deployment public key PEM")
    tamper = sub.add_parser("tamper-demo", help="write a separate intentionally altered ZIP")
    tamper.add_argument("package", type=Path)
    tamper.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "tamper-demo":
        print(create_tampered_demo(args.package, args.output))
        return 0
    result = verify_package(args.package, args.public_key)
    print(result["status"])
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
