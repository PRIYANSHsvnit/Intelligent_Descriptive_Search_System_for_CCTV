"""Per-camera media prep: produce ONE web-playable H.264/MP4 per camera by probing
the source and doing the minimum (never a blanket transcode).

  - H.264/MP4 already faststart  -> copy as-is (zero cost)
  - H.264/MP4 without faststart   -> remux (-c copy +faststart), no re-encode
  - anything else (CityFlow AVI/msmpeg4v2, HEVC, ...) -> transcode to H.264

The UI seeks within this file to ts_start/ts_end; crops are the grid thumbnails.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from . import paths


def _probe(video) -> tuple[str, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name:format=format_name",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    codec = d.get("streams", [{}])[0].get("codec_name", "")
    fmt = d.get("format", {}).get("format_name", "")
    return codec, fmt


def _has_faststart(video) -> bool:
    # moov before mdat => faststart. Cheap heuristic via ffprobe on the first packets.
    r = subprocess.run(
        ["ffprobe", "-v", "trace", "-i", str(video)],
        capture_output=True, text=True,
    )
    log = r.stderr
    moov = log.find("type:'moov'")
    mdat = log.find("type:'mdat'")
    return moov != -1 and (mdat == -1 or moov < mdat)


def run(scene: str, cam: str, force: bool = False) -> dict:
    src = paths.cam_video(scene, cam)
    out = paths.cam_out(scene, cam) / "media"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{cam}.mp4"

    if dst.exists() and not force:
        return {"cam": cam, "action": "cached", "video_ref": paths.rel_key(dst)}

    codec, fmt = _probe(src)
    is_mp4 = "mp4" in fmt or "mov" in fmt

    if codec == "h264" and is_mp4 and _has_faststart(src):
        shutil.copy2(src, dst)
        action = "copy"
    elif codec == "h264" and is_mp4:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)],
            check=True, capture_output=True,
        )
        action = "remux"
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-g", "30", "-an",
             "-movflags", "+faststart", str(dst)],
            check=True, capture_output=True,
        )
        action = "transcode"

    return {"cam": cam, "action": action, "src_codec": codec, "video_ref": paths.rel_key(dst)}
