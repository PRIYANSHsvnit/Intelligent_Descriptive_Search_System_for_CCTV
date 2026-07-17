"""Stage (person attributes / VLM): Qwen3-VL-4B zero-shot attributes via llama.cpp.

Supervised closed-set attribute classifiers are forced to answer every crop and are
miscalibrated under domain shift, so on Surat footage their wrong answers arrive with
high confidence. A small VLM fixes that structurally — "unknown" is an allowed answer,
so bad crops abstain instead of mislabeling (validated on SUR01/c004, see the
qwen3vl-person-attrs memory / plan.md "Person attributes").

Per PERSON tracklet, ALL of its K crops go in ONE multi-image request: the prompt states
they show the same tracked person, which lets the model (a) reject bystanders that appear
in only some crops — a single-crop run blended a helmeted scooter rider behind the subject
into "headwear: helmet" — and (b) combine views (a backpack may only be visible from
behind). This stage is the SOLE source of `person_attrs` (the region-split SigLIP color
stage was removed: fixed torso/legs crop fractions mean-pooled bystanders into wrong
labels — e.g. a second person in 1 of 3 crops — and with margin 0 it never abstained, so
its mislabels outranked better VLM answers). Writes into `person_attrs` JSONB:

  apparent_gender / age / backpack / headwear  ->  {"name": ...}   ("unknown" -> key omitted)
  upper_color / lower_color                    ->  {"name": ...}, vocab-constrained in the
                                                   prompt; off-vocab answers abstain

No confidence field: the VLM emits a label, not a calibrated prob. SOFT ranking signal
only, never a hard filter (gender/age especially — unscored until the eval set exists).

Runs llama-server (nix CUDA build, models/llama-cpp-cuda) as a child process, started
lazily on the first cam and kept for the whole stage pass; run_ingest calls shutdown()
after the pass so the ~4.7 GB of VRAM is freed before the next GPU stage. llama.cpp is
not torch — no gpu_setup shim needed. Skips gracefully if the binary or GGUFs are missing.

Throughput: the server runs VLM_PARALLEL continuous-batching slots and this stage fires
that many requests concurrently. On the 6 GB 4050, serial was ~1.3 s/tracklet; 4 slots +
q8_0 KV cache gives ~0.39 s/tracklet (~3.3x, plateaus at 4 — the card's compute ceiling).
KV must be quantized (-ctk/-ctv q8_0): 4 slots x 2048 ctx in f16 OOMs the 6 GB card; q8
halves the KV footprint so it fits (measured ~5.7/6.1 GB). Per-slot ctx must still hold
K_CROPS images + prompt, so total -c = VLM_CTX_PER_SLOT * VLM_PARALLEL.
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import paths

# --- model / server locations (env-overridable) -------------------------------
_MODELS = paths.REPO_ROOT / "models"
VLM_SERVER_BIN = os.environ.get(
    "VLM_SERVER_BIN", str(_MODELS / "llama-cpp-cuda" / "bin" / "llama-server"))
VLM_GGUF = os.environ.get(
    "VLM_GGUF", str(_MODELS / "QWEN VLM" / "Qwen3-VL-4B-Instruct-UD-Q6_K_XL.gguf"))
VLM_MMPROJ = os.environ.get("VLM_MMPROJ", str(_MODELS / "QWEN VLM" / "mmproj-F16.gguf"))
VLM_PORT = int(os.environ.get("VLM_PORT", "8090"))
VLM_PARALLEL = int(os.environ.get("VLM_PARALLEL", "4"))   # concurrent slots; 4 saturates the 4050
# Persons tracked for fewer frames than this are skipped: a 1-2 frame "person" in a
# crowd is tracker flicker whose crops the VLM answers all-"unknown" on anyway — the
# tracklet stays stored/searchable (SigLIP vector), it just gets no attribute chips.
VLM_MIN_DETECTIONS = int(os.environ.get("VLM_MIN_DETECTIONS", "3"))
VLM_CTX_PER_SLOT = 2048        # per slot: fits K_CROPS images + prompt (default ctx OOMs 6 GB)
VLM_CTX = VLM_CTX_PER_SLOT * VLM_PARALLEL   # llama.cpp splits total -c across the -np slots
VLM_MAX_TOKENS = 120
VLM_LOAD_TIMEOUT_S = 180       # model load + warmup before /health goes ok
VLM_REQUEST_TIMEOUT_S = 120

_URL = f"http://127.0.0.1:{VLM_PORT}/v1/chat/completions"
_HEALTH = f"http://127.0.0.1:{VLM_PORT}/health"

# Clothing palette (kept from the removed SigLIP region stage: no vehicle-ish
# "silver", has pink/purple). Baked into the prompt so colors stay canonical /
# filterable; an off-vocab answer is treated as an abstain at merge time.
VLM_COLOR_VOCAB = (
    "white", "black", "gray", "red", "blue", "green",
    "yellow", "orange", "brown", "pink", "purple",
)
_COLOR_CHOICES = "/".join(f'"{c}"' for c in VLM_COLOR_VOCAB)

PROMPT = (
    "These images are crops of the SAME person tracked across consecutive CCTV frames "
    "(same camera, moments apart, the angle may change). Other people may partially "
    "appear in some crops; the subject is the person visible in ALL of them, roughly "
    "centered. Combine evidence from every view (e.g. a backpack may only be visible "
    'from behind; if a clear rear view shows no backpack, answer "no"). Answer ONLY '
    'with a JSON object, no other text. Use "unknown" whenever the crops are too '
    'small, blurry, or occluded to tell. Keys: gender ("male"/"female"/"unknown"), '
    'age_group ("child"/"adult"/"elderly"/"unknown"), backpack ("yes"/"no"/"unknown"), '
    'headwear ("none"/"cap"/"helmet"/"turban"/"scarf"/"unknown"), upper_color and '
    f"lower_color (each one of: {_COLOR_CHOICES}/\"unknown\")."
)

# VLM response key -> person_attrs key (gender is presentation, never identity).
_ATTR_KEYS = {
    "gender": "apparent_gender",
    "age_group": "age",
    "backpack": "backpack",
    "headwear": "headwear",
}
_COLOR_KEYS = ("upper_color", "lower_color")
# Keys this stage OWNS: cleared before merging so stale values from a previous attribute
# pass never survive next to fresh answers; an abstained attribute must end up absent,
# not outdated. Colors are owned too since the SigLIP region stage was removed — old
# SigLIP-written labels must not survive a re-run. (carrying/footwear/…types are legacy
# keys of removed stages.)
_OWNED_KEYS = (*_ATTR_KEYS.values(), *_COLOR_KEYS,
               "carrying", "footwear", "upper_type", "lower_type")

_proc: subprocess.Popen | None = None


def _server_ready() -> bool:
    try:
        with urllib.request.urlopen(_HEALTH, timeout=2) as r:
            return b'"ok"' in r.read()
    except (urllib.error.URLError, OSError):
        return False


def _ensure_server() -> str | None:
    """Start llama-server once for the whole stage pass. Returns an error string or None."""
    global _proc
    if _proc is not None and _proc.poll() is None and _server_ready():
        return None
    for p, what in ((VLM_SERVER_BIN, "llama-server"), (VLM_GGUF, "model gguf"),
                    (VLM_MMPROJ, "mmproj gguf")):
        if not os.path.exists(p):
            return f"no {what} at {p}"
    log = open(paths.OUTPUT_ROOT / "vlm_server.log", "ab")  # noqa: SIM115 - outlives scope
    _proc = subprocess.Popen(
        [VLM_SERVER_BIN, "-m", VLM_GGUF, "--mmproj", VLM_MMPROJ,
         "-ngl", "99", "-c", str(VLM_CTX), "-np", str(VLM_PARALLEL), "-cb",
         "-ctk", "q8_0", "-ctv", "q8_0",      # q8 KV: 4 slots x 2048 in f16 OOMs 6 GB
         "--host", "127.0.0.1", "--port", str(VLM_PORT)],
        stdout=log, stderr=log)
    atexit.register(shutdown)
    deadline = time.time() + VLM_LOAD_TIMEOUT_S
    while time.time() < deadline:
        if _proc.poll() is not None:
            return f"llama-server exited rc={_proc.returncode} (see output/vlm_server.log)"
        if _server_ready():
            return None
        time.sleep(1)
    shutdown()
    return f"llama-server not ready after {VLM_LOAD_TIMEOUT_S}s (see output/vlm_server.log)"


def shutdown() -> None:
    """Free the ~4.7 GB of VRAM; run_ingest calls this after the stage pass."""
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def _ask(crop_paths: list[str]) -> dict | None:
    """One multi-image chat request -> parsed attr dict (None on transport/parse failure)."""
    content: list[dict] = []
    for p in crop_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": PROMPT})
    body = json.dumps({"temperature": 0, "max_tokens": VLM_MAX_TOKENS,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(_URL, body, {"Content-Type": "application/json"})
    for attempt in (1, 2):                      # one retry on transient failure
        try:
            with urllib.request.urlopen(req, timeout=VLM_REQUEST_TIMEOUT_S) as r:
                resp = json.load(r)
            txt = resp["choices"][0]["message"]["content"].strip()
            txt = txt.removeprefix("```json").removesuffix("```").strip()
            out = json.loads(txt)
            return out if isinstance(out, dict) else None
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            if attempt == 2:
                return None
            time.sleep(1)
    return None


def run(scene: str, cam: str) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    all_persons = [(i, t) for i, t in enumerate(tracklets) if t.get("entity_type") == "person"]
    persons = [(i, t) for i, t in all_persons if t["num_detections"] >= VLM_MIN_DETECTIONS]
    skipped_short = len(all_persons) - len(persons)
    if not persons:
        return {"cam": cam, "persons": 0, "skipped_short": skipped_short}

    err = _ensure_server()
    if err:
        return {"cam": cam, "skipped": err}

    t0 = time.time()
    labeled = failures = colors = 0

    # Resolve each person's crops up front, then fan the requests out across the server's
    # VLM_PARALLEL slots. _ask is pure/thread-safe; the merge below stays single-threaded so
    # the tracklets/counter mutations never race.
    work = [(i, crops) for i, t in persons
            if (crops := [str(paths.OUTPUT_ROOT / k) for k in t["crop_refs"]
                          if (paths.OUTPUT_ROOT / k).exists()])]
    with ThreadPoolExecutor(max_workers=VLM_PARALLEL) as ex:
        answers = list(ex.map(lambda w: (w[0], _ask(w[1])), work))

    for i, ans in answers:
        if ans is None:
            failures += 1
            continue
        prev = tracklets[i].get("person_attrs") or {}
        attrs = {k: v for k, v in prev.items() if k not in _OWNED_KEYS}
        wrote = False
        for vk, sk in _ATTR_KEYS.items():
            name = str(ans.get(vk, "unknown")).strip().lower()
            if name and name != "unknown":
                attrs[sk] = {"name": name}
                wrote = True
        for ck in _COLOR_KEYS:                  # off-vocab (incl. "unknown") -> abstain
            name = str(ans.get(ck, "unknown")).strip().lower()
            if name in VLM_COLOR_VOCAB:
                attrs[ck] = {"name": name}
                colors += 1
                wrote = True
        tracklets[i]["person_attrs"] = attrs   # write even if all-unknown: stale keys stripped
        if wrote:
            labeled += 1

    (out / "tracklets.json").write_text(json.dumps(tracklets, indent=2))
    return {"cam": cam, "persons": len(persons), "skipped_short": skipped_short,
            "labeled": labeled, "colors": colors, "failures": failures,
            "secs": round(time.time() - t0, 1)}
