"""Stage (person attributes): region-split SigLIP outfit color, GPU/fp16 (CPU fallback).

For each PERSON tracklet, splits its K crops into an upper-body (torso/shirt) and a
lower-body (legs/trousers) region, encodes each region with the SAME SigLIP2 image
encoder search uses, mean-pools per region across the crops, and names the color of
each by zero-shot cosine against a clothing-noun prompt ensemble (exactly as
color_siglip does for vehicles, but region-localized). Writes into each person
tracklet's `person_attrs` JSONB:

  {"upper_color": {"name": "red",  "conf": 0.72},
   "lower_color": {"name": "blue", "conf": 0.61}}

Two independent, region-localized colors discriminate people far better than one
blended dominant color (see plan.md "Person attributes"). SOFT ranking signal only,
never a hard filter; a region whose top-2 colors are within PERSON_COLOR_MIN_MARGIN
abstains (no key written) rather than committing a mislabel. Vehicles are skipped.
`person_attrs` is additive/nullable — the VLM stage (vlm_attrs: apparent gender, age,
backpack, headwear) fills the same dict.

Run AFTER detect (needs crops on disk); placed near the siglip/color passes so the
SigLIP model loads are grouped. Call gpu_setup.ensure_gpu_libs() BEFORE importing.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from . import paths
from .cfg import (
    COLOR_PROMPT_TEMPLATES,
    PERSON_COLOR_MIN_MARGIN,
    PERSON_COLOR_SOFTMAX_T,
    PERSON_COLOR_VOCAB,
    PERSON_LOWER_NOUNS,
    PERSON_REGION_X,
    PERSON_REGIONS,
    PERSON_UPPER_NOUNS,
    SIGLIP_MODEL,
)

_BATCH = 64
_REGION_NOUNS = {"upper_color": PERSON_UPPER_NOUNS, "lower_color": PERSON_LOWER_NOUNS}


def _load_bgr(rel_key: str) -> np.ndarray | None:
    return cv2.imread(str(paths.OUTPUT_ROOT / rel_key))


def _region_rgb(crop_bgr: np.ndarray, y_frac: tuple[float, float]) -> np.ndarray:
    """Crop the given vertical band (sides trimmed) and return it as RGB."""
    h, w = crop_bgr.shape[:2]
    y0, y1 = int(y_frac[0] * h), int(y_frac[1] * h)
    x0, x1 = int(PERSON_REGION_X[0] * w), int(PERSON_REGION_X[1] * w)
    reg = crop_bgr[y0:y1, x0:x1]
    if reg.size == 0:
        reg = crop_bgr
    return cv2.cvtColor(reg, cv2.COLOR_BGR2RGB)


def _text_matrix(model, processor, dev, nouns) -> np.ndarray:
    """(C, D) L2-normed text embedding per color, averaged over the noun x template ensemble."""
    prompts: list[str] = []
    owner: list[int] = []
    for ci, color in enumerate(PERSON_COLOR_VOCAB):
        for noun in nouns:
            for tmpl in COLOR_PROMPT_TEMPLATES:
                prompts.append(tmpl.format(color=color, noun=noun))
                owner.append(ci)
    inputs = processor(
        text=prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    ).to(dev)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):          # transformers 5.x output object
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy()
    owner_arr = np.asarray(owner)
    text = np.zeros((len(PERSON_COLOR_VOCAB), feats.shape[1]), dtype=np.float32)
    for ci in range(len(PERSON_COLOR_VOCAB)):
        m = feats[owner_arr == ci].mean(axis=0)
        text[ci] = m / (np.linalg.norm(m) + 1e-12)
    return text


def run(scene: str, cam: str, device: int = 0) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    region_keys = list(PERSON_REGIONS)                   # ["upper_color", "lower_color"]
    R = len(region_keys)

    # Flatten (tracklet_index, region_index, image) over PERSON tracklets only, so we
    # batch every region crop across the whole camera through the encoder once.
    owner_idx: list[int] = []
    owner_reg: list[int] = []
    imgs: list[np.ndarray] = []
    persons = 0
    for i, t in enumerate(tracklets):
        if t.get("entity_type") != "person":
            continue
        persons += 1
        for k in t["crop_refs"]:
            crop = _load_bgr(k)
            if crop is None:
                continue
            for r, rk in enumerate(region_keys):
                imgs.append(_region_rgb(crop, PERSON_REGIONS[rk]))
                owner_idx.append(i)
                owner_reg.append(r)

    if not imgs:
        return {"cam": cam, "persons": persons, "region_labels": 0}

    use_cuda = torch.cuda.is_available()
    dev = f"cuda:{device}" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=dtype).to(dev).eval()

    sums: np.ndarray | None = None
    counts = np.zeros((n, R), dtype=np.int32)
    with torch.no_grad():
        for b in range(0, len(imgs), _BATCH):
            batch = imgs[b:b + _BATCH]
            pv = processor(images=batch, return_tensors="pt").pixel_values.to(dev, dtype)
            feats = model.get_image_features(pixel_values=pv)
            if not isinstance(feats, torch.Tensor):      # transformers 5.x output object
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy()
            if sums is None:
                sums = np.zeros((n, R, feats.shape[1]), dtype=np.float32)
            for j in range(len(batch)):
                ti, ri = owner_idx[b + j], owner_reg[b + j]
                sums[ti, ri] += feats[j]
                counts[ti, ri] += 1

    texts = [_text_matrix(model, processor, dev, _REGION_NOUNS[rk]) for rk in region_keys]

    labeled = 0
    for i, t in enumerate(tracklets):
        if t.get("entity_type") != "person":
            continue
        attrs = dict(t.get("person_attrs") or {})
        for r, rk in enumerate(region_keys):
            if counts[i, r] == 0:
                continue
            v = sums[i, r] / counts[i, r]
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                continue
            v = v / norm
            sims = texts[r] @ v                           # (C,) cosine (both L2-normed)
            order = np.argsort(-sims)
            if float(sims[order[0]] - sims[order[1]]) < PERSON_COLOR_MIN_MARGIN:
                continue                                  # too ambiguous — abstain
            e = np.exp((sims - sims.max()) * PERSON_COLOR_SOFTMAX_T)
            conf = float(e[order[0]] / e.sum())
            attrs[rk] = {"name": PERSON_COLOR_VOCAB[int(order[0])], "conf": round(conf, 3)}
            labeled += 1
        if attrs:
            t["person_attrs"] = attrs

    (out / "tracklets.json").write_text(json.dumps(tracklets, indent=2))

    peak = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0.0
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return {"cam": cam, "persons": persons, "region_labels": labeled,
            "vram_peak_mb": round(peak, 1)}
