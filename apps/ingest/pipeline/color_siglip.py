"""Stage 6c (color name): SigLIP2 zero-shot color naming, GPU/fp16 (CPU fallback).

Replaces the brittle HSV `dominant_color_name` for the display-only `color` column.
Instead of hand-tuned HSV thresholds (which the road, windows, and shadows in
CityFlow crops routinely fool into "gray/silver"), this classifies color in the
SAME SigLIP2 space search already uses:

  color = argmax_c  cos( tracklet_semantic_vector , text("a photo of a {c} car") )

It REUSES the mean-pooled tracklet vector (vec/semantic.npy) written by the siglip
pass — no image re-encode — and only runs the (tiny) text tower over a fixed color
vocabulary. Writes the winning name into each tracklet's `color` field in
tracklets.json, which store.py already reads via t.get("color").

Run AFTER siglip and BEFORE store. The 56-dim HSV signature (attributes.py) is left
untouched; the matcher still fuses it.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from . import paths
from .cfg import (
    COLOR_MIN_MARGIN,
    COLOR_PROMPT_NOUNS,
    COLOR_PROMPT_TEMPLATES,
    COLOR_VOCAB,
    SEMANTIC_DIM,
    SIGLIP_MODEL,
)


def _color_text_matrix(model, processor, dev) -> np.ndarray:
    """(C, SEMANTIC_DIM) L2-normed text embedding per color, averaged over the prompt ensemble."""
    prompts: list[str] = []
    owner: list[int] = []
    for ci, color in enumerate(COLOR_VOCAB):
        for noun in COLOR_PROMPT_NOUNS:
            for tmpl in COLOR_PROMPT_TEMPLATES:
                prompts.append(tmpl.format(color=color, noun=noun))
                owner.append(ci)

    # SigLIP requires fixed-length padding to 64 tokens.
    inputs = processor(
        text=prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    ).to(dev)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):  # transformers 5.x output object
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy()

    owner_arr = np.asarray(owner)
    text = np.zeros((len(COLOR_VOCAB), feats.shape[1]), dtype=np.float32)
    for ci in range(len(COLOR_VOCAB)):
        m = feats[owner_arr == ci].mean(axis=0)
        text[ci] = m / (np.linalg.norm(m) + 1e-12)
    return text


def run(scene: str, cam: str, device: int = 0) -> dict:
    out = paths.cam_out(scene, cam)
    tracklets = json.loads((out / "tracklets.json").read_text())
    n = len(tracklets)

    sem_path = out / "vec" / "semantic.npy"
    if not sem_path.exists():
        return {"cam": cam, "skipped": "vec/semantic.npy missing (run siglip first)", "tracklets": n}
    sem = np.load(sem_path)
    if sem.shape != (n, SEMANTIC_DIM):
        raise ValueError(f"{sem_path} has shape {sem.shape}, expected {(n, SEMANTIC_DIM)}")

    use_cuda = torch.cuda.is_available()
    dev = f"cuda:{device}" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=dtype).to(dev).eval()

    text = _color_text_matrix(model, processor, dev)  # (C, D)

    sims = sem @ text.T                                # (N, C) cosine (both L2-normed)
    order = np.argsort(-sims, axis=1)
    rows = np.arange(n)
    top1 = sims[rows, order[:, 0]]
    top2 = sims[rows, order[:, 1]] if len(COLOR_VOCAB) > 1 else np.full(n, -np.inf)
    margin = top1 - top2

    named = 0
    for i, t in enumerate(tracklets):
        if np.linalg.norm(sem[i]) < 1e-6:              # empty/no-crop tracklet
            t["color"] = None
            continue
        if margin[i] < COLOR_MIN_MARGIN:               # too ambiguous to commit a name
            t["color"] = None
            continue
        t["color"] = COLOR_VOCAB[int(order[i, 0])]
        named += 1

    (out / "tracklets.json").write_text(json.dumps(tracklets, indent=2))

    peak = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0.0
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return {"cam": cam, "tracklets": n, "colored": named, "vram_peak_mb": round(peak, 1)}
