"""Eval: VLM structured-attribute search vs SigLIP semantic search, on SUR01 persons.

SUR01 person tracklets already carry a closed-vocab VLM attribute index (person_attrs:
apparent_gender/age/backpack/headwear/upper_color/lower_color). This harness runs a set of
person-description queries through BOTH:
  - ATTR: filter/rank by how many query-specified attributes match person_attrs (exact,
    closed-vocab). No embedding.
  - SIGLIP: encode the raw query text with the locked SigLIP model, cosine vs the stored
    semantic_vector (what the product ranks on today).
For each query it writes a side-by-side montage PNG (top row ATTR, bottom row SIGLIP) so a
human/independent model can score precision@K.

Run:  uv run python eval_attr_vs_siglip.py [--k 8] [--out <dir>]
"""

from __future__ import annotations

from gpu_setup import ensure_gpu_libs

ensure_gpu_libs()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

from pipeline.db import connect  # noqa: E402
from pipeline import paths  # noqa: E402

SIGLIP_MODEL = "google/siglip2-so400m-patch14-224"
SCENE = "SUR01"

# query text (for SigLIP) paired with the structured predicate (for ATTR). The predicate
# keys map to person_attrs keys; values must be in the closed vocab seen in the data.
QUERIES = [
    ("a man",                                   {"apparent_gender": "male"}),
    ("a woman",                                 {"apparent_gender": "female"}),
    ("a person wearing a helmet",               {"headwear": "helmet"}),
    ("a person wearing a turban",               {"headwear": "turban"}),
    ("a person carrying a backpack",            {"backpack": "yes"}),
    ("a man in a red shirt",                    {"apparent_gender": "male", "upper_color": "red"}),
    ("a woman in a black top",                  {"apparent_gender": "female", "upper_color": "black"}),
    ("a person in a white shirt and black pants", {"upper_color": "white", "lower_color": "black"}),
    ("a man wearing a cap with a backpack",     {"apparent_gender": "male", "headwear": "cap", "backpack": "yes"}),
    ("a person in a pink shirt",                {"upper_color": "pink"}),
]


def _attr_name(pa: dict, key: str):
    v = pa.get(key) if pa else None
    return v.get("name") if isinstance(v, dict) else None


def load_persons():
    conn = connect()
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tracklet_id, person_attrs, semantic_vector, crop_refs, num_detections "
            "FROM tracklets WHERE scene=%s AND entity_type='person' "
            "AND person_attrs IS NOT NULL AND semantic_vector IS NOT NULL",
            (SCENE,),
        )
        rows = cur.fetchall()
    conn.close()
    ids = [r[0] for r in rows]
    attrs = [r[1] for r in rows]
    def _tonp(v):
        if hasattr(v, "to_numpy"):
            return v.to_numpy()
        return np.asarray(list(v), dtype=np.float32)
    vecs = np.stack([_tonp(r[2]).astype(np.float32) for r in rows])
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    crops = [r[3][0] if r[3] else None for r in rows]
    ndet = np.array([r[4] or 0 for r in rows])
    return ids, attrs, vecs, crops, ndet


def attr_rank(attrs, ndet, pred, k):
    """Score = # predicate keys that match. Keep only rows matching ALL predicates
    (strict), rank ties by num_detections (longer track = more reliable attrs)."""
    scores = np.array([sum(_attr_name(pa, key) == val for key, val in pred.items())
                       for pa in attrs])
    full = len(pred)
    idx = np.where(scores == full)[0]
    idx = idx[np.argsort(-ndet[idx])]      # tie-break: reliability
    return idx[:k], scores, len(idx)


def siglip_rank(model, proc, vecs, query, k):
    inp = proc(text=[query], padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        f = model.get_text_features(**inp)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        f = torch.nn.functional.normalize(f, dim=-1)[0].float().numpy()
    sims = vecs @ f
    return np.argsort(-sims)[:k], sims


def crop_img(ref, size=(150, 200)):
    if not ref:
        return Image.new("RGB", size, (40, 40, 40))
    p = paths.OUTPUT_ROOT / ref
    try:
        im = Image.open(p).convert("RGB")
    except Exception:
        return Image.new("RGB", size, (40, 40, 40))
    im.thumbnail(size)
    canvas = Image.new("RGB", size, (20, 20, 20))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def label_line(pa):
    parts = []
    for key, short in (("apparent_gender", ""), ("upper_color", "top:"),
                       ("lower_color", "bot:"), ("headwear", ""), ("backpack", "bag:")):
        n = _attr_name(pa, key)
        if n and n not in ("none", "no"):
            parts.append(f"{short}{n}")
    return " ".join(parts)[:34]


def montage(query, pred, attr_idx, sig_idx, attrs, crops, out_png):
    cw, ch, pad, lab = 150, 200, 6, 22
    k = max(len(attr_idx), len(sig_idx))
    W = pad + k * (cw + pad)
    rowH = ch + lab
    H = 60 + 2 * (rowH + 24)
    img = Image.new("RGB", (W, H), (15, 15, 15))
    d = ImageDraw.Draw(img)
    d.text((pad, 8), f"QUERY: {query}    pred={pred}", fill=(255, 255, 255))

    def draw_row(y0, title, idxs):
        d.text((pad, y0), title, fill=(180, 220, 255))
        for j, i in enumerate(idxs):
            x = pad + j * (cw + pad)
            img.paste(crop_img(crops[i]), (x, y0 + 20))
            d.text((x, y0 + 20 + ch), label_line(attrs[i]), fill=(200, 200, 200))

    draw_row(44, f"ATTR  (top {len(attr_idx)})", attr_idx)
    draw_row(44 + rowH + 24, f"SIGLIP  (top {len(sig_idx)})", sig_idx)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="/tmp/claude-1000/-home-zer0-Desktop-Intelligent-Descriptive-Search-System-for-CCTV/63675c82-af33-4831-87ef-af90b9c6153f/scratchpad/attr_eval")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading persons ...")
    ids, attrs, vecs, crops, ndet = load_persons()
    print(f"  {len(ids)} SUR01 person tracklets with attrs+vectors")
    print("loading SigLIP ...")
    proc = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).eval()

    for n, (q, pred) in enumerate(QUERIES):
        a_idx, _, n_match = attr_rank(attrs, ndet, pred, args.k)
        s_idx, _ = siglip_rank(model, proc, vecs, q, args.k)
        png = out / f"q{n:02d}.png"
        montage(q, pred, a_idx, s_idx, attrs, crops, png)
        print(f"[q{n:02d}] '{q}'  attr_matches={n_match}  -> {png.name}")
    print(f"\nmontages in {out}")


if __name__ == "__main__":
    main()
