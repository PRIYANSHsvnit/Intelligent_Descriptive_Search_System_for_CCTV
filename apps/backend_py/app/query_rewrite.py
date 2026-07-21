"""Query preprocessing for text search: translate (Gujarati/Hindi/…→English) and
rewrite the raw fragment into a caption-style phrase before SigLIP encoding.

SigLIP was trained on caption-style text, so "a photo of a man in a red shirt"
retrieves measurably better than the raw fragment "man red shirt". A single Groq
`llama-3.1-8b-instant` call does both jobs (translate + caption) in one round-trip.

FAIL-OPEN by design: search must never depend on the network. If GROQ_API_KEY is
unset, or the call errors/times out, we fall back to a fixed template wrapper
`"a photo of {q}"` — which is itself the cheap SigLIP win, so we get it either way.
"""

from __future__ import annotations

import os

import httpx

GROQ_MODEL = os.environ.get("GROQ_QUERY_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_S = float(os.environ.get("GROQ_QUERY_TIMEOUT_S", "3.0"))


def _api_key() -> str:
    # read lazily so a .env loaded at app startup (config.py) is always picked up,
    # regardless of import order.
    return os.environ.get("GROQ_API_KEY", "").strip()

# The rewrite must ADD NOTHING: an 8B model told to "improve" a query happily invents
# attributes ("man red shirt" -> "...standing on a busy street at night"), and every
# invented word pollutes the SigLIP vector and drags in wrong results.
_SYSTEM_PROMPT = (
    "You rewrite CCTV search queries into a short English caption for an "
    "image-retrieval model.\n"
    "Rules:\n"
    "- If the query is not in English, translate it to English.\n"
    "- Rewrite as a concise caption of only what would be visible in a photo, "
    'e.g. "a photo of a man in a red shirt" or "a photo of a red motorcycle".\n'
    "- Do NOT add any attribute, object, color, count, location, time, or action "
    "that is not present in the original query.\n"
    "- Preserve negations verbatim (without / no / not) — e.g. "
    '"motorcycle without helmet" stays "a photo of a motorcycle rider without a '
    'helmet". Never drop a negated attribute.\n'
    "- Output ONLY the caption. No quotes, no explanation, no trailing punctuation."
)

# Only successful Groq rewrites are cached, so a transient failure retries next time
# instead of pinning the fallback string forever.
_cache: dict[str, str] = {}


def _template(q: str) -> str:
    """Static caption wrapper — the fallback and the zero-latency baseline."""
    return f"a photo of {q.strip()}"


def _groq_rewrite(q: str) -> str | None:
    """One Groq chat call. Returns the caption, or None on any failure (caller
    falls back to the template). temperature=0 keeps it deterministic/cacheable."""
    try:
        resp = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {_api_key()}"},
            json={
                "model": GROQ_MODEL,
                "temperature": 0,
                "max_tokens": 64,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
            },
            timeout=GROQ_TIMEOUT_S,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # strip stray wrapping quotes the model sometimes adds despite instructions
        text = text.strip('"').strip("'").strip()
        return text or None
    except Exception:
        return None


def rewrite(q: str) -> str:
    """Text query → SigLIP-ready caption. Groq translate+rewrite when configured,
    static template fallback otherwise. Never raises."""
    q = q.strip()
    if not q:
        return q
    if not _api_key():
        return _template(q)
    if q in _cache:
        return _cache[q]
    out = _groq_rewrite(q)
    if out is None:
        return _template(q)  # not cached: retry Groq on the next identical query
    _cache[q] = out
    return out
