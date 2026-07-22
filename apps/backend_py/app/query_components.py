"""Deterministic decomposition of short CCTV descriptions for SigLIP retrieval.

This deliberately handles a small visible vocabulary. It never invents attributes and
never turns negated attributes into positive components. The complete caption is always
retained, so an unrecognised phrase still participates in semantic search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COLORS = (
    "white", "black", "gray", "grey", "silver", "red", "blue", "green",
    "yellow", "orange", "brown", "pink", "purple",
)
CLOTHING = (
    "t-shirt", "tshirt", "shirt", "top", "trousers", "pants", "jeans",
    "kurta", "dress", "saree", "sari", "jacket", "hoodie",
)
HEADWEAR = ("cap", "helmet", "turban", "scarf", "hat")
CARRIED = ("backpack", "rucksack", "bag")
VEHICLES = (
    "auto-rickshaw", "auto rickshaw", "hatchback", "sedan", "suv", "motorcycle",
    "motorbike", "scooter", "bicycle", "pickup truck", "pickup", "truck", "bus",
    "minibus", "van", "car", "vehicle",
)

_COLOR = "|".join(map(re.escape, COLORS))
_CLOTHING = "|".join(sorted(map(re.escape, CLOTHING), key=len, reverse=True))
_HEADWEAR = "|".join(map(re.escape, HEADWEAR))
_VEHICLE = "|".join(sorted(map(re.escape, VEHICLES), key=len, reverse=True))
_NEGATABLE = "|".join(sorted(map(re.escape, CLOTHING + HEADWEAR + CARRIED),
                              key=len, reverse=True))
_NEGATED = re.compile(
    rf"\b(?:without|no|not wearing)\s+(?:an?\s+)?(?:({_COLOR})\s+)?({_NEGATABLE})\b",
    re.I,
)
_PREFIX = re.compile(r"^(?:a\s+)?(?:photo|picture|cctv image)\s+of\s+", re.I)


@dataclass(frozen=True)
class QueryPlan:
    full_caption: str
    components: tuple[str, ...]
    entity_hint: str | None


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def infer_entity(text: str) -> str | None:
    low = text.lower()
    if _contains(low, VEHICLES):
        return "vehicle"
    if (_contains(low, ("person", "man", "woman", "boy", "girl", "rider"))
            or _contains(low, CLOTHING + HEADWEAR + CARRIED)):
        return "person"
    return None


def _subject(text: str) -> str:
    if re.search(r"\bwoman\b", text):
        return "a woman"
    if re.search(r"\bman\b", text):
        return "a man"
    if re.search(r"\brider\b", text):
        return "a rider"
    return "a person"


def _visible_description(caption: str) -> str:
    return _PREFIX.sub("", caption.strip().rstrip("."), count=1).strip()


def _entity_caption(description: str, entity: str | None) -> str:
    # Rewritten queries commonly arrive without an article (for example,
    # "red hatchback").  SigLIP prompt templates are more natural when the
    # visible subject remains a noun phrase rather than a bag of tags.
    if entity in {"vehicle", "person"} and not re.match(r"^(?:a|an|the)\b", description, re.I):
        description = f"a {description}"
    if entity == "vehicle":
        return f"a CCTV image of {description}"
    if entity == "person":
        return f"a CCTV image of {description}"
    return f"a photo of {description}"


def build_query_plan(caption: str, entity_type: str | None = None) -> QueryPlan:
    description = _visible_description(caption)
    low = description.lower()
    entity = entity_type or infer_entity(low)
    full = _entity_caption(description, entity)
    negated = {m.group(2).lower() for m in _NEGATED.finditer(low)}
    components: list[str] = []

    def add(value: str) -> None:
        if value != full and value not in components and len(components) < 3:
            components.append(value)

    if entity == "person":
        subject = _subject(low)
        for m in re.finditer(rf"\b({_COLOR})\s+({_CLOTHING})\b", low):
            phrase = m.group(0).replace("grey", "gray")
            if m.group(2) not in negated:
                add(f"a CCTV image of {subject} wearing a {phrase}")
        for m in re.finditer(rf"\b(?:{_COLOR})\s+({_HEADWEAR})\b|\b({_HEADWEAR})\b", low):
            item = (m.group(1) or m.group(2) or "").lower()
            phrase = m.group(0)
            if item and item not in negated:
                add(f"a CCTV image of {subject} wearing a {phrase}")
        for item in CARRIED:
            if re.search(rf"\b{re.escape(item)}\b", low) and item not in negated:
                add(f"a CCTV image of {subject} carrying a {item}")
    elif entity == "vehicle":
        for m in re.finditer(rf"\b({_COLOR})\s+({_VEHICLE})\b", low):
            add(f"a CCTV image of a {m.group(0).replace('grey', 'gray')} vehicle")

    return QueryPlan(full_caption=full, components=tuple(components), entity_hint=entity)
