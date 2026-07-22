"""Collect pooled search results for human labelling and score retrieval methods.

The backend must be running for --collect. Human judgements are deliberately external
to every model being evaluated: set relevance to 1, 0, or unclear in the generated CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "retrieval_eval"
METHODS = {
    "baseline_mean": {"aggregation": "legacy_mean", "compositional": "false"},
    "max_composite": {"aggregation": "max", "compositional": "true"},
    "top2_composite": {"aggregation": "top2_mean", "compositional": "true"},
}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def collect(api: str, queries_path: Path, output: Path, limit: int) -> None:
    queries = json.loads(queries_path.read_text())
    old = {}
    if output.exists():
        with output.open(newline="") as handle:
            for row in csv.DictReader(handle):
                old[(row["query_id"], row["tracklet_id"])] = (
                    row.get("relevance", ""), row.get("reviewer", ""), row.get("notes", ""))

    pooled: dict[tuple[str, str], dict] = {}
    for query in queries:
        for method, options in METHODS.items():
            params = {
                "q": query["query"], "scene": query.get("scene", "SUR01"),
                "type": query.get("type", ""), "camera_id": query.get("camera_id", ""),
                "limit": str(limit), **options,
            }
            for key in ("t0", "t1"):
                if query.get(key) is not None:
                    params[key] = str(query[key])
            url = f"{api.rstrip('/')}/search?{urllib.parse.urlencode(params)}"
            result = _get(url)
            print(f"{query['id']:20} {method:16} {len(result.get('results', [])):3} results")
            for rank, item in enumerate(result.get("results", []), 1):
                key = (query["id"], item["tracklet_id"])
                crop_url = item.get("crop_url") or ""
                row = pooled.setdefault(key, {
                    "query_id": query["id"], "query_text": query["query"],
                    "group": query["group"], "tracklet_id": item["tracklet_id"],
                    "camera_id": item["camera_id"], "subtype": item["subtype"],
                    "crop_url": crop_url,
                    "image_url": f"{api.rstrip('/')}{crop_url}" if crop_url else "",
                    "relevance": "", "reviewer": "", "notes": "",
                    **{f"rank_{name}": "" for name in METHODS},
                })
                row[f"rank_{method}"] = rank

    for key, labels in old.items():
        if key in pooled:
            pooled[key]["relevance"], pooled[key]["reviewer"], pooled[key]["notes"] = labels
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id", "query_text", "group", "tracklet_id", "camera_id", "subtype",
        "crop_url", "image_url", *[f"rank_{name}" for name in METHODS],
        "relevance", "reviewer", "notes",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(pooled.values(), key=lambda r: (r["query_id"], r["tracklet_id"])))
    print(f"wrote {len(pooled)} pooled rows to {output}")


def _dcg(labels: list[int]) -> float:
    return sum(value / math.log2(rank + 2) for rank, value in enumerate(labels))


def score(labels_path: Path) -> None:
    with labels_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_query = defaultdict(list)
    for row in rows:
        value = row.get("relevance", "").strip().lower()
        if value not in {"0", "1"}:
            continue
        row["rel"] = int(value)
        by_query[row["query_id"]].append(row)
    if not by_query:
        raise SystemExit("no 0/1 human relevance labels found")

    print("method             queries   P@5   P@10  recall@20  nDCG@10")
    for method in METHODS:
        metrics = []
        rank_key = f"rank_{method}"
        for rows_q in by_query.values():
            ranked = sorted((r for r in rows_q if r[rank_key]), key=lambda r: int(r[rank_key]))
            if not ranked:
                continue
            rel = [r["rel"] for r in ranked]
            positives = sum(r["rel"] for r in rows_q)
            p5 = sum(rel[:5]) / 5
            p10 = sum(rel[:10]) / 10
            recall20 = sum(rel[:20]) / positives if positives else 0.0
            ideal = sorted((r["rel"] for r in rows_q), reverse=True)[:10]
            ndcg = _dcg(rel[:10]) / _dcg(ideal) if _dcg(ideal) else 0.0
            metrics.append((p5, p10, recall20, ndcg))
        means = [sum(m[i] for m in metrics) / len(metrics) for i in range(4)]
        print(f"{method:18} {len(metrics):7} " + " ".join(f"{v:7.3f}" for v in means))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--queries", type=Path, default=ROOT / "queries.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "judgements.csv")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.collect:
        collect(args.api, args.queries, args.labels, args.limit)
    if args.score:
        score(args.labels)
    if not args.collect and not args.score:
        parser.error("choose --collect and/or --score")


if __name__ == "__main__":
    main()
