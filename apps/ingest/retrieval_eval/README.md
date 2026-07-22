# Retrieval judgement guide

This directory contains the fixed query set and the pooled candidates used to compare
legacy mean-vector search with crop-level `max` and `top2_mean` compositional search.

## Label the pool

1. Start the backend on `http://127.0.0.1:8010` so each `image_url` is viewable.
2. Give a copy of `judgements.csv` to each reviewer.
3. For every row, inspect the image and enter `1` for relevant or `0` for not relevant.
   Leave genuinely ambiguous images blank and explain them in `notes`.
4. Enter the reviewer's name in `reviewer`. Reviewers must judge the query text, not the
   detector subtype or any model-generated metadata.
5. Resolve disagreements together and retain one final labelled CSV.

The same tracklet appears only once per query even when several methods retrieved it. Its
three rank columns show which result lists contributed it, keeping the review blind to a
single preferred method.

## Score the methods

From `apps/ingest`, run:

```bash
uv run python evaluate_retrieval.py --score
```

The scorer refuses to report accuracy while labels are missing. Choose the production
aggregation only after comparing Precision@5, Precision@10, pooled Recall@20, and nDCG@10.
