# Retrieval Improvement Plan (Lightweight, No Default VLM)

## Implementation status (2026-07-23)

The non-re-ID plan is implemented.

- Added `tracklet_crops` with crop-level 1152-d SigLIP vectors.
- Backfilled SUR01: 25,349 tracklets / 73,976 crop vectors (2.92 views per existing
  tracklet; old footage has three crops, while new detections use `K_CROPS=5`).
- Added `legacy_mean`, `mean`, `best_single`, `max`, and `top2_mean` scoring modes.
- Added ANN candidate union, exact crop reranking, deterministic query decomposition,
  entity-specific prompts, strict filters, and conservative fragment grouping.
- Added temporally diverse crop selection and configurable person-box expansion for new runs.
- Updated the live writer to persist the same crop-vector schema.
- Collected 466 pooled judgements across 15 queries and three retrieval methods in
  `apps/ingest/retrieval_eval/judgements.csv`.
- Measured on the SUR01 vehicle subset: HNSW candidate retrieval + exact reranking took
  0.386 s versus 3.359 s for a complete exact crop scan in a single-vector probe.

Human relevance labels are the remaining release gate. `top2_mean` is the configured candidate,
but it must not be presented as an accuracy improvement until the pooled CSV is labelled and
`evaluate_retrieval.py --score` confirms it.

## Goal

Improve descriptive retrieval accuracy without putting Qwen3-VL back into the default
ingest path. The default system remains:

```text
UVH-26 + CrowdHuman -> tracking -> diverse crops -> SigLIP2 -> pgvector -> clips
```

Qwen `vlm_attrs` remains an optional ablation only. Existing `person_attrs` must not be
required for search, scoring, or filtering.

This plan improves the system in this order:

1. establish a labelled retrieval benchmark;
2. preserve and search individual crop embeddings;
3. improve crop diversity;
4. add compositional SigLIP scoring;
5. use broad ANN retrieval followed by exact reranking;
6. stop silently relaxing explicit filters;
7. reduce duplicate tracklets;
8. specialize prompts and signals for people versus vehicles.

Reference-image identity search is explained separately near the end. It is not required
for the first retrieval milestone.

---

## Former baseline and problems addressed

- Previously, `pipeline/detect_track.py` retained the top `K_CROPS=5` crops using only
  `area * Laplacian sharpness`.
- Previously, `pipeline/embed_siglip.py` embedded every retained crop, then destroyed the individual
  vectors by mean-pooling them into one `semantic_vector` per tracklet.
- Text and uploaded-image searches compared against that one mean vector.
- The backend could silently remove type, colour, and time filters when no row matched.
- Search deduplication only caught overlapping fragments or shared `global_id` values.
- Person and vehicle queries used the same retrieval strategy.

The main expected failure is loss of view-specific evidence. A backpack may be clear in a
rear crop but absent in the front crops; averaging all views weakens the backpack signal.

## Non-goals

- Do not restore Qwen to default ingestion.
- Do not add a second large image model to descriptive text search.
- Do not hard-filter people by inferred gender, clothing, bag, or headwear.
- Do not fine-tune a model until the benchmark proves where the errors are.
- Do not replace forensic evidence rows when grouping duplicate results; grouping is a
  presentation/retrieval operation and the original tracklets remain immutable evidence.

---

## Phase 0 - Build the retrieval benchmark

Implementation should not begin with schema changes until the current baseline is recorded.

### Query set

Create 15-25 queries divided into fixed groups:

- vehicle type: `white SUV`, `red hatchback`, `yellow auto-rickshaw`;
- single person attribute: `person with a backpack`, `person wearing a helmet`;
- clothing: `person in a yellow shirt`, `person in a white shirt`;
- compositional: `person in a white shirt and black trousers`;
- difficult/rare: `person wearing a black cap`, `person wearing a turban`;
- reference-image look-alike queries.

Every query must specify its intended scene, camera/location, time range, and entity type.
That separates visual-ranking errors from filter errors.

### Labelling protocol

For each method, pool the top 20 results, deduplicate exact repeated tracklets, shuffle them,
and have two teammates independently mark `relevant`, `not relevant`, or `unclear`. Resolve
disagreements together. Never use Qwen-generated attribute chips as ground truth.

Store judgements in a small versioned CSV/JSON file containing at least:

```text
query_id, query_text, tracklet_id, relevance, reviewer
```

### Metrics

Report per query group and overall:

- Precision@5 and Precision@10;
- Recall@20 where the labelled pool supports it;
- nDCG@10;
- number of unique physical-looking results in the top 10;
- database retrieval latency and end-to-end query latency.

Record the current mean-vector SigLIP system as `baseline-v1`.

---

## Phase 1 - Persist individual crop embeddings

### Schema

Keep `tracklets.semantic_vector` temporarily for backward compatibility and add:

```sql
CREATE TABLE IF NOT EXISTS tracklet_crops (
  tracklet_id      TEXT NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
  crop_index       SMALLINT NOT NULL,
  frame_no         INT,
  crop_ref         TEXT NOT NULL,
  quality          REAL,
  semantic_vector  vector(1152) NOT NULL,
  PRIMARY KEY (tracklet_id, crop_index)
);

CREATE INDEX IF NOT EXISTS tracklet_crops_semantic_hnsw
  ON tracklet_crops USING hnsw (semantic_vector vector_cosine_ops);
```

`frame_no` and `quality` are nullable during the first migration because existing crop files
do not contain this metadata. New detection runs must populate both.

### Ingest artifact

Change SigLIP output from only:

```text
semantic.npy                 shape (tracklets, 1152)
```

to also write a crop-aligned artifact, for example:

```text
semantic_crops.npz
  vectors                    shape (number_of_crops, 1152)
  tracklet_indices           shape (number_of_crops,)
  crop_indices               shape (number_of_crops,)
```

The current embedding stage already computes each crop vector before averaging, so this adds
storage and DB writes but almost no model inference. Continue writing the mean vector until
all consumers have migrated.

### Store behavior

Within one transaction:

1. upsert the parent `tracklets` rows;
2. delete/reconcile stale crop rows for the affected tracklets;
3. upsert their current crop rows;
4. commit only after crop image files exist.

The operation must be idempotent: rerunning `store` cannot duplicate crop rows.

### Backfill

Existing crop JPEGs are sufficient. Re-run only `siglip` and `store`; detection and tracking
do not need to be repeated for the first multi-vector experiment.

---

## Phase 2 - Compare crop aggregation methods

Implement these scoring modes behind a configuration flag so the same query set can be
evaluated without re-ingesting:

1. `mean`: current normalized mean vector;
2. `best_single`: similarity of crop index 0 only;
3. `max`: maximum query-to-crop similarity;
4. `top2_mean`: average of the two highest crop similarities;
5. `mean3` and `mean5`: offline ablations for the current pooling choice.

For a tracklet with one crop, `top2_mean` falls back to that crop's score.

Expected default candidate: `top2_mean`. `max` has the best chance of recovering an attribute
visible in only one view, but it is also most vulnerable to one contaminated crop containing a
bystander. The benchmark, not intuition, chooses the winner.

Ship a new aggregation method only if it improves the difficult/compositional query group
without materially reducing the simple vehicle group.

---

## Phase 3 - Improve crop selection and diversity

Do not combine this experiment with Phase 2. First measure multi-vector search using today's
crops; then change crop selection as a separate variable.

### Persist crop metadata

Change `_TrackAcc` crop candidates to retain:

```text
frame_no, quality, width, height, crop
```

Write selected `frame_no` and `quality` into `tracklets.json` or a companion crop manifest so
the store stage can populate `tracklet_crops`.

### Candidate reservoir

Keeping only the globally sharpest five frames often produces nearly identical adjacent
frames. Maintain a small candidate reservoir per active tracklet, such as 15 candidates, and
select the final five after the track ends.

Final selection should provide:

- one good crop from the early portion of the track;
- one from the middle;
- one from the late portion;
- two best remaining crops that are not near-duplicates.

Reject a candidate when it is too close in time to a selected crop and visually near-identical.
Start with a cheap frame-gap rule; add embedding similarity only if the benchmark shows a need.
Do not run SigLIP inside the detector loop.

### Person-box expansion ablation

Test the existing tight person crop against approximately 5% and 10% expansion, clamped to the
frame. Expansion should be slightly larger horizontally and above the head than below the feet.

Manually score whether expansion:

- recovers caps, helmets, scarves, bags, and lower clothing;
- introduces other people or vehicle/background contamination.

Choose separate crop-padding settings for people and vehicles. Zero padding remains valid if
expanded crops reduce retrieval quality.

---

## Phase 4 - Compositional SigLIP retrieval

### Deterministic component extraction

After Gujarati/Hindi translation, parse a small controlled English vocabulary:

- colours;
- vehicle types from the UVH-26 class map;
- person nouns (`man`, `woman`, `person`, `rider`);
- clothing nouns (`shirt`, `t-shirt`, `trousers`, `pants`, `kurta`, `dress`);
- headwear (`cap`, `helmet`, `turban`, `scarf`);
- carried items (`backpack`, `bag`).

The parser must preserve the complete rewritten caption and produce at most three useful
component captions. Example:

```text
full:       a CCTV image of a person wearing a yellow shirt and a black cap
component1: a CCTV image of a person wearing a yellow shirt
component2: a CCTV image of a person wearing a black cap
```

Do not treat a negated phrase such as `without a helmet` as a positive helmet component.
Initially leave negation only in the full caption rather than implementing an unsafe hard
exclusion.

### Candidate generation

Run ANN retrieval independently for the full caption and every component, then take the union
of their candidate tracklet IDs. This prevents the full-query embedding from hiding a candidate
that strongly matches one important component.

### Score normalization

Raw similarities from different text prompts are not directly calibrated. Normalize each
prompt's exact candidate scores using a robust candidate-set method such as percentile rank or
z-score with clipping.

Evaluate at least:

```text
weighted:
  0.50 * full + 0.25 * component1 + 0.25 * component2

soft-AND:
  0.50 * full + 0.50 * harmonic_mean(components)
```

The soft-AND version rewards results that satisfy every component, but missing components must
not become irreversible hard filters.

---

## Phase 5 - Broad ANN retrieval, exact reranking

For each full/component text vector:

```text
HNSW crop search
    -> union a broad candidate set
    -> fetch all stored crop vectors for candidate tracklets
    -> exact cosine scores in memory
    -> per-tracklet crop aggregation
    -> compositional score fusion
    -> duplicate grouping
    -> final top N
```

Start by retrieving roughly 500-1,000 crop rows per prompt, because five crops from one
tracklet can otherwise consume the candidate list. Tune this using measured recall and latency.

Also benchmark an exact filtered scan. At approximately 125,000 crop vectors it may be usable
for the prototype, particularly after camera/time/entity filters. If exact scan latency is
acceptable, prefer its guaranteed recall and simpler behavior for the hackathon demo.

Track database time separately from SigLIP text-encoding time. Target no measurable ingest
inference increase, database retrieval under a few hundred milliseconds on SUR01, and an
end-to-end interactive response.

---

## Phase 6 - Never silently weaken explicit filters

Remove the current automatic fallback that discards entity type, colour, or time constraints
when no results are returned.

An explicit officer-selected constraint is authoritative. Return:

```text
0 results with the selected filters.
Suggestions:
- expand the time window;
- remove the colour constraint;
- search both entity types.
```

The UI may let the officer click one of those actions and issue a new search. The API must say
which filters were applied and must never return out-of-window evidence without an explicit
request.

Person clothing colour is semantic evidence, not the vehicle-level `color` column. Never apply
the vehicle colour column as a hard person filter.

---

## Phase 7 - Reduce duplicate fragments

First improve result grouping without mutating evidence:

- always collapse shared `global_id` rows for the card grid while retaining a sightings list;
- group same-camera fragments that are non-overlapping, temporally close, same subtype, and
  visually similar;
- use exact/validated plate agreement as a strong vehicle signal;
- never merge simultaneous same-camera objects merely because they look alike.

Return a representative best crop plus `N sightings/fragments`. The detail view can expose every
original tracklet and timestamp.

Only after the grouping benchmark is satisfactory should the offline matcher assign persistent
same-camera group IDs. Preserve original tracklet IDs for audit and export.

Measure `unique physical-looking results@10` alongside Precision@10.

---

## Phase 8 - Separate person and vehicle retrieval

### Vehicle path

Use:

```text
entity type + UVH subtype + vehicle colour + plate + multi-crop SigLIP + time/location
```

Subtype, validated plate, camera, and time may be hard filters when explicitly selected.
Colour should be relaxable because lighting changes paint appearance.

Prompt family to benchmark:

```text
a CCTV image of a red hatchback vehicle
a photo of a red hatchback
```

### Person path

Use:

```text
entity type + multi-crop compositional SigLIP + time/location
```

Do not require `person_attrs`. Clothing, headwear, and bags are semantic ranking signals.

Prompt family to benchmark:

```text
a CCTV image of a person wearing a yellow shirt and black cap
a photo of a person in a yellow shirt and black cap
```

Choose templates by query-group metrics rather than selecting one visually pleasing example.

---

## Reference-image search: what the two modes mean

Reference-image search answers two different police questions.

### Mode A - Find visually similar objects

Example: an officer uploads a white hatchback crop and wants other white hatchbacks.

SigLIP is suitable because it represents broad semantic appearance: object type, colour,
clothing, and visible accessories. It may correctly return several different white hatchbacks,
because they all look semantically similar.

```text
uploaded white hatchback -> SigLIP -> visually similar white hatchbacks
```

This is what `POST /search/image` currently implements.

### Mode B - Track this exact physical target

Example: an officer uploads one getaway-car crop and wants the same car at another camera,
not every vehicle of the same colour and type.

This requires a re-identification encoder. It is trained or evaluated to preserve instance
details across viewpoint changes: body shape, stickers, damage, trim, cargo, and other fine
appearance cues.

```text
uploaded getaway car -> same re-ID encoder used at ingest -> same physical car candidates
```

DINOv3-L is a candidate vehicle re-ID encoder because the existing experiments show strong
vehicle results. That evidence does not prove person re-ID accuracy; people need their own
labelled same-person/different-person evaluation before the UI makes an identity claim.

The current code does **not** provide uploaded-image exact-target search. `/search/image` always
uses SigLIP. `/search/similar?vec=reid` can use an existing stored tracklet's re-ID vector, but it
cannot encode a new uploaded image with the re-ID model, and it silently falls back to SigLIP
when the stored re-ID vector is missing. That fallback must not be presented as identity search.

To implement exact-target upload later:

1. accept a tightly cropped target image (or detect and let the officer choose the box);
2. encode it with exactly the same re-ID model and preprocessing used during ingest;
3. compare it with individual stored re-ID crop vectors, not only a tracklet mean;
4. apply camera topology and plausible travel-time constraints;
5. label results as candidates for officer verification, never confirmed identity;
6. if multiple reference views are available, use the best query-to-gallery pairing or a
   carefully evaluated multi-view aggregate.

A full CCTV frame is a poor reference because background, road, and nearby people can dominate
the embedding. “Tight crop” means the target occupies most of the uploaded image while retaining
useful details such as a backpack or vehicle outline.

Recommended UI labels:

- `Find similar appearance` -> SigLIP;
- `Find same target (experimental)` -> re-ID, shown only when real re-ID vectors exist.

Never silently change the second mode into the first.

---

## Implementation order and file map

### Milestone A - measurable multi-crop retrieval

1. Add benchmark judgements and evaluator.
2. Add `tracklet_crops` to `infra/schema.sql`.
3. Preserve per-crop vectors in `pipeline/embed_siglip.py`.
4. Store them transactionally in `pipeline/store.py`.
5. Add aggregation modes and exact reranking in `backend_py/app/engine.py`.
6. Backfill SUR01 using existing crop JPEGs.
7. Run the benchmark and choose `max` versus `top2_mean`.

### Milestone B - better crops

1. Persist frame/quality metadata in `pipeline/detect_track.py`.
2. Add temporal/diversity selection.
3. A/B person crop padding.
4. Re-ingest only the benchmark cameras first.

### Milestone C - compositional search and safety

1. Add deterministic query decomposition next to `query_rewrite.py`.
2. Add per-component candidate union and normalized score fusion.
3. Remove automatic filter relaxation.
4. Return actionable zero-result suggestions.
5. Add entity-specific prompt templates.

### Milestone D - result grouping

1. Add conservative same-camera fragment grouping.
2. Return sightings without deleting original evidence rows.
3. Evaluate unique-result quality and false merges.

---

## Release gates

Do not make a new method the default unless:

- the labelled benchmark shows a repeatable improvement, especially on compositional queries;
- simple vehicle-query accuracy does not materially regress;
- no explicit filter is silently discarded;
- query latency remains interactive;
- ingest GPU memory stays within the lightweight target;
- rerunning ingest/store is idempotent;
- every displayed result remains traceable to its original tracklet, crop, camera, and timestamp.

The first likely production candidate is:

```text
temporally diverse five crops
+ individual SigLIP vectors
+ top-two crop aggregation
+ full-caption/component candidate union
+ exact reranking
+ explicit time/location/entity filters
+ conservative duplicate grouping
```

The benchmark must confirm each `+` independently.
