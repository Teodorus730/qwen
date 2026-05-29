# Topic domain next experiment decision

## Purpose

This document compares next experiment paths after the held-out v2-test result for `weak_topic_domain_v2.1`.

## Current evidence

`weak_topic_domain_v2.1` is the current transparent MVP baseline, but it should not be tuned further on v2-test.

Held-out v2-test showed a substantial generalization drop:

- coverage: 0.7889;
- accuracy on answered: 0.6197;
- strict accuracy: 0.4889.

FineWeb was the weakest subset:

- coverage: 0.7333;
- accuracy on answered: 0.3182;
- strict accuracy: 0.2333.

External review also identified a schema issue: the current `topic.domain` mixes semantic subject labels with genre/function labels such as `reference`, `education`, `media`, `commercial`, and `unknown`.

## Option A: embedding bake-off on current mixed `topic.domain`

### Pros

- Fastest path from current pseudo-gold to embedding comparison.
- Uses existing eval scripts and labels with minimal schema work.
- Provides a direct comparison against `weak_topic_domain_v2.1`.

### Cons

- The target is conceptually mixed.
- Embeddings may be rewarded for detecting genre/function rather than semantic topic.
- Labels like `commercial`, `education`, and `reference` do not represent the same axis as `math`, `software`, or `science`.
- Results would be hard to interpret and could bake schema confusion into the next classifier.

### Decision

Not recommended as the primary path.

## Option B: schema cleanup, then embedding bake-off on `semantic_topic_domain`

### Pros

- Creates a cleaner target for embedding label descriptions.
- Separates semantic topic from genre/function, quality/noise, surface features, and provenance.
- Reduces false conclusions in later NLL profiling.
- Avoids tuning the current mixed baseline on v2-test.
- Gives a more meaningful comparison between rule baseline, embedding classifier, and later supervised methods.

### Cons

- Requires new or remapped pseudo-gold.
- May require a new held-out split if the schema changes substantially.
- Delays the embedding experiment until label descriptions and dev labels exist.

### Decision

Recommended as the main path.

## Option C: start NLL pilot using robust axes only

### Pros

- Can proceed without changing classifiers or downloading models in this session.
- Uses already mature metadata:
  - provenance/dataset;
  - deterministic text stats;
  - tokenizer stats;
  - surface features;
  - quality/noise fields.
- Avoids overstating weak topic labels.
- Builds profiling plumbing while semantic labels mature.

### Cons

- Does not solve semantic topic classification.
- Topic-based NLL claims remain limited until cleaned labels exist.

### Decision

Recommended as a limited parallel path.

## Option D: train supervised classifier on current pseudo-gold

### Pros

- Could improve numerical performance on current labels.
- Creates a trainable baseline for later comparison.

### Cons

- Learns the current mixed taxonomy problem.
- Current pseudo-gold is small and weak.
- FineWeb held-out behavior suggests brittle boundaries and label ambiguity.
- A supervised classifier trained now may overfit schema artifacts and make later cleanup harder.

### Decision

Not recommended now.

## Recommendation

Choose **B + limited C**.

Primary path:

1. Cleanly separate axes in `annotation_v2`.
2. Create or remap v1-dev pseudo-gold for `semantic_topic_domain` and `genre_function`.
3. Run an embedding bake-off on `semantic_topic_domain`, not the current mixed `topic.domain`.
4. Keep v2-test held out; if schema changes substantially, create a fresh held-out v3 split or remap v2-test with explicit audit notes.

Parallel limited path:

1. Start NLL/profiling plumbing using robust axes only:
   - provenance;
   - deterministic text stats;
   - tokenizer stats;
   - surface flags;
   - quality/noise fields.
2. Treat current `weak_topic_domain_v2.1` as exploratory metadata only.

## Final decision

Do not run an embedding bake-off against the current mixed `topic.domain` as the main experiment.

Use `weak_topic_domain_v2.1` as a transparent legacy weak baseline, freeze it for comparison, and build the next semantic experiment around `semantic_topic_domain`.
