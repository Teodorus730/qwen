# Data samples

This folder stores small stage2 artifacts for local checks and team review.

Canonical policy: see `../docs/data_policy_ru.md`.

## What belongs here

- small synthetic/local outputs used by smoke tests;
- canonical classifier benchmark outputs;
- tiny real samples after review;
- small run stats that explain how outputs were produced.

## What does not belong here

- full datasets;
- HF caches;
- model caches;
- dependency folders;
- temporary large sweeps;
- accidental full-dataset outputs;
- MiniLM embedding outputs unless explicitly promoted;
- NLL/logprob outputs for the current stage2 scope.

## Current useful files

Synthetic/local:

- `edge_case_chunks_sample.jsonl`
- `edge_case_chunks_labeled.jsonl`
- `classifier_benchmark_chunks.jsonl`
- `classifier_benchmark_labeled.jsonl`
- `classifier_benchmark_lexical_labeled.jsonl`

Tiny real samples:

- `real_samples/fineweb_edu_sample10bt_chunks.jsonl`
- `real_samples/fineweb_edu_sample10bt_labeled_rule_based.jsonl`
- `real_samples/fineweb_edu_sample10bt_labeled_lexical.jsonl`
- `real_samples/finemath_chunks.jsonl`
- `real_samples/finemath_labeled_rule_based.jsonl`
- `real_samples/finemath_labeled_lexical.jsonl`

## Old sweep outputs

Some untracked benchmark sweep outputs may exist locally, for example `*_maxdocs20*`, `*_maxdocs40*`, or `*_target120*`.

Do not delete or commit them during the current cleanup. Later, decide whether to summarize, ignore, move to a scratch folder, or remove them.
