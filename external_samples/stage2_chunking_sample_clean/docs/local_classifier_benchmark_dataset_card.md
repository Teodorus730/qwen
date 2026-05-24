# Local classifier benchmark dataset card

## What this is

`examples/local_docs_classifier_benchmark.jsonl` is a hand-authored synthetic benchmark for local stage2 smoke tests. It is small, local, and designed to validate label plumbing and rule-based classifier behavior.

## Why it exists

The benchmark checks whether `expected_*` fields survive chunking, whether predicted labels can be compared to golden labels, and whether simple rule changes improve obvious cases without downloading data or models.

## Covered categories

- math calculus
- math algebra
- code documentation
- commercial product pages
- forum Q&A
- biology, physics, and environmental science articles
- urban infrastructure explainers
- multilingual English/Russian notes
- pure boilerplate/navigation
- wiki/reference style articles
- legal/government notices
- news-like articles

## Easy vs hard cases

Easy cases use direct, obvious category cues. Hard cases mix useful text with footer/cookie noise, combine code and math, include markdown or table-like formatting, or use headings that could be mistaken for educational content.

## What this is not

This is not real web data and not a FineWeb sample. It must not be used for scientific conclusions about FineWeb, FineWeb-Edu, FineMath, or model training data quality.

## Labels

Expected labels are human-authored and heuristic. They are useful for regression and smoke tests, not as a final taxonomy.

## Known limitations

- Synthetic wording is cleaner than real web text.
- Keyword-based rules can overfit to these examples.
- Mixed chunks are labeled at chunk level, not span level.
- Boilerplate is detected as a label, not removed.

## Regeneration

Generate chunks:

```bash
python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_classifier_benchmark.jsonl --max-docs 100 --out data_samples\classifier_benchmark_chunks.jsonl --stats-out data_samples\classifier_benchmark_run_stats.json
```

Generate labels and evaluate:

```bash
python scripts\classify_chunks_rule_based.py --input data_samples\classifier_benchmark_chunks.jsonl --output data_samples\classifier_benchmark_labeled.jsonl
python scripts\evaluate_chunk_labels.py --input data_samples\classifier_benchmark_labeled.jsonl
```

Or run the local pipeline:

```bash
python scripts\run_local_benchmark_pipeline.py
```

## Relation to future real HF samples

This benchmark should stay as a fast local regression check. Real HF streaming samples can be added later for realism, but they should not replace this tiny deterministic smoke test.
