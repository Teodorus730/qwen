# Chunk JSONL schema

This document describes the current local MVP schema for stage2 chunking and labeling. It is intentionally small and practical: enough for smoke tests, benchmark evaluation, and later probability profiling.

## 1. Raw/local input schema

Local input files are JSONL files where each line is one source document.

Required:

- `text`: source document text.

Optional:

- `id` / `doc_id`: source document identifier, used for human tracking when present.
- `dataset`: dataset/source label for the row.
- `source_type`: broad text format/source label for the row.
- `expected_source_type`: benchmark/golden source_type.
- `expected_domain`: benchmark/golden domain.
- `expected_field`: benchmark/golden field.
- `expected_subfield`: benchmark/golden subfield, may be null.
- `expected_label_note`: short human note explaining the expected label.

## 2. Chunk sample output schema

Used by `data_samples/edge_case_chunks_sample.jsonl` and future chunk samples.

Required:

- `chunk_id`: unique chunk identifier.
- `dataset`: dataset/source label.
- `source_type`: text format/source label. This may be inherited from input metadata or guessed from dataset.
- `domain`: predicted or placeholder domain, may be null.
- `field`: predicted or placeholder field, may be null.
- `subfield`: predicted or placeholder subfield, may be null.
- `confidence`: heuristic confidence, may be null.
- `token_count`: positive integer token estimate.
- `text`: chunk text.

Optional benchmark fields:

- `expected_source_type`
- `expected_domain`
- `expected_field`
- `expected_subfield`
- `expected_label_note`

## 3. Labeled chunk output schema

Used by `data_samples/edge_case_chunks_labeled.jsonl` and classifier outputs.

Includes all chunk sample fields above, plus:

- `label_method`: how the label fields were produced.

## 4. `source_type` vs `domain/field/subfield`

- `source_type` is the type or format of the text, for example `forum_qa`, `code`, `commercial_product`, or `boilerplate_or_noise`.
- `domain/field/subfield` describes the topic or knowledge area, for example `stem/mathematics/calculus` or `science/biology/article`.

These two layers are related but separate. A code documentation page can contain math, and an educational page can contain boilerplate.

## 5. Current `label_method` values

- `rule_based_keyword_v1`: transparent keyword/rule baseline.
- `rule_based_unknown`: fallback when no rule is confident.
- `existing_label_passthrough`: classifier preserved existing labels.
- `embedding_nearest_label_v1`: future/optional embedding nearest-label baseline.
- `embedding_nearest_label_low_confidence`: future/optional embedding baseline below threshold.

## 6. Important notes

- Rule-based labels are baseline/smoke-test labels, not final taxonomy labels.
- The taxonomy is not final.
- `domain`, `field`, and `subfield` can be null.
- `confidence` is heuristic and should not be interpreted as calibrated probability.
- Later OpenAlex-like taxonomy should be used as a taxonomy source, not as a magic classifier.
- `expected_*` fields are benchmark/golden fields, not predictions.
