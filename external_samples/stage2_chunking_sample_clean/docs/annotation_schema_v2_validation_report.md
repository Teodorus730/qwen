# Annotation schema v2 validation report

## Purpose

This report documents a lightweight validation layer for `annotation_v2` deterministic feature records.

The goal is quality control for the feature layer, not classifier tuning. The validation checks schema consistency, deterministic text statistics, bounded numeric fields, duplicate chunk ids, and basic feature sanity warnings.

This report does not claim that the heuristics are semantically accurate. It only establishes that the generated `annotation_v2` records are structurally stable enough to review and use as downstream grouping metadata.

## Files validated

Canonical feature outputs:

| Dataset | Feature file | Records | Errors | Warnings |
| --- | --- | ---: | ---: | ---: |
| FineWeb | `data_samples/real_samples/real_hf_benchmark_v1_fineweb_features_v2.jsonl` | 113 | 0 | 0 |
| FineWeb-Edu | `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_features_v2.jsonl` | 88 | 0 | 0 |
| FineMath | `data_samples/real_samples/real_hf_benchmark_v1_finemath_features_v2.jsonl` | 195 | 0 | 0 |
| Total | all three files | 396 | 0 | 0 |

Validation summaries:

- `data_samples/real_hf_benchmark_v1_fineweb_features_v2_validation.json`
- `data_samples/real_hf_benchmark_v1_fineweb_edu_features_v2_validation.json`
- `data_samples/real_hf_benchmark_v1_finemath_features_v2_validation.json`
- `data_samples/real_hf_benchmark_v1_features_v2_validation_summary.json`

Review sample:

- `data_samples/real_hf_benchmark_v1_features_v2_review_sample.jsonl`

## Validation scripts

Created:

- `scripts/validate_annotation_features_v2.py`
- `scripts/sample_annotation_features_v2_review.py`
- `scripts/run_annotation_features_v2_validation.py`

The validator checks:

- valid JSONL;
- required top-level `text`;
- required `annotation_v2` sections;
- provenance `dataset` and `chunk_id`;
- duplicate `chunk_id`;
- deterministic `char_count` and `byte_count`;
- text stats non-negative / bounded logic;
- required surface booleans;
- surface density values in `[0, 1]`;
- `has_numbers` agreement with digit presence;
- `quality.noise_level` enum;
- `quality.noise_score` in `[0, 1]`;
- `noise_reasons` as a list of strings;
- non-empty schema version containing `annotation_v2`.

## Schema consistency status

All three canonical files passed validation:

- errors: 0;
- warnings: 0;
- duplicate chunk ids: 0;
- schema version: `annotation_v2_deterministic_features_v1` for all 396 records;
- `has_numbers` matches actual digit presence in all checked records.

This means the deterministic feature layer is structurally stable.

## Feature distribution sanity notes

Validator counts match the earlier feature extractor report.

| Dataset | Clean | Partial noise | Mostly noise | Math notation | Code | Numbers | Table/list | URLs/links | Boilerplate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 104 | 8 | 1 | 0 | 3 | 98 | 4 | 3 | 9 |
| FineWeb-Edu | 81 | 7 | 0 | 0 | 1 | 70 | 2 | 10 | 6 |
| FineMath | 156 | 39 | 0 | 67 | 13 | 173 | 24 | 5 | 0 |

Sanity interpretation:

- FineMath has the expected concentration of math notation, symbol-heavy structure, table/list signals, and code-like/math-help fragments.
- FineWeb and FineWeb-Edu carry most explicit boilerplate markers.
- `has_numbers` is intentionally broad and should be treated as a surface statistic, not a math label.
- FineMath `partial_noise` often means symbol-heavy or TeX-like format; it should not automatically be interpreted as bad data.

## Review sample

The review sampler created 30 compact review records:

- FineWeb: 5 records;
- FineWeb-Edu: 5 records;
- FineMath: 20 records.

The sample intentionally over-represents interesting records after guaranteeing per-dataset coverage:

- `has_math_notation=true`: 16;
- `has_code=true`: 3;
- `has_boilerplate_markers=true`: 1;
- `partial_noise`: 9;
- high `symbol_density >= 0.12`: 9;
- high `digit_density >= 0.08`: 5.

This is a review sample, not a representative distribution sample.

## Suspicious cases to inspect manually

The review sample is designed to expose cases where deterministic features may be questionable:

- math-heavy chunks marked as `partial_noise` because of high symbol and punctuation density;
- patent/reference-like or technical procedure text that may trigger code-like patterns;
- boilerplate marker hits inside otherwise clean prose;
- FineMath records without math notation but with numeric or table/list structure;
- URL/link-heavy records where the useful prose may still dominate.

No schema errors were found, but heuristic quality still needs manual review.

## Stability assessment

The deterministic feature layer is stable enough for the next stage:

- schema is consistent;
- records remain valid JSONL;
- original chunk fields are preserved;
- deterministic counts match the actual text;
- feature ranges are bounded;
- review candidates can be generated reproducibly.

This is a good foundation for downstream grouping and later annotation schema v2 work.

## What this does not prove

Passing validation does not prove that:

- `has_math_notation` has high recall for all math content;
- `has_code` precisely separates code from technical reference text;
- `noise_level` detects low-quality generated prose;
- deterministic features can replace semantic topic labels.

Schema validity and heuristic accuracy are separate questions.

## Recommended next step

Immediate next step:

```text
Manually review data_samples/real_hf_benchmark_v1_features_v2_review_sample.jsonl.
```

Use the review placeholders to mark:

- whether `has_math_notation` looks correct;
- whether `has_code` looks correct;
- whether `noise_level` is reasonable;
- short notes for false positives and false negatives.

After that review, decide whether to make a very small deterministic heuristic pass or move to weak `topic.domain` v2. Do not tune MiniLM, taxonomy, or thresholds before this review.

## What not to do yet

- Do not implement weak topic labeling in this validation step.
- Do not run a new embedding model bake-off.
- Do not change semantic taxonomy.
- Do not merge deterministic surface features back into old `source_type`.
- Do not treat `noise_level` as final data-quality truth.
