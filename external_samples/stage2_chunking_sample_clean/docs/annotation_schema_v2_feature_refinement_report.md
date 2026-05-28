# Annotation schema v2 feature refinement report

## Purpose

The first deterministic `annotation_v2` feature extractor validated structurally, but the 30-record review sample showed that `noise_level` was weaker than the surface flags. The main issue was conceptual: symbol-heavy or formula-heavy useful text was sometimes treated as partial noise.

This refinement separates special-format signals from actual residue/noise signals while keeping the old `annotation_v2` fields backward compatible.

## Changes made

Schema version:

- `annotation_v2_deterministic_features_v2`

New `annotation_v2.surface` fields:

- `is_symbol_heavy`
- `has_scientific_formula`
- `has_api_or_command_syntax`

New `annotation_v2.quality` fields:

- `has_ui_residue`
- `has_forum_residue`

The refined `noise_level` logic no longer treats high symbol or punctuation density as noise by itself. Symbol-heavy math, chemistry, and formula text can remain `clean` unless there are actual boilerplate, UI, URL, list/menu, or forum residue signals.

## Validation

Validated refined outputs:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_features_v2_refined.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_features_v2_refined.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_features_v2_refined.jsonl`

Aggregate result:

| Dataset | Records | Errors | Warnings |
| --- | ---: | ---: | ---: |
| FineWeb | 113 | 0 | 0 |
| FineWeb-Edu | 88 | 0 | 0 |
| FineMath | 195 | 0 | 0 |
| Total | 396 | 0 | 0 |

## Old vs refined review metrics

Review sample:

- 30 records
- FineMath: 20
- FineWeb: 5
- FineWeb-Edu: 5

| Feature | Old accuracy | Refined accuracy | Old precision | Refined precision | Old recall | Refined recall | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `has_math_notation` | 0.9667 | 0.9667 | 1.0000 | 0.9444 | 0.9412 | 1.0000 | Chemistry formula FN fixed; one patent interface FP added. |
| `has_code` | 0.9667 | 0.9667 | 1.0000 | 0.8000 | 0.7500 | 1.0000 | Maple/API-style FN fixed; one trig expression FP added. |
| `noise_level` | 0.6333 | 0.9000 | n/a | n/a | n/a | n/a | Major improvement from separating formula-heavy useful text from noise. |

## New field coverage

Across all refined benchmark v1 feature records:

| Field | FineWeb | FineWeb-Edu | FineMath |
| --- | ---: | ---: | ---: |
| `is_symbol_heavy` | 0 | 0 | 45 |
| `has_scientific_formula` | 6 | 2 | 55 |
| `has_api_or_command_syntax` | 16 | 19 | 55 |
| `has_ui_residue` | 28 | 18 | 18 |
| `has_forum_residue` | 1 | 1 | 1 |

Noise distribution after refinement:

| Dataset | Clean | Partial noise | Mostly noise |
| --- | ---: | ---: | ---: |
| FineWeb | 105 | 7 | 1 |
| FineWeb-Edu | 79 | 8 | 1 |
| FineMath | 189 | 6 | 0 |

## Remaining errors

Refined `has_math_notation` false positive:

- `FineWeb_000009_043`: patent/reference prose about RS-422/RS-485 interfaces. The scientific formula detector treats technical interface notation as formula-like.

Refined `has_code` false positive:

- `FineMath_000002_000`: trigonometric expression with function-like syntax was flagged as API/command syntax. It is math notation, not code.

Refined `noise_level` misses:

- `FineMath_000026_026`: UI residue is detected, but the score remains below `partial_noise`.
- `FineMath_000002_007`: forum residue is detected, but useful math dominates the score.

Refined `noise_level` false positive:

- `FineWeb_000009_051`: patent prose with display markers is detected as UI/forum residue, but the reviewed label considered it clean patent content.

## Interpretation

The refinement achieved the main goal: useful formula-heavy text is no longer broadly penalized as noisy. `noise_level` improved from 0.6333 to 0.9000 on the reviewed sample.

The trade-off is acceptable for this stage:

- `has_math_notation` recall improved to 1.0000, with one technical-notation false positive.
- `has_code` recall improved to 1.0000, with one math-expression false positive.
- The new fields make these cases auditable instead of collapsing them into one overloaded noise score.

## Usefulness for NLL/profiling

The refined deterministic layer is better suited for future NLL/logprob/perplexity profiling because it separates:

- useful special formats, such as math formulas and API/help syntax;
- actual UI/forum/boilerplate residue;
- coarse quality/noise level.

This allows future profiling to group chunks by surface and quality signals without assuming that symbol-heavy text is low quality.

## What not to tune yet

Do not tune topic/domain labeling from this result.

Do not lower or raise MiniLM thresholds.

Do not expand the semantic taxonomy based on these deterministic feature metrics.

Do not treat the 30-record review sample as a final test set. It is a quality-control sample for heuristics, not a representative benchmark.

## Recommended next step

Use the refined outputs as the current deterministic feature layer, then run a small manual review pass focused only on the remaining edge cases:

- technical interface notation vs scientific formula;
- math function syntax vs API/command syntax;
- UI/forum residue that should or should not affect `noise_level`.

After that, the next implementation step should be tokenization-aware stats once the tokenizer policy is fixed.
