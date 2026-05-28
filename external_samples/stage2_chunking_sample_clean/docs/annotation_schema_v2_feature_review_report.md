# Annotation schema v2 feature review report

## Purpose

This report reviews a small biased sample of `annotation_v2` deterministic feature outputs. The goal is to assess heuristic quality, not just schema validity.

No extractor heuristics, taxonomy labels, classifiers, thresholds, MiniLM outputs, HF samples, or NLL/profiling code were changed.

## Reviewed sample

Input sample:

- `data_samples/real_hf_benchmark_v1_features_v2_review_sample.jsonl`

Reviewed output:

- `data_samples/real_hf_benchmark_v1_features_v2_review_labeled.jsonl`

Evaluation output:

- `data_samples/real_hf_benchmark_v1_features_v2_review_eval.json`

Records reviewed: 30.

Dataset distribution:

| Dataset | Records |
| --- | ---: |
| FineMath | 20 |
| FineWeb | 5 |
| FineWeb-Edu | 5 |

The sample is intentionally biased toward interesting cases: math notation, code-like text, noise-like text, high symbol density, and high digit density. It should not be treated as a representative benchmark distribution.

## Review rules

- `review_has_math_notation=true` only when the text contains explicit formulas, symbolic expressions, TeX-like notation, equation-like structure, or mathematical notation. Ordinary numbers alone do not count.
- `review_has_code=true` only when the text contains explicit code, API/help syntax, command/output blocks, markup, SQL, stack traces, or function/class/import-like syntax. Patent prose about software does not count as code.
- `review_noise_level=clean` means the main text is useful and not noticeably polluted.
- `review_noise_level=partial_noise` means useful main text exists but is mixed with navigation, UI, boilerplate, forum metadata, links, or other non-content.
- `review_noise_level=mostly_noise` means the chunk is mostly menu/footer/cookie/navigation/SEO/repetitive junk.
- `review_noise_level=unknown` means the reviewer cannot judge.

## Metrics

### `has_math_notation`

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9667 |
| Precision | 1.0000 |
| Recall | 0.9412 |
| F1 | 0.9697 |
| True positives | 16 |
| False positives | 0 |
| False negatives | 1 |
| True negatives | 13 |

False negative:

- `FineMath_000000_002`: TeX-like chemical formulas such as `${\text{C"_12"H"_22"O}}_{11}$` were not detected as math/scientific notation.

Interpretation: `has_math_notation` is reliable enough as a high-precision surface feature on this sample. It may miss chemistry/scientific formula notation unless the extractor is extended.

### `has_code`

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9667 |
| Precision | 1.0000 |
| Recall | 0.7500 |
| F1 | 0.8571 |
| True positives | 3 |
| False positives | 0 |
| False negatives | 1 |
| True negatives | 26 |

False negative:

- `FineMath_000009_000`: Maple help/API-style function signature and parameter documentation was not detected as code/API-like text.

Interpretation: `has_code` is conservative and high precision on this sample. It avoids treating patent software prose as code, which is good. The main gap is API/help documentation that is code-adjacent but mostly prose.

### `noise_level`

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6333 |

Confusion matrix:

| Review \ Extractor | clean | partial_noise |
| --- | ---: | ---: |
| clean | 19 | 9 |
| partial_noise | 2 | 0 |

No reviewed records were labeled `mostly_noise` or `unknown`.

Main mismatch types:

1. Formula-heavy math marked as `partial_noise` because high symbol/punctuation density triggered noise.
2. Patent text marked as `partial_noise` because words like `main menu` were interpreted as boilerplate markers even though they were part of the patent content.
3. UI/footer fragments in otherwise useful educational text missed as `clean`.
4. Forum metadata around useful math formula content missed as `clean`.

Interpretation: `noise_level` is structurally valid but not yet reliable as a quality label. It is currently better understood as a mixture of:

- visible boilerplate/link/list noise;
- symbol-heavy or special-format text;
- partial UI/forum residue.

For future NLL/profiling, it may be better to separate `quality.noise_level` from a surface field such as `surface.symbol_heavy` or `surface.special_format`.

## False positives and false negatives

### Math notation

False positives: none in this sample.

False negatives:

- Chemistry/scientific formula notation using TeX-like chemical formulas.

Suggested future fix:

- Add chemistry/scientific notation patterns to `has_math_notation` or introduce a separate `has_scientific_formula` feature.

### Code

False positives: none in this sample.

False negatives:

- Maple help/API-style documentation with `Calling Sequence`, function signatures, and parameters.

Suggested future fix:

- Add a conservative `has_api_or_command_syntax` feature or extend `has_code` to include API/help command signatures.

### Noise level

False positives:

- Symbol-heavy math marked as partial noise.
- Patent/reference text with content words like `main menu` marked as boilerplate.

False negatives:

- Useful text with trailing UI fragments.
- Useful math posts mixed with forum metadata.

Suggested future fix:

- Do not treat high symbol density alone as noise for math-like chunks.
- Split `symbol_heavy` from `noise_level`.
- Make boilerplate markers context-aware so `main menu` inside patent text is not automatically noise.
- Add weak signals for forum/UI residue, such as `posted by`, `responses`, `notes/highlights`, `report an issue`, and `image attributions`.

## Reliability assessment

`has_math_notation`: reliable enough for MVP as a high-precision surface flag, with known recall gaps for chemistry/scientific notation.

`has_code`: reliable enough for MVP as a conservative high-precision flag, with known recall gaps for API/help documentation.

`noise_level`: useful as an exploratory signal, but not reliable enough as a headline quality label. It should be treated cautiously until symbol-heavy format and true noise are separated.

## Should extractor be changed now?

Not immediately.

The review sample is small and intentionally biased. The best next move is not broad tuning, but one narrow design decision:

```text
Separate symbol-heavy/special-format detection from quality/noise.
```

After that, a small heuristic pass can be considered:

- add `surface.has_scientific_formula` or improve math notation for chemistry formulas;
- add `surface.has_api_or_command_syntax`;
- reduce noise false positives from high symbol density;
- add explicit UI/forum residue markers.

## Recommended next step

Immediate next implementation step:

```text
Revise annotation_v2 deterministic schema to split surface special-format signals from quality noise.
```

Concretely, add or plan fields such as:

- `surface.is_symbol_heavy`;
- `surface.has_scientific_formula`;
- `surface.has_api_or_command_syntax`;
- `quality.has_ui_residue`;
- `quality.has_forum_residue`.

Do this as a small schema-v2 deterministic feature pass, not as topic classification and not as MiniLM tuning.

## What not to do yet

- Do not tune MiniLM thresholds based on this review.
- Do not change semantic taxonomy.
- Do not implement weak `topic.domain` until deterministic quality/surface axes are clearer.
- Do not use `noise_level` alone as a data filtering rule.
- Do not treat this 30-record biased review as final accuracy evidence.
