# Annotation schema v2 feature extractor report

## Purpose

This report documents the first implementation step for annotation schema v2: a deterministic feature layer for existing chunks.

The goal was not to replace the old classifiers, tune MiniLM, change taxonomy, or implement a new topic classifier. The goal was to add cheap, reproducible fields that separate provenance/text shape/surface signals/noise from semantic topic labels.

## Why deterministic features first

`real_hf_benchmark_v1` showed that the old single `source_type` label mixed too many axes: provenance, topic, genre/function, surface format, and quality/noise.

Deterministic features are the safest first v2 layer because they:

- do not require HF streaming;
- do not require MiniLM or model downloads;
- do not modify existing `source_type`, `domain`, `field`, or `subfield`;
- are explainable and reproducible;
- can support later NLL/logprob/perplexity grouping even before topic labeling is solved.

## Files

Scripts:

- `scripts/extract_annotation_features_v2.py`
- `scripts/inspect_annotation_features_v2.py`
- `scripts/sanity_check_features_v2_against_pseudo_gold.py`

Feature outputs:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_features_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_features_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_features_v2.jsonl`

Summary outputs:

- `data_samples/real_hf_benchmark_v1_features_v2_summary.json`
- `data_samples/real_hf_benchmark_v1_features_v2_summary_fineweb.json`
- `data_samples/real_hf_benchmark_v1_features_v2_summary_fineweb_edu.json`
- `data_samples/real_hf_benchmark_v1_features_v2_summary_finemath.json`
- `data_samples/real_hf_benchmark_v1_features_v2_sanity.json`

## Input chunks

The extractor used raw real benchmark chunk files, not labeled outputs:

| Dataset | Input file | Records |
| --- | --- | ---: |
| FineWeb | `data_samples/real_samples/real_hf_benchmark_v1_fineweb_chunks.jsonl` | 113 |
| FineWeb-Edu | `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_chunks.jsonl` | 88 |
| FineMath | `data_samples/real_samples/real_hf_benchmark_v1_finemath_chunks.jsonl` | 195 |

Total: 396 chunks.

## Implemented fields

Each output record preserves the original chunk fields and adds:

```json
"annotation_v2": {
  "provenance": {},
  "text_stats": {},
  "surface": {},
  "quality": {},
  "schema_version": "annotation_v2_deterministic_features_v1"
}
```

### Provenance

Implemented fields:

- `dataset`
- `source_dataset`
- `source_config`
- `source_split`
- `chunk_id`
- `document_id`

The current benchmark chunks mostly contain `dataset` and `chunk_id`; missing provenance fields remain `null` rather than inferred from text.

### Text stats

Implemented fields:

- `char_count`
- `byte_count`
- `line_count`
- `avg_line_length`
- `nonempty_line_count`
- `word_count_rough`

### Surface features

Implemented fields:

- `has_math_notation`
- `has_code`
- `has_numbers`
- `has_table_or_list`
- `has_urls_or_links`
- `has_boilerplate_markers`
- `symbol_density`
- `digit_density`
- `uppercase_ratio`
- `punctuation_density`

### Quality

Implemented fields:

- `noise_level`: `clean`, `partial_noise`, `mostly_noise`, or `unknown`
- `noise_score`
- `noise_reasons`

The noise heuristic is intentionally simple. It uses explicit surface signals: boilerplate markers, URLs, table/list structure, high symbol or punctuation density, repeated lines, and very short/missing text.

## Feature distributions

### By dataset

| Dataset | Records | Mean char count | Mean line count |
| --- | ---: | ---: | ---: |
| FineWeb | 113 | 1704.8496 | 1.8142 |
| FineWeb-Edu | 88 | 1664.2727 | 1.4773 |
| FineMath | 195 | 636.5590 | 9.4718 |
| All | 396 | 1169.7803 | 5.5101 |

FineMath chunks are shorter and more line-structured, which matches the observed math/help/Q&A style data.

### Surface flags

| Dataset | Math notation | Code | Numbers | Table/list | URLs/links | Boilerplate markers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 0 / 113 | 3 / 113 | 98 / 113 | 4 / 113 | 3 / 113 | 9 / 113 |
| FineWeb-Edu | 0 / 88 | 1 / 88 | 70 / 88 | 2 / 88 | 10 / 88 | 6 / 88 |
| FineMath | 67 / 195 | 13 / 195 | 173 / 195 | 24 / 195 | 5 / 195 | 0 / 195 |
| All | 67 / 396 | 17 / 396 | 341 / 396 | 30 / 396 | 18 / 396 | 15 / 396 |

Most math notation flags occur in FineMath, which is desirable. `has_numbers` is broad and should be treated as a surface statistic rather than a math label.

### Noise level

| Dataset | Clean | Partial noise | Mostly noise |
| --- | ---: | ---: | ---: |
| FineWeb | 104 | 8 | 1 |
| FineWeb-Edu | 81 | 7 | 0 |
| FineMath | 156 | 39 | 0 |
| All | 341 | 54 | 1 |

Top noise reasons:

- `high_symbol_density`: 45
- `high_punctuation_density`: 37
- `table_or_list_structure`: 30
- `urls_or_links`: 18
- `boilerplate_markers`: 15
- `many_urls`: 3
- `high_uppercase_ratio`: 2

FineMath partial-noise cases are mostly formula-heavy or TeX-like chunks. This should not automatically mean bad data; for NLL/profiling it may be useful as a "symbol-heavy math" control.

## Sanity check against pseudo-gold

Pseudo-gold file:

- `data_samples/real_hf_benchmark_v1_pseudo_gold_v2.jsonl`

Joined records:

- 120 / 120 pseudo-gold records matched feature records by `chunk_id`.

Lightweight checks:

| Check | Matches | Rate |
| --- | ---: | ---: |
| `expected_source_type=math` has `has_math_notation=true` | 8 / 15 | 0.5333 |
| `expected_source_type=code` has `has_code=true` | 1 / 2 | 0.5000 |
| `expected_source_type=boilerplate_or_noise` has `partial_noise` or `mostly_noise` | 0 / 2 | 0.0000 |
| FineMath reviewed records have `has_math_notation=true` | 13 / 40 | 0.3250 |
| FineWeb reviewed records have boilerplate markers | 7 / 40 | 0.1750 |
| FineWeb-Edu reviewed records have boilerplate markers | 2 / 40 | 0.0500 |

## Interpretation

### What looks useful

The deterministic layer already separates several important axes:

- FineMath has much higher math notation, symbol density, table/list, and line structure than FineWeb/FineWeb-Edu.
- Boilerplate markers appear mostly in FineWeb and FineWeb-Edu, not in FineMath.
- URL/link markers catch a small but visible part of real web noise.
- `noise_reasons` make the heuristic auditable, which is better than a black-box quality label.

These fields are useful enough to keep for future NLL/probability profiling as grouping and control variables.

### Likely true positives

Strong-looking examples include:

- trigonometry and algebra chunks with formulas marked as `has_math_notation`;
- Maple/linear algebra command/help chunks marked as code-like or symbol-heavy;
- Feedly/product/app page with links and list structure marked as mostly noise;
- web pages with visible footer/menu/cookie/product markers marked as partial noise.

### Known false positives

Some patent/reference chunks trigger `has_code` because they contain terms such as function, program, pass-through mode, or structured technical procedure. This is expected risk: patent text often looks code-like without being code.

Some formula-heavy math chunks receive `partial_noise` because high symbol/punctuation density is also a noise signal. For math-heavy data this should be interpreted as "symbol-heavy/special format", not necessarily low quality.

### Known false negatives

Two pseudo-gold `boilerplate_or_noise` records are low-quality generated/SEO prose about triangles. They do not contain explicit boilerplate markers, repeated lines, high symbol density, or obvious link/menu structure. A deterministic surface heuristic does not reliably detect this kind of semantic low quality.

One pseudo-gold `code` record is a Maple help page that is mostly prose plus command signatures. It is close to API/reference documentation and may require a broader `technical_reference` or `software_help` feature later.

## Limitations

The feature layer does not decide semantic topic. It deliberately does not emit `topic.domain`, `topic.field`, or `topic.subfield`.

The quality heuristic is not a full data-quality model. It catches visible surface noise better than low-quality generated prose.

The math/code flags are conservative and should be evaluated as feature signals, not as final `source_type`.

The current summary is based on `real_hf_benchmark_v1`, which is a dev benchmark. It should not be overfit indefinitely.

## Usefulness for NLL/probability profiling

The feature layer is useful for future profiling because it creates cheap grouping variables:

- normal prose vs symbol-heavy chunks;
- math-notation chunks vs non-math chunks;
- code/API-like chunks vs prose;
- URL/boilerplate/list-heavy chunks vs cleaner text;
- short/line-heavy FineMath chunks vs longer web prose chunks.

These groups can help explain NLL/logprob differences without forcing everything through a fragile single `source_type`.

## Recommended next step

Next small implementation step:

1. Review a small sample of `annotation_v2` outputs manually.
2. Decide whether `quality.noise_level` should distinguish `symbol_heavy` from actual noise.
3. Add a lightweight validation/schema check for `annotation_v2` records.
4. Only after that, design a separate weak `topic.domain` v2 labeler with abstention.

## What not to tune yet

Do not tune MiniLM or semantic thresholds based on these deterministic features.

Do not merge `has_math_notation` or `has_code` back into old `source_type`.

Do not implement full OpenAlex-style field/subfield topic labeling yet.

Do not use these features as final truth labels; they are explainable signals for analysis and downstream grouping.
