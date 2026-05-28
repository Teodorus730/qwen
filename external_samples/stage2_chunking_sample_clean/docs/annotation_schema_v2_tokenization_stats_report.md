# Annotation schema v2 tokenization stats report

## Purpose

This report adds tokenizer-aware statistics to the `annotation_v2` feature layer. This is not topic classification, MiniLM tuning, NLL scoring, or model training. It is a reproducible text/token statistics layer needed before future NLL, logprob, perplexity, and data-selection profiling.

## Tokenizer policy recap

Project assumption:

- model weights are trained from scratch;
- architecture target is Qwen-like / Qwen-family;
- tokenizer is frozen infrastructure;
- using a Qwen-family tokenizer does not mean using Qwen pretrained weights.

Canonical MVP tokenizer used here:

| Field | Value |
| --- | --- |
| tokenizer_name | `Qwen/Qwen3.5-0.8B-Base` |
| revision | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| tokenizer_class | `Qwen2Tokenizer` |
| vocab_size | 248044 |
| len_tokenizer | 248077 |
| bos_token_id | null |
| eos_token_id | 248044 |
| pad_token_id | 248044 |
| unk_token_id | null |
| add_special_tokens | false |

The tokenizer was loaded via `AutoTokenizer` only. No model weights were loaded.

## Files processed

Inputs:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_features_v2_refined.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_features_v2_refined.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_features_v2_refined.jsonl`

Outputs:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_features_v2_tokenized.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_features_v2_tokenized.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_features_v2_tokenized.jsonl`

Validation summaries:

- `data_samples/real_hf_benchmark_v1_fineweb_features_v2_tokenized_validation.json`
- `data_samples/real_hf_benchmark_v1_fineweb_edu_features_v2_tokenized_validation.json`
- `data_samples/real_hf_benchmark_v1_finemath_features_v2_tokenized_validation.json`

Aggregate summary:

- `data_samples/real_hf_benchmark_v1_tokenization_stats_summary.json`

## Schema changes

Tokenized records use:

- `annotation_v2.schema_version = annotation_v2_deterministic_features_v2_tokenized_v1`

Added under `annotation_v2.tokenizer`:

- `tokenizer_name`
- `revision`
- `tokenizer_class`
- `vocab_size`
- `len_tokenizer`
- `bos_token_id`
- `eos_token_id`
- `pad_token_id`
- `unk_token_id`
- `add_special_tokens`

Added under `annotation_v2.text_stats`:

- `token_count`
- `token_per_byte`
- `tokens_per_char`
- `bytes_per_token`
- `chars_per_token`
- `tokens_per_word_rough`
- `unique_word_count_rough`
- `avg_word_length_rough`
- `latin_char_count`
- `cyrillic_char_count`
- `latin_char_ratio`
- `cyrillic_char_ratio`

## Validation status

| Dataset | Records | Errors | Warnings |
| --- | ---: | ---: | ---: |
| FineWeb | 113 | 0 | 0 |
| FineWeb-Edu | 88 | 0 | 0 |
| FineMath | 195 | 0 | 0 |
| Total | 396 | 0 | 0 |

## Token distributions by dataset

| Dataset | Records | Mean token_count | Mean token_per_byte | Mean bytes_per_token | Mean tokens_per_word_rough |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 113 | 373.0000 | 0.2202 | 4.6043 | 1.3025 |
| FineWeb-Edu | 88 | 359.8750 | 0.2168 | 4.6611 | 1.2948 |
| FineMath | 195 | 177.7538 | 0.2896 | 3.7367 | 1.6088 |
| Overall | 396 | 273.9394 | 0.2536 | 4.1897 | 1.4516 |

FineMath chunks are shorter on average in this benchmark, but more token-dense. This matches the expectation that formulas, symbols, compact equations, and math/code-like notation produce more tokens per byte than prose.

## Token stats by surface and noise features

| Group | Records | Mean token_per_byte | Mean tokens_per_word_rough |
| --- | ---: | ---: | ---: |
| `has_math_notation=true` | 105 | 0.3357 | 1.7899 |
| `has_math_notation=false` | 291 | 0.2240 | 1.3296 |
| `has_code=true` | 34 | 0.3261 | 1.8629 |
| `has_code=false` | 362 | 0.2468 | 1.4130 |
| `is_symbol_heavy=true` | 45 | 0.4119 | 2.1552 |
| `is_symbol_heavy=false` | 351 | 0.2333 | 1.3614 |
| `noise_level=clean` | 373 | 0.2556 | 1.4576 |
| `noise_level=partial_noise` | 21 | 0.2212 | 1.3472 |
| `noise_level=mostly_noise` | 2 | 0.2330 | 1.4397 |

The strongest token-density signal is not noise. It is special surface format:

- math notation;
- code/API-like syntax;
- symbol-heavy content.

This supports the recent refinement: symbol-heavy useful text should not be treated as noise by default.

## Highest token-density examples

The top token-per-byte records are mostly FineMath:

| Rank | chunk_id | Dataset | token_per_byte | token_count | Notes |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `FineMath_000016_002` | FineMath | 0.6907 | 306 | Matrix multiplication equations. |
| 2 | `FineMath_000002_006` | FineMath | 0.5726 | 142 | Trigonometric identity proof. |
| 3 | `FineMath_000006_001` | FineMath | 0.5429 | 209 | Asymptote/geometry code-like drawing block. |
| 4 | `FineMath_000016_004` | FineMath | 0.5383 | 260 | Numeric matrix calculation. |
| 5 | `FineMath_000002_001` | FineMath | 0.5375 | 129 | Trig formula answer. |
| 6 | `FineMath_000002_008` | FineMath | 0.5370 | 138 | Trig symbolic computation. |
| 7 | `FineMath_000009_002` | FineMath | 0.4964 | 555 | Maple matrix command/output block. |
| 8 | `FineMath_000019_001` | FineMath | 0.4958 | 176 | Enumerated probability outcomes. |
| 9 | `FineMath_000009_003` | FineMath | 0.4938 | 636 | Maple symbolic matrix output. |
| 10 | `FineMath_000025_002` | FineMath | 0.4824 | 123 | TeX derivative formulas. |

The highest token-count records include both formula-heavy FineMath and longer FineWeb/FineWeb-Edu prose/table-like chunks. This means `token_count` and `token_per_byte` capture different things:

- `token_count`: absolute sequence length;
- `token_per_byte`: tokenizer density / compression difficulty.

## Answers to the benchmark questions

Are FineMath chunks more token-dense than FineWeb/FineWeb-Edu?

- Yes in this dev benchmark. FineMath mean `token_per_byte` is 0.2896 vs FineWeb 0.2202 and FineWeb-Edu 0.2168.

Do `has_math_notation` chunks have higher token density?

- Yes. Mean `token_per_byte` is 0.3357 with math notation vs 0.2240 without.

Do `has_code` chunks have higher token density?

- Yes. Mean `token_per_byte` is 0.3261 with code/API-like syntax vs 0.2468 without.

Do `is_symbol_heavy` chunks have higher token density?

- Strongly yes. Mean `token_per_byte` is 0.4119 vs 0.2333.

Does `noise_level` correlate with token density?

- Not strongly here. `partial_noise` and `mostly_noise` are not more token-dense than `clean`. This reinforces that future profiling should separate quality/noise from special-format density.

## Implications for future NLL/profiling

Tokenization-aware stats look useful for future NLL/probability profiling because they provide cheap grouping variables before any model scoring:

- high token density vs normal prose;
- math/code/symbol-heavy surface formats;
- absolute token length;
- tokens per rough word;
- language-script mix via Latin/Cyrillic ratios.

For future NLL analysis, compare chunks within similar tokenization regimes. Formula-heavy chunks may naturally produce different token-level loss profiles than prose, even if they are clean and useful.

## Caveats

- This is the real_hf_benchmark_v1 dev benchmark, not a held-out test.
- Token stats depend on the chosen tokenizer and pinned revision.
- No NLL, logits, or model weights were used.
- `token_per_byte` is a tokenizer-density feature, not a quality score.
- `tokens_per_word_rough` depends on the existing rough word-count heuristic.

## Recommended next step

Keep the tokenized `annotation_v2` layer as a reproducible feature layer. The next practical step should be annotation_v2 pseudo-gold for the same 120 benchmark chunks under the new multi-axis schema, before implementing weak `topic.domain` v2.
