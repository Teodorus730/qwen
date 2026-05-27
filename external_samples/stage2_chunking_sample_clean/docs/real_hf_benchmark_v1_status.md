# Real HF benchmark v1 status

## Purpose

`real_hf_benchmark_v1` is a small real-data benchmark for checking whether the current stage2 labeling hypotheses hold outside the synthetic benchmark:

- rule-based labels may be useful for `source_type`;
- MiniLM nearest-label embeddings may be useful for `domain` / `field` / `subfield`;
- hybrid labels may be more useful than either baseline alone.

These are still hypotheses. This benchmark prepares review candidates for pseudo-gold labeling; it does not prove classifier quality by itself.

## Datasets used

All samples were collected with Hugging Face streaming and reservoir sampling. No full dataset processing was run.

| Dataset label | HF dataset | Config | Split | Text field | Docs sampled | Source type injected |
| --- | --- | --- | --- | --- | ---: | --- |
| FineWeb | `HuggingFaceFW/fineweb` | `sample-10BT` | `train` | `text` | 30 | `unknown` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `train` | `text` | 30 | `unknown` |
| FineMath | `HuggingFaceTB/finemath` | `finemath-4plus` | `train` | `text` | 30 | `unknown` |

Sampling parameters:

- `stream-limit`: 1000
- `sample-size`: 30 per dataset
- `seed`: 42
- `min-text-chars`: 500

OpenWebMath is not part of the current benchmark. It remains optional later comparison only.

## Availability checks

All three dataset/config/split combinations were checked with `streaming=True` and one example before sampling.

| Dataset | Status | Notes |
| --- | --- | --- |
| FineWeb | OK | First streamed example had `text`; HF request emitted retry/unauthenticated warnings but succeeded. |
| FineWeb-Edu | OK | First streamed example had `text`; sampling succeeded. |
| FineMath | OK | First streamed example had `text`; sampling succeeded. |

Stage2-local HF cache variables were used during checks/sampling to avoid user-cache permission noise.

## Generated files

Sampled docs:

- `examples/real_hf_benchmark_v1_fineweb_docs.jsonl`
- `examples/real_hf_benchmark_v1_fineweb_edu_docs.jsonl`
- `examples/real_hf_benchmark_v1_finemath_docs.jsonl`

Pipeline outputs are under `data_samples/real_samples/` with the `real_hf_benchmark_v1_*` prefix:

- `*_chunks.jsonl`
- `*_labeled_rule_based.jsonl`
- `*_labeled_lexical.jsonl`
- `*_labeled_embedding_minilm.jsonl`
- `*_labeled_hybrid.jsonl`
- `*_run_stats.json`

Unified review candidates:

- `data_samples/real_hf_benchmark_v1_review_candidates.jsonl`

## Chunk counts

| Dataset | Docs | Chunks |
| --- | ---: | ---: |
| FineWeb | 30 | 113 |
| FineWeb-Edu | 30 | 88 |
| FineMath | 30 | 195 |
| Total | 90 | 396 |

## Hybrid prediction distributions

### FineWeb

`source_type`:

- `unknown`: 75
- `educational`: 30
- `commercial_product`: 8

`domain`:

- `null`: 75
- `education`: 26
- `commercial`: 8
- `science`: 4

`label_method`:

- `hybrid_rule_fallback_low_confidence`: 112
- `hybrid_rule_source_type_minilm_domain`: 1

### FineWeb-Edu

`source_type`:

- `unknown`: 52
- `educational`: 31
- `commercial_product`: 5

`domain`:

- `null`: 52
- `education`: 22
- `science`: 9
- `commercial`: 5

`label_method`:

- `hybrid_rule_fallback_low_confidence`: 85
- `hybrid_rule_source_type_minilm_domain`: 3

### FineMath

`source_type`:

- `unknown`: 118
- `educational`: 53
- `math`: 22
- `forum_qa`: 1
- `commercial_product`: 1

`domain`:

- `null`: 118
- `education`: 47
- `stem`: 23
- `science`: 5
- `web`: 1
- `commercial`: 1

`label_method`:

- `hybrid_rule_fallback_low_confidence`: 186
- `hybrid_rule_source_type_minilm_domain`: 9

## Low-confidence counts

| Dataset | Lexical low-confidence | MiniLM low-confidence |
| --- | ---: | ---: |
| FineWeb | 100 / 113 | 112 / 113 |
| FineWeb-Edu | 76 / 88 | 85 / 88 |
| FineMath | 155 / 195 | 186 / 195 |

The high low-confidence rate means the current taxonomy descriptions and thresholds should be reviewed after pseudo-gold labeling. Do not tune thresholds from these unlabeled outputs alone.

## Review candidates

`data_samples/real_hf_benchmark_v1_review_candidates.jsonl` contains 120 records:

- FineWeb: 40
- FineWeb-Edu: 40
- FineMath: 40

Each record includes:

- hybrid prediction as the primary `*_pred` fields;
- `rule_based_label`;
- `lexical_label`;
- `embedding_label`;
- full chunk text;
- empty pseudo-gold fields.

All pseudo-gold fields are intentionally `null` at this stage:

- `expected_source_type`
- `expected_domain`
- `expected_field`
- `expected_subfield`
- `review_note`
- `review_confidence`

## Warnings and blockers

- HF emitted unauthenticated/retry warnings during streaming, but all three samples completed.
- The earlier one-dataset smoke output used a doubled `real_hf_real_hf_*` prefix. It is historical and was left untouched.
- The benchmark v1 runner now produces clean `real_hf_benchmark_v1_*` names.
- MiniLM confidence once exceeded the validation schema range by a tiny numeric margin; the embedding script now clamps cosine-derived confidence into `[0, 1]` before writing JSONL.
- Hybrid currently falls back to rule-based labels for most chunks because MiniLM confidence is low on many real chunks.
- These files are ready for pseudo-gold review, not for final claims about classifier quality.

## Suitability for next step

The benchmark is suitable for the next manual/Codex pseudo-gold labeling step:

1. Review `data_samples/real_hf_benchmark_v1_review_candidates.jsonl`.
2. Fill pseudo-gold `expected_*` fields according to `docs/real_hf_review_labeling_guide.md`.
3. Evaluate rule-based, lexical, MiniLM, and hybrid outputs against the reviewed labels.
4. Only then decide whether to adjust taxonomy descriptions, confidence thresholds, or hybrid fallback rules.
