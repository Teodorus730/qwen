# Annotation schema v2 pseudo-gold report

## Purpose

This report documents `real_hf_benchmark_v1_annotation_v2_pseudo_gold.jsonl`, a reviewed pseudo-gold reference for annotation schema v2.

The file covers the same 120 `real_hf_benchmark_v1` review candidates:

- 40 FineWeb chunks;
- 40 FineWeb-Edu chunks;
- 40 FineMath chunks.

The goal is to evaluate future annotation schema v2 components:

- deterministic surface features;
- quality/noise labels;
- future weak `topic.domain` classifier;
- confidence and abstention policy.

This is not HF streaming, MiniLM inference, NLL scoring, or model training.

## Why new pseudo-gold was needed

The old pseudo-gold used a strict tuple:

```text
expected_source_type + expected_domain + expected_field + expected_subfield
```

That was useful for discovering that `source_type` was overloaded, but it is not the right primary target for annotation schema v2.

Annotation schema v2 treats labels as independent axes:

- topic is a coarse semantic grouping;
- surface features describe visible format;
- quality/noise describes residue and usability;
- provenance and token/text stats are metadata/features, not labels.

This avoids making `full_label_accuracy` the main metric.

## Files

Pseudo-gold:

- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold.jsonl`

Summary:

- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold_summary.json`

Guidelines:

- `docs/annotation_schema_v2_pseudo_gold_guidelines.md`

Summary script:

- `scripts/summarize_annotation_v2_pseudo_gold.py`

## Integrity

| Check | Result |
| --- | ---: |
| Records | 120 |
| Duplicate `chunk_id` | 0 |
| Required review fields missing | 0 |
| Invalid topic domains | 0 |
| Invalid noise levels | 0 |
| Invalid confidence values | 0 |

## Dataset distribution

| Dataset | Records |
| --- | ---: |
| FineWeb | 40 |
| FineWeb-Edu | 40 |
| FineMath | 40 |

## Topic domain distribution

| Domain | Records |
| --- | ---: |
| science | 31 |
| technology | 24 |
| stem | 23 |
| education | 11 |
| media | 7 |
| reference | 6 |
| software | 5 |
| humanities | 5 |
| unknown | 3 |
| social_sciences | 2 |
| commercial | 2 |
| government | 1 |

Abstention:

- abstained records: 3 / 120
- abstention rate: 0.025

The low abstention rate reflects the fact that most benchmark records have a usable coarse topic even when field/subfield would be uncertain. Abstentions are reserved for mostly noisy or unclear chunks.

## Surface distributions

| Field | Count | Share |
| --- | ---: | ---: |
| `review_has_math_notation` | 32 | 0.2667 |
| `review_has_code` | 28 | 0.2333 |
| `review_is_symbol_heavy` | 11 | 0.0917 |
| `review_has_scientific_formula` | 16 | 0.1333 |
| `review_has_api_or_command_syntax` | 27 | 0.2250 |

The surface axes are intentionally separate from topic. For example, a science or technology chunk may contain API-style or formula-like surface syntax without becoming a `software` or `stem` topic.

## Quality distributions

| Noise level | Records |
| --- | ---: |
| clean | 77 |
| partial_noise | 41 |
| mostly_noise | 2 |
| unknown | 0 |

Residue flags:

| Field | Count | Share |
| --- | ---: | ---: |
| `review_has_ui_residue` | 46 | 0.3833 |
| `review_has_forum_residue` | 4 | 0.0333 |

Many `partial_noise` records still contain useful primary text. They should not be excluded automatically from future analysis, but they should be separable in NLL/profiling.

## Confidence distribution

Overall review confidence:

| Confidence | Records |
| --- | ---: |
| 1.0 | 18 |
| 0.8 | 84 |
| 0.6 | 15 |
| 0.2 | 3 |

Topic confidence:

| Confidence | Records |
| --- | ---: |
| 1.0 | 20 |
| 0.8 | 82 |
| 0.6 | 15 |
| 0.2 | 3 |

## Ambiguous cases

Main ambiguity groups:

- Mostly noisy FineMath-like text where topic is not reliable.
- Science vs education when educational prose explains science content.
- Technology/patent chunks where genre/function is patent-like but topic is technology.
- Reference vs humanities/social science for article-style historical or political text.
- API/help/math tool syntax where `has_code` and `has_math_notation` can both be true.

These ambiguities are expected. The point of schema v2 is to make them visible instead of forcing one overloaded `source_type`.

## What this enables next

The new pseudo-gold enables:

- broader evaluation of deterministic surface features;
- evaluation of `noise_level` and residue flags;
- implementation of weak `topic.domain` v2;
- reporting `accuracy_on_answered` together with coverage/abstention rate;
- comparing topic predictions without using old full-label accuracy as the primary KPI.

Recommended evaluation for future weak topic classifier:

```text
coverage = answered / total
accuracy_on_answered = correct_answered / answered
abstention_rate = abstained / total
confusion_matrix over review_topic_domain
```

## Limitations

- This is pseudo-gold, not independent human gold.
- It is a dev benchmark, not a held-out test.
- Review labels are intended for coarse schema v2 evaluation, not final taxonomy tuning.
- Field/subfield from old pseudo-gold are preserved only for traceability.
- No NLL/logprob/perplexity was computed.
- No model weights were used.

## Recommendation

This pseudo-gold is sufficient for the next implementation step: weak `topic.domain` v2 with confidence and abstention.

Do not implement field/subfield prediction as the primary target yet. The next classifier should first prove that it can produce useful coarse topic labels with honest abstention.
