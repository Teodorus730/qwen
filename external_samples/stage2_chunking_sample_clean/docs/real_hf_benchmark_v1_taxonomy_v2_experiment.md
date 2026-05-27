# Real HF benchmark v1 taxonomy v2 experiment

## Purpose

The v1 real HF benchmark showed very low metrics even after pseudo-gold audit. The audit did not find a critical evaluator bug, but it did show that many pseudo-gold labels were forced into a tiny taxonomy.

This experiment tests taxonomy coverage only:

- add a small number of frequent missing labels;
- create `pseudo_gold_v2` from the same 120 reviewed records;
- rerun lexical, MiniLM, and hybrid on the same benchmark chunks;
- compare v1 and v2 without changing thresholds, classifiers, chunking, or scoring.

This is a dev benchmark experiment, not a final test-set result.

## Labels added

Nine labels were added to `taxonomy/simple_domain_labels.json`:

| Domain | Field | Subfield | Reason |
| --- | --- | --- | --- |
| `technology` | `patents` | `POS_systems` | Covers the largest gap: patent-style POS/rental/transaction systems. |
| `technology` | `patents` | `hardware_networking` | Covers hardware/networking patent-style chunks. |
| `stem` | `mathematics` | `statistics` | Covers statistics, forecasting, probability, and model-evaluation chunks. |
| `stem` | `mathematics` | `arithmetic_measurement` | Covers inches/feet/yards and basic measurement conversions. |
| `science` | `chemistry` | `article` | Covers chemistry tutorials in FineMath. |
| `stem` | `mathematics` | `geometry` | Covers geometry/diagram/shape math chunks. |
| `stem` | `mathematics` | `topology` | Covers advanced topology/analysis chunks. |
| `humanities` | `art_history` | `article` | Covers art history and artist reference chunks. |
| `social_sciences` | `social_issues` | `article` | Covers gender/social-issue/advocacy chunks. |

Labels not added in this experiment: automotive, food, medicine, sports/coaching, personal essays, religion, and other one-off gaps. They remain candidates for a later taxonomy pass.

## Pseudo-gold v2

Created:

- `data_samples/real_hf_benchmark_v1_pseudo_gold_v2.jsonl`

Source:

- `data_samples/real_hf_benchmark_v1_pseudo_gold.jsonl`

Records changed: 49 / 120.

Changes by original gap:

- `patents/POS_systems`: 17
- `statistics`: 5
- `patents/hardware_networking`: 4
- `arithmetic/measurement`: 4
- `chemistry`: 3
- `art_history`: 2
- `probability/statistics`: 2
- `geometry`: 2
- `topology`: 2
- plus one-record variants mapped to the new labels: `analysis/topology`, `art_history/political_history`, `art_history/religion`, `gender_studies`, `history/art_history`, `patents/computer_vision`, `social_issues/gender`, `statistics/linear_algebra/finance`.

All v2 expected domain/field/subfield tuples are represented in the updated taxonomy.

## Outputs

Lexical v2:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_labeled_lexical_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_labeled_lexical_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_labeled_lexical_v2.jsonl`

MiniLM v2:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_labeled_embedding_minilm_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_labeled_embedding_minilm_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_labeled_embedding_minilm_v2.jsonl`

Hybrid v2:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_labeled_hybrid_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_labeled_hybrid_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_labeled_hybrid_v2.jsonl`

Evaluation outputs:

- `data_samples/real_hf_benchmark_v1_eval_rule_based_v2.json`
- `data_samples/real_hf_benchmark_v1_eval_lexical_v2.json`
- `data_samples/real_hf_benchmark_v1_eval_minilm_v2.json`
- `data_samples/real_hf_benchmark_v1_eval_hybrid_v2.json`

## V1 metrics

| Classifier | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule-based | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.1000 |
| lexical | 0.0833 | 0.0750 | 0.0667 | 0.1167 | 0.0583 |
| MiniLM | 0.0000 | 0.0000 | 0.0000 | 0.0250 | 0.0000 |
| hybrid | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.1000 |

## V2 metrics

| Classifier | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule-based | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.0750 |
| lexical v2 | 0.0750 | 0.3667 | 0.3667 | 0.3333 | 0.0583 |
| MiniLM v2 | 0.0000 | 0.1333 | 0.1333 | 0.1333 | 0.0000 |
| hybrid v2 | 0.1833 | 0.2667 | 0.2667 | 0.3250 | 0.0917 |

## Changes

Lexical improved strongly on semantic labels:

- domain: 0.0750 -> 0.3667
- field: 0.0667 -> 0.3667
- subfield: 0.1167 -> 0.3333

MiniLM improved from zero semantic accuracy:

- domain: 0.0000 -> 0.1333
- field: 0.0000 -> 0.1333
- subfield: 0.0250 -> 0.1333

Hybrid became meaningfully different from rule-based:

- domain: 0.1417 -> 0.2667
- field: 0.1417 -> 0.2667
- subfield: 0.2333 -> 0.3250

Full-label accuracy did not improve because it is dominated by strict `source_type` matching. MiniLM does not predict source_type, lexical mostly leaves source_type as `unknown`, and hybrid preserves rule-based source_type.

## Low-confidence counts

| Classifier | V1 low-confidence | V2 low-confidence |
| --- | ---: | ---: |
| lexical | 331 / 396 | 212 / 396 |
| MiniLM | 383 / 396 | 320 / 396 |

V2 taxonomy coverage substantially reduced low-confidence outputs, especially for lexical baseline.

## Hybrid contribution

Hybrid method counts:

| Version | Rule fallback | MiniLM semantic label used |
| --- | ---: | ---: |
| v1 | 383 | 13 |
| v2 | 320 | 76 |

Hybrid v2 uses MiniLM semantic labels much more often, but still depends on rule-based source_type and still falls back for most chunks.

## Remaining taxonomy gaps

Remaining frequent or notable gaps:

- sports/coaching
- spam/low_quality_math
- automotive
- food
- medicine/psychology
- medicine/clinical_trials
- medicine/genetics
- psychology/animal_behavior
- history / international relations / religion
- personal essays
- computer science algorithms
- public health
- business/marketing

These were not added in v2 to keep the experiment controlled.

## Interpretation

Taxonomy coverage clearly matters. The v2 labels improved semantic classification for lexical, MiniLM, and hybrid without changing thresholds or classifier logic.

The strongest result is lexical semantic improvement. MiniLM also improves, but remains weak under the current threshold and label-text setup. Hybrid is now better than rule-based on domain/field/subfield, but not on full-label accuracy.

The v2 experiment supports the idea that the previous real benchmark was partly coverage-limited. It does not prove that MiniLM is ready, and it does not justify threshold tuning yet.

## Caveats

- Taxonomy v2 was designed from the same 120-record dev benchmark, so improvements are not final generalization evidence.
- Pseudo-gold v2 is still pseudo-gold and should get human review before final reporting.
- Full-label accuracy remains a harsh metric because source_type and semantic topic are different tasks.
- Rule-based was not updated, so it cannot benefit from new semantic labels.

## Next steps

1. Human-review pseudo-gold v2 changes.
2. Decide whether source_type should be evaluated separately from domain/field/subfield in headline metrics.
3. Inspect MiniLM top-k outputs for v2 mismatches.
4. Only after this, consider threshold or label-description tuning.
5. Keep the benchmark sample fixed for any next comparison.
