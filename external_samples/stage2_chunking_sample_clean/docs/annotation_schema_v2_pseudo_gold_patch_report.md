# Annotation schema v2 pseudo-gold patch report

## Purpose

This report documents a small patched version of the annotation_v2 pseudo-gold benchmark based on targeted manual review of weak `topic.domain` errors. The patch is done before implementing `weak_topic_domain_v2.1`, so that future classifier changes are evaluated against a cleaner reference.

No classifier, evaluator, taxonomy, feature extractor, tokenizer stats, HF sample, MiniLM run, or NLL step was changed.

## Inputs

Original pseudo-gold:

- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold.jsonl`

Manual review:

- `data_samples/real_hf_benchmark_v1_topic_domain_v2_manual_review_labeled.jsonl`

Existing unchanged weak-topic predictions:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_topic_domain_v2.jsonl`

Generated artifacts:

- `scripts/patch_annotation_v2_pseudo_gold_from_review.py`
- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold_patched.jsonl`
- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold_patch_summary.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_patched_fineweb.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_patched_fineweb_edu.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_patched_finemath.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_patched_summary.json`

## Patch Rules

The patch script only modifies records where:

- `recommended_action == "pseudo_gold_patch"`, or
- `manual_review_decision` is `revise_gold_domain` or `mark_gold_abstained`.

It does not modify:

- `keep_gold` records;
- `ambiguous_keep_for_embedding_bakeoff` records unless explicitly marked as `pseudo_gold_patch`;
- classifier predictions;
- original pseudo-gold file.

Every output record includes patch metadata:

- `pseudo_gold_patch_applied`
- `pseudo_gold_patch_reason`
- `pseudo_gold_patch_source = "weak_topic_domain_v2_manual_review"`

## Patch Summary

| Item | Count |
| --- | ---: |
| total records | 120 |
| patched records | 11 |
| unchanged records | 109 |
| revised domain records | 9 |
| newly abstained records | 2 |
| original gold abstained | 3 |
| patched gold abstained | 5 |

Domain changes:

| Change | Count |
| --- | ---: |
| `reference -> humanities` | 2 |
| `science -> social_sciences` | 2 |
| `reference -> unknown` | 1 |
| `education -> humanities` | 3 |
| `reference -> social_sciences` | 1 |
| `science -> unknown` | 1 |
| `education -> commercial` | 1 |

Records marked abstained:

- `FineWeb-Edu_000014_002`
- `FineWeb-Edu_000003_007`

## Topic Distribution Change

| Domain | Original | Patched |
| --- | ---: | ---: |
| science | 31 | 28 |
| technology | 24 | 24 |
| stem | 23 | 23 |
| education | 11 | 7 |
| media | 7 | 7 |
| reference | 6 | 2 |
| humanities | 5 | 10 |
| software | 5 | 5 |
| unknown | 3 | 5 |
| social_sciences | 2 | 5 |
| commercial | 2 | 3 |
| government | 1 | 1 |

The largest conceptual cleanup is that several old `reference` and `education` labels were moved to semantic topic domains (`humanities`, `social_sciences`, `commercial`) or abstained. This matches the annotation_v2 policy: `topic.domain` should prefer semantic grouping, while reference/education-as-function should not dominate unless it is the primary topic.

## Old vs Patched Evaluation

The same unchanged `weak_topic_domain_v2` predictions were evaluated against original and patched gold.

| Metric | Original gold | Patched gold | Change |
| --- | ---: | ---: | ---: |
| records_total | 120 | 120 | 0 |
| answered_count | 96 | 96 | 0 |
| abstained_count | 24 | 24 | 0 |
| coverage_rate | 0.8000 | 0.8000 | 0 |
| correct_answered | 67 | 72 | +5 |
| accuracy_on_answered | 0.6979 | 0.7500 | +0.0521 |
| accuracy_counting_abstain_wrong | 0.5583 | 0.6000 | +0.0417 |
| gold_abstained_count | 3 | 5 | +2 |

Per-dataset comparison:

| Dataset | Original coverage | Original answered accuracy | Patched coverage | Patched answered accuracy |
| --- | ---: | ---: | ---: | ---: |
| FineWeb | 0.8500 | 0.7059 | 0.8500 | 0.7647 |
| FineWeb-Edu | 0.7750 | 0.6774 | 0.7750 | 0.7742 |
| FineMath | 0.7750 | 0.7097 | 0.7750 | 0.7097 |

The patch mostly improves FineWeb and FineWeb-Edu because most reviewed pseudo-gold ambiguity was there. FineMath remains unchanged because the manual review mostly confirmed its current labels while identifying classifier rule gaps.

## Patched Confusion Highlights

Remaining important confusions after patch:

- `technology -> commercial`: 5
- `technology -> ABSTAIN`: 2
- `stem -> ABSTAIN`: 7
- `science -> stem`: 4
- `science -> education`: 2
- `software -> stem`: 2
- `media -> ABSTAIN`: 3
- `unknown -> reference/software`: 2 answered noisy cases

Resolved or reduced by patch:

- `reference` shrank from 6 to 2 records, making it less of a mixed genre/function bucket.
- `science -> social_sciences` dropped from 3 to 1.
- Some old `education` errors now correctly count as `commercial` or `humanities`.

## Does This Change v2.1 Priorities?

Yes, but in a clarifying way. The patched baseline shows that part of the apparent error was pseudo-gold ambiguity, not classifier weakness. However, several clear rule-fixable clusters remain.

Highest-priority safe v2.1 fixes:

1. **Patent/POS technology guard**
   - Still needed: `technology -> commercial` remains 5.
   - Add or protect technology evidence for POS/patent/system-flow language.

2. **Broad stem/math/statistics keywords**
   - Still needed: `stem -> ABSTAIN` remains 7.
   - Add generic terms for math problems, measurements, MAPE/statistics, forecasting/model evaluation, topology/analysis.

3. **Formula-heavy science override**
   - Still needed: `science -> stem` remains 4.
   - Math notation should support surface characterization, but science terms/units/context should be able to win topic.

4. **Science vocabulary expansion**
   - Still needed for botany, animal behavior, force/physics, units, adhesion, mirror/radius examples.

5. **Noisy/single-keyword abstention**
   - Still needed: patched `unknown` has 2 answered noisy cases (`reference`, `software`).
   - Single `http` or `encyclopedia` evidence should not override noisy text quality.

Lower priority:

- Broadly improving `reference` recall. After patch, true `reference` is only 2 records and should remain narrow.
- Tuning education as a general bucket. Several old education labels were function-like rather than topic-like.

## Cases Remaining for Embedding Bake-Off

The following classes should not be overfit with keyword rules:

- climate/environment/community/policy mixed cases;
- educational function vs underlying science/humanities topic;
- media/memoir/history boundary cases;
- food/lifestyle/media chunks where the allowed domain set has no clean food domain;
- reference-like pages whose actual semantic topic is another domain.

These are better suited to a future topic-only embedding-model bake-off after v2.1 has fixed the obvious weak-rule gaps.

## Recommendation

The patched gold should become the dev reference for the next `weak_topic_domain_v2.1` experiment.

Recommended next implementation step:

1. Keep original pseudo-gold for provenance/history.
2. Use `real_hf_benchmark_v1_annotation_v2_pseudo_gold_patched.jsonl` for v2.1 evaluation.
3. Implement only conservative v2.1 changes:
   - patent/POS guard;
   - broad stem/science keyword additions;
   - formula-heavy science override;
   - noisy/single-keyword abstention.
4. Re-evaluate v2.0 vs v2.1 on patched gold.

Do not run embedding bake-off or held-out v2-test until after this small rule baseline is cleaned up.

