# BGE-M3 semantic topic error analysis and v1.1 label-card report

## Purpose

This report summarizes dev-only error analysis and label-card refinement for the cleaned target `semantic_topic_domain_v1`.

No v2-test data was used. No new models were downloaded. No MiniLM, NLL/logits, HF streaming, feature extraction, tokenization stats, old mixed `topic.domain` classifier, or old pseudo-gold files were changed.

## Inputs

Gold/dev target:

- `data_samples/real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl`

Original BGE-M3 predictions:

- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3.jsonl`
- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_eval.json`

Original label cards:

- `taxonomy/semantic_topic_domain_v1_descriptions.json`

New contrastive label cards:

- `taxonomy/semantic_topic_domain_v1_1_descriptions.json`

Model:

```text
C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181
```

## Original BGE-M3 metrics

Gold-abstained records are excluded by the evaluator.

| Metric | Value |
| --- | ---: |
| records_total | 118 |
| coverage_rate | 1.000000 |
| accuracy_on_answered | 0.381356 |
| strict_accuracy | 0.381356 |
| top_k_contains_gold | 0.796610 |
| macro_f1 | 0.294358 |

Per dataset:

| Dataset | Accuracy |
| --- | ---: |
| FineMath | 0.552632 |
| FineWeb | 0.225000 |
| FineWeb-Edu | 0.375000 |

Original top confusions:

| Count | Gold | Predicted |
| ---: | --- | --- |
| 18 | `engineering_technology` | `computer_science_software` |
| 8 | `natural_science` | `math` |
| 3 | `math` | `other_unclear` |
| 3 | `lifestyle_everyday` | `other_unclear` |
| 3 | `humanities_arts` | `lifestyle_everyday` |
| 3 | `computer_science_software` | `other_unclear` |

## Error analysis summary

Created:

- `scripts/analyze_semantic_topic_embedding_errors.py`
- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_error_analysis.json`
- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_error_examples.jsonl`

Key findings:

- top-1 accuracy was weak, but top-k signal was useful;
- gold was present in top-k for 94 of 118 evaluated records;
- 49 records had gold in top-k while top-1 was wrong;
- the largest error cluster was engineering/patent/device text predicted as software;
- formula-heavy science was often pulled toward math;
- humanities/social/policy/lifestyle boundaries remained soft for essay-like web content.

This supported a label-card refinement step rather than abandoning BGE-M3.

## Why v1.1 label cards were created

`semantic_topic_domain_v1_1_descriptions.json` keeps the same domain labels but makes descriptions more contrastive.

Main improvements:

- `engineering_technology` now explicitly includes patents, devices, circuits, terminals, sensors, adapters, physical systems, electronics, infrastructure, industrial systems, and POS hardware;
- `computer_science_software` now explicitly excludes physical devices and patent apparatus unless software is the main subject;
- `natural_science` now explicitly includes biology, chemistry, physics, ecology, climate, atoms, molecules, species, and forces;
- `math` now explicitly excludes physics/chemistry/biology text that merely contains formulas;
- `humanities_arts`, `lifestyle_everyday`, and `social_sciences` now include stronger contrastive negative notes;
- `business_economics` now explicitly separates business subject matter from `product_commercial` genre/function.

## v1 vs v1.1 metrics

| Run | Coverage | Accuracy on answered | Strict accuracy | Top-k contains gold | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 + v1 cards | 1.000000 | 0.381356 | 0.381356 | 0.796610 | 0.294358 |
| BGE-M3 + v1.1 cards | 1.000000 | 0.567797 | 0.567797 | 0.864407 | 0.435499 |
| BGE-M3 + v1.1 cards + deterministic top-k rerank | 1.000000 | 0.635593 | 0.635593 | 0.864407 | 0.459891 |

Per-dataset accuracy:

| Run | FineMath | FineWeb | FineWeb-Edu |
| --- | ---: | ---: | ---: |
| v1 cards | 0.552632 | 0.225000 | 0.375000 |
| v1.1 cards | 0.552632 | 0.575000 | 0.575000 |
| v1.1 + rerank | 0.657895 | 0.650000 | 0.600000 |

The largest gain came from clarifying engineering vs software and removing genre-like ambiguity from domain descriptions.

## v1.1 remaining confusions

Top v1.1 label-card confusions:

| Count | Gold | Predicted |
| ---: | --- | --- |
| 7 | `natural_science` | `math` |
| 4 | `engineering_technology` | `computer_science_software` |
| 3 | `natural_science` | `social_sciences` |
| 3 | `natural_science` | `other_unclear` |
| 3 | `math` | `other_unclear` |
| 2 | `medicine_health` | `lifestyle_everyday` |
| 2 | `lifestyle_everyday` | `other_unclear` |
| 2 | `humanities_arts` | `lifestyle_everyday` |
| 2 | `humanities_arts` | `law_government_policy` |
| 2 | `engineering_technology` | `math` |

The engineering/software confusion dropped from 18 to 4, which is the clearest improvement.

Natural science vs math remains the largest boundary problem after v1.1.

## Reranking policy

Created:

- `scripts/rerank_semantic_topic_embedding_topk.py`
- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_reranked.jsonl`
- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_reranked_eval.json`

The reranker uses saved top-k results and deterministic text guards only. It does not use gold labels and does not run new embedding inference.

Implemented guards:

- engineering vs software:
  - prefer `engineering_technology` when patent/device/apparatus/circuit/terminal/sensor/adapter/POS/physical-system terms are present;
  - prefer `computer_science_software` when software/API/code terms dominate without physical-system terms.
- natural science vs math:
  - prefer `natural_science` when biology/chemistry/physics/entity terms appear;
  - prefer `math` when theorem/proof/topology/equation/statistics terms dominate.

Reranked top confusions:

| Count | Gold | Predicted |
| ---: | --- | --- |
| 3 | `natural_science` | `social_sciences` |
| 3 | `natural_science` | `other_unclear` |
| 3 | `math` | `other_unclear` |
| 2 | `natural_science` | `math` |
| 2 | `medicine_health` | `lifestyle_everyday` |
| 2 | `lifestyle_everyday` | `other_unclear` |
| 2 | `humanities_arts` | `lifestyle_everyday` |
| 2 | `humanities_arts` | `law_government_policy` |
| 2 | `computer_science_software` | `math` |

The reranker improved dev accuracy, but it is a hand-built dev policy and should be treated as overfit-risk until tested on a cleaned held-out split.

## Threshold/margin sweep

Created:

- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards_threshold_sweep.json`

Best strict accuracy in the small sweep:

| threshold | margin_threshold | coverage | accuracy_on_answered | strict_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0.0 | 0.991525 | 0.572650 | 0.567797 |

Best accuracy on answered:

| threshold | margin_threshold | coverage | accuracy_on_answered | strict_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.05 | 0.169492 | 0.850000 | 0.144068 |

Interpretation:

- score thresholding did not materially beat raw v1.1 top-1 strict accuracy;
- margin filtering can increase precision but loses too much coverage for a default classifier;
- reranking was more useful than blunt margin abstention on this dev set.

## Dev overfit risk

The improvements are real on v1-dev, but they are not held-out evidence.

Risk factors:

- v1.1 cards were written after inspecting v1-dev error clusters;
- reranker guards target known v1-dev failure modes;
- v2-test has not been relabeled under cleaned axes and was not used;
- some domains have very low support, especially `business_economics`, `medicine_health`, and `social_sciences`.

This means the v1.1 result is best interpreted as protocol development, not final model quality.

## Should BGE-M3 continue as a candidate?

Yes.

Reasons:

- local no-download run works;
- top-k semantic signal is strong enough to support reranking;
- contrastive label-card changes improved top-1 substantially;
- remaining errors are interpretable and concentrated in boundary cases.

BGE-M3 should remain a candidate for the semantic-topic bake-off, but it needs a cleaned held-out evaluation before any quality claim.

## Recommended next step

1. Freeze a dev protocol candidate:
   - v1.1 label cards;
   - deterministic top-k reranker as an experimental policy;
   - explicit reporting of coverage, top-k hit rate, and confusion matrix.

2. Decide held-out strategy:
   - relabel v2-test under cleaned axes with an audit trail; or
   - create a fresh v3 held-out split.

3. Compare another embedding model only after approval/download:
   - MiniLM after restoring/downloading full snapshot;
   - E5 or Qwen embedding candidates after explicit approval.

Do not use these v1-dev results as held-out quality.
