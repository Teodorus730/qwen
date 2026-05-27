# Real HF benchmark v1 evaluation

## Caveat

This evaluation uses `data_samples/real_hf_benchmark_v1_pseudo_gold.jsonl`.

The labels are pseudo-gold review labels produced from real chunk text. They are useful for checking whether the current stage2 classifiers behave plausibly on real data, but they are not final human-adjudicated ground truth.

No classifier, threshold, taxonomy, or hybrid rule was tuned during this evaluation.

## Files evaluated

Pseudo-gold:

- `data_samples/real_hf_benchmark_v1_pseudo_gold.jsonl`

Predictions:

- rule-based: `data_samples/real_samples/real_hf_benchmark_v1_*_labeled_rule_based.jsonl`
- lexical: `data_samples/real_samples/real_hf_benchmark_v1_*_labeled_lexical.jsonl`
- MiniLM: `data_samples/real_samples/real_hf_benchmark_v1_*_labeled_embedding_minilm.jsonl`
- hybrid: `data_samples/real_samples/real_hf_benchmark_v1_*_labeled_hybrid.jsonl`

Evaluation script:

- `scripts/evaluate_predictions_against_pseudo_gold.py`

## Overall metrics

| Classifier | Source type accuracy | Domain accuracy | Field accuracy | Subfield accuracy | Full label accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule-based | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.1000 |
| lexical | 0.0833 | 0.0750 | 0.0667 | 0.1167 | 0.0583 |
| MiniLM | 0.0000 | 0.0000 | 0.0000 | 0.0250 | 0.0000 |
| hybrid | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.1000 |

## Per-dataset metrics

### Rule-based

| Dataset | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 0.0500 | 0.0500 | 0.0500 | 0.1000 | 0.0500 |
| FineWeb-Edu | 0.2500 | 0.3000 | 0.3000 | 0.3500 | 0.1750 |
| FineMath | 0.2500 | 0.0750 | 0.0750 | 0.2500 | 0.0750 |

### Lexical

| Dataset | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 0.0750 | 0.0500 | 0.0500 | 0.0750 | 0.0500 |
| FineWeb-Edu | 0.0500 | 0.0750 | 0.0750 | 0.1000 | 0.0500 |
| FineMath | 0.1250 | 0.1000 | 0.0750 | 0.1750 | 0.0750 |

### MiniLM

| Dataset | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FineWeb-Edu | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FineMath | 0.0000 | 0.0000 | 0.0000 | 0.0750 | 0.0000 |

### Hybrid

| Dataset | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 0.0500 | 0.0500 | 0.0500 | 0.1000 | 0.0500 |
| FineWeb-Edu | 0.2500 | 0.3000 | 0.3000 | 0.3500 | 0.1750 |
| FineMath | 0.2500 | 0.0750 | 0.0750 | 0.2500 | 0.0750 |

## Interpretation

The current real-data results are much weaker than the synthetic benchmark. This is mostly because the real chunks contain many topics and formats that the tiny taxonomy and current rules do not cover well.

MiniLM does not currently help on this benchmark. In the generated MiniLM outputs, most real chunks were marked low-confidence and left with null semantic labels. Because the hybrid builder falls back to rule-based labels when MiniLM is low-confidence, hybrid is effectively identical to rule-based on the evaluated candidates.

The rule-based classifier is still the strongest current baseline in this run, but its absolute scores are low. It detects some educational/math/product-ish cases, especially in FineWeb-Edu and FineMath, while missing many patent-style, health, history, social science, chemistry, statistics, and general web cases.

The lexical baseline is weaker than rule-based here. It often remains low-confidence and null on real chunks.

## Hypothesis check

H1: rule-based is better for `source_type`.

Partially supported. Rule-based has the best source-type accuracy overall at 0.1833, tied by hybrid because hybrid mostly falls back to rule-based. The absolute score is low, so this is only a weak readiness signal.

H2: MiniLM is better for semantic `domain` / `field` / `subfield`.

Not supported by this run. MiniLM produced too many low-confidence null labels under the current threshold/taxonomy setup. This does not prove MiniLM is unsuitable; it shows the current MiniLM nearest-label configuration is not ready on real chunks.

H3: hybrid is better overall.

Not supported yet. Hybrid matched rule-based because MiniLM rarely contributed confident labels. The architecture remains reasonable, but it needs better taxonomy descriptions, threshold review, and/or label candidate design before it can improve over rule-based.

## Where hybrid helps

In this run, hybrid rarely used MiniLM labels. Its main benefit is safety: it avoided replacing rule-based labels with low-confidence MiniLM outputs.

## Where hybrid fails

Hybrid fails wherever rule-based fails and MiniLM is low-confidence. That includes many real categories not represented in the tiny taxonomy:

- patents and technical system descriptions;
- chemistry;
- statistics/probability;
- art history, politics, religion, and social issues;
- automotive, sports, food, and personal essays;
- health and medicine.

## What remains unproven

The benchmark does not yet prove the final value of MiniLM or hybrid labeling. It shows that the current tiny taxonomy and confidence setup are insufficient for real mixed web data.

Before making quality claims, the next stage should:

1. review pseudo-gold labels manually;
2. inspect MiniLM top-k outputs on mismatches;
3. improve taxonomy coverage or descriptions deliberately;
4. rerun MiniLM without changing the benchmark sample;
5. compare rule-based, lexical, MiniLM, and hybrid again.
