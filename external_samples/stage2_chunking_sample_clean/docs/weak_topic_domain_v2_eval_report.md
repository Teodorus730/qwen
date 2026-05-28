# Weak topic.domain v2 evaluation report

## Setup

This report evaluates a weak coarse `topic.domain` v2 baseline for annotation schema v2.

Inputs:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_features_v2_tokenized.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_features_v2_tokenized.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_features_v2_tokenized.jsonl`
- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold.jsonl`

Outputs:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_topic_domain_v2.jsonl`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_fineweb.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_fineweb_edu.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_finemath.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_eval_summary.json`

Method:

- `weak_topic_domain_v2_keyword_surface_prior`

No HF streaming, MiniLM, AutoModel, model weights, NLL/logits, taxonomy changes, or old classifier changes were used.

## Allowed domains

- `stem`
- `science`
- `technology`
- `software`
- `humanities`
- `social_sciences`
- `commercial`
- `government`
- `media`
- `reference`
- `education`
- `unknown`

## Method summary

The classifier uses:

- lexical keyword profiles per domain;
- annotation_v2 surface features;
- weak provenance priors;
- abstention when evidence is weak or ambiguous.

Surface features support but do not force topic:

- math/symbol-heavy features support `stem`;
- code/API syntax supports `software`;
- scientific formulas support `science` and weakly `stem`.

Provenance priors are intentionally weak:

- FineMath weakly supports `stem`;
- FineWeb-Edu weakly supports `education`;
- FineWeb does not force a topic.

The confidence score is a normalized heuristic score, not a calibrated probability.

## Overall metrics

Evaluation uses the 120-record annotation_v2 pseudo-gold set.

| Metric | Value |
| --- | ---: |
| records_total | 120 |
| answered_count | 96 |
| abstained_count | 24 |
| coverage_rate | 0.8000 |
| accuracy_on_answered | 0.6979 |
| accuracy_counting_abstain_wrong | 0.5583 |
| gold_abstained_count | 3 |

Interpretation:

- The baseline answers most records.
- On answered records it is a usable transparent baseline, not a strong classifier.
- Abstention is meaningful and should remain part of the MVP evaluation.

## Per-dataset metrics

| Dataset | Records | Coverage | Accuracy on answered | Accuracy counting abstain wrong |
| --- | ---: | ---: | ---: | ---: |
| FineWeb | 40 | 0.8500 | 0.7059 | 0.6000 |
| FineWeb-Edu | 40 | 0.7750 | 0.6774 | 0.5250 |
| FineMath | 40 | 0.7750 | 0.7097 | 0.5500 |

The baseline is fairly balanced across the three datasets. FineWeb has slightly higher coverage; FineWeb-Edu has the lowest accuracy due to mixed educational/reference/science/humanities content.

## Per-domain metrics

| Gold domain | Records | Coverage | Accuracy on answered |
| --- | ---: | ---: | ---: |
| stem | 23 | 0.6957 | 1.0000 |
| technology | 24 | 0.9167 | 0.7727 |
| science | 31 | 0.8387 | 0.5769 |
| education | 11 | 0.7273 | 0.6250 |
| humanities | 5 | 1.0000 | 0.8000 |
| software | 5 | 1.0000 | 0.6000 |
| media | 7 | 0.5714 | 0.7500 |
| reference | 6 | 0.6667 | 0.2500 |
| commercial | 2 | 0.5000 | 0.0000 |
| social_sciences | 2 | 1.0000 | 1.0000 |
| government | 1 | 1.0000 | 1.0000 |
| unknown | 3 | 0.6667 | 0.0000 |

Small domains should not be overinterpreted. The clearest reliable domain is `stem` when the classifier answers. The weakest useful domain is `reference`.

## Main confusion groups

Largest or most meaningful confusions:

- `technology -> commercial`: 5 records, mostly POS/patent chunks with rental, sale, invoice, customer, and transaction vocabulary.
- `stem -> ABSTAIN`: 7 records, usually math/statistics text with weak lexical evidence beyond the FineMath prior.
- `science -> ABSTAIN`: 5 records.
- `science -> stem`: 4 records, often physics/measurement formula chunks.
- `science -> social_sciences`: 3 records, especially climate/disaster/community text.
- `reference -> media/humanities/software/ABSTAIN`: reference remains poorly separated with simple keywords.
- `software -> stem`: 2 records, mostly math-tool/code-like material inside FineMath.

## Where it works

The baseline works reasonably well for:

- formula-heavy or explicit math/STEM records;
- patent/technology records when technology vocabulary dominates commercial vocabulary;
- obvious humanities/social/government records;
- many FineMath records when math surface features are visible.

The evidence output is useful for debugging because it shows exactly which keywords and surface features triggered the prediction.

## Where it fails

The baseline fails or abstains in predictable places:

- POS patent text mixes technology and commercial language.
- Science education text often mixes science, education, and social science.
- Reference pages are hard to identify without a stronger genre/function signal.
- FineMath contains physics lessons and software/math-tool examples that are not purely `stem`.
- Some gold-abstained noisy records still contain keywords that trigger a domain.

## Is this enough for MVP domain grouping?

Yes, as a transparent MVP baseline.

It is not enough as a final topic classifier. But it is good enough to:

- establish a dependency-free lower bound;
- test the annotation_v2 evaluation policy;
- support early NLL/profiling grouping with confidence and abstention;
- identify failure clusters before using embeddings.

For downstream use, prefer:

```text
use answered high-confidence records for grouping;
keep abstained records separate;
do not treat weak topic.domain as ground truth.
```

## Recommended next steps

1. Manually inspect failure clusters, especially:
   - `technology -> commercial`;
   - `science -> social_sciences`;
   - `reference` failures;
   - gold-abstained records that received predictions.
2. Add a small report or review file for those clusters before changing thresholds.
3. Later, compare this baseline with an embedding-based topic classifier.
4. Later, test on a held-out benchmark v2-test.

Do not tune this baseline aggressively on benchmark v1. It is a dev set, not final evaluation.
