# real_hf_benchmark_v2_test topic.domain evaluation

## Purpose

`real_hf_benchmark_v2_test` is the first held-out check for
`weak_topic_domain_v2.1`. The earlier `real_hf_benchmark_v1` split is a dev
benchmark because it was used for annotation_v2 pseudo-gold creation, manual
review, patched gold, error analysis, and v2.1 rule selection.

The results below are test results. They should be reported, not used for
immediate tuning.

## Pseudo-gold

Held-out pseudo-gold file:

- `data_samples/real_hf_benchmark_v2_test_annotation_v2_pseudo_gold.jsonl`

Records:

- total: 90
- FineWeb: 30
- FineWeb-Edu: 30
- FineMath: 30

Topic domain distribution:

| Domain | Count |
| --- | ---: |
| stem | 22 |
| science | 18 |
| humanities | 17 |
| commercial | 12 |
| unknown | 6 |
| software | 3 |
| government | 3 |
| technology | 3 |
| media | 3 |
| education | 2 |
| social_sciences | 1 |

Gold abstention:

- 6 / 90 = 0.0667

Surface/quality notes:

- `review_has_math_notation`: 25 / 90
- `review_has_code`: 2 / 90
- `review_is_symbol_heavy`: 11 / 90
- `review_has_scientific_formula`: 5 / 90
- `review_noise_level`: clean 73, partial_noise 11, mostly_noise 6

## Held-out metrics

Predictions evaluated:

- `data_samples/real_samples/real_hf_benchmark_v2_test_fineweb_topic_domain_v2_1.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_fineweb_edu_topic_domain_v2_1.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_finemath_topic_domain_v2_1.jsonl`

All 90 pseudo-gold records were matched by `chunk_id`.

| Split | Coverage | Accuracy on answered | Strict accuracy |
| --- | ---: | ---: | ---: |
| v1 patched dev | 0.9083 | 0.8165 | 0.7417 |
| v2-test held-out | 0.7889 | 0.6197 | 0.4889 |

Per-dataset v2-test metrics:

| Dataset | Coverage | Accuracy on answered | Strict accuracy |
| --- | ---: | ---: | ---: |
| FineWeb | 0.7333 | 0.3182 | 0.2333 |
| FineWeb-Edu | 0.8667 | 0.7692 | 0.6667 |
| FineMath | 0.7667 | 0.7391 | 0.5667 |

## Main confusions

The largest held-out failures are:

- `unknown -> social_sciences`: adult/noisy spam chunks triggered shallow `women` evidence instead of abstaining.
- `commercial -> ABSTAIN`: commercial product/service pages often lacked explicit commercial keywords or had competing art/science signals.
- `science -> ABSTAIN` and `science -> stem`: science passages with measurements/statistics were often treated as math-like rather than scientific.
- `software -> stem`: code/matrix snippets from FineMath were pulled toward `stem` by matrix/function/provenance evidence.
- `technology -> stem` or `technology -> ABSTAIN`: electronics/smart-city content did not have enough robust technology vocabulary.
- `media -> government/humanities/ABSTAIN`: media/review/lifestyle chunks remain weakly represented in the keyword baseline.

FineWeb is the hardest split. It contains commercial pages, legal terms,
product listings, adult spam, rescue pages, lifestyle/interview content, and
media blurbs. Keyword-only classification is brittle on this mixture.

## Interpretation

The held-out result is meaningfully degraded from v1 dev:

- coverage dropped from 0.9083 to 0.7889;
- accuracy on answered dropped from 0.8165 to 0.6197;
- strict accuracy dropped from 0.7417 to 0.4889.

This suggests that `weak_topic_domain_v2.1` is useful as a transparent MVP
baseline, but it does not generalize strongly enough to be treated as a final
topic labeler.

The baseline is strongest on:

- FineMath stem/math chunks;
- FineWeb-Edu science/humanities chunks;
- obvious education cases.

It is weakest on:

- noisy web spam;
- commercial/legal/service pages;
- technology vs stem boundaries;
- science vs stem boundaries;
- media/reference/function-like content.

## Decision

Keep `weak_topic_domain_v2.1` as the current MVP baseline for annotation_v2
because it provides interpretable coverage and a useful abstention mechanism.
Do not keep tuning it on v2-test.

The next appropriate steps are:

1. label a larger or more diverse held-out sample only if higher confidence is
   needed;
2. run an embedding-model bake-off for `topic.domain` later, using this v2-test
   only as a reported held-out check;
3. use `accuracy_on_answered + coverage` as primary metrics, not full-label
   accuracy.

## No-tuning warning

Do not modify `weak_topic_domain_v2.1` based on this report in the same test
cycle. Any classifier changes after reading v2-test results should be evaluated
on a later v3/final holdout.
