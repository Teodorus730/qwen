# Weak topic.domain v2.1 evaluation report

## Purpose

`weak_topic_domain_v2.1` is a conservative rule-baseline improvement for coarse `annotation_v2.topic.domain`. It was implemented only after:

- detailed v2 error analysis;
- targeted manual review of ambiguous cases;
- creation of patched annotation_v2 pseudo-gold.

The goal is not to build a final classifier. The goal is to make the transparent weak baseline less obviously wrong before future embedding-model comparison and held-out testing.

No HF streaming, MiniLM, AutoModel, NLL/logits, taxonomy edits, pseudo-gold edits, feature extractor edits, or tokenizer-stat edits were performed in this step.

## Implementation Summary

Updated script:

- `scripts/classify_topic_domain_v2.py`

Compatibility:

- `--version v2` preserves the original v2 behavior and remains the default.
- `--version v2_1` enables the new conservative fixes.

Method names:

- v2: `weak_topic_domain_v2_keyword_surface_prior`
- v2.1: `weak_topic_domain_v2_1_keyword_surface_prior`

Conservative v2.1 changes:

1. **Patent/POS technology guard**
   - Adds technology support when patent/POS/system/apparatus/embodiment/terminal/circuit/transaction/display context is present.
   - Caps commercial evidence only inside strong patent/POS context.
   - Does not globally suppress commercial product/service pages.

2. **Broader stem/math/statistics terms**
   - Adds generic terms such as ratio, percent, distribution, MAPE, forecast, measurement, formula, sequence, graph, topology, function, integral, problem, holdout, sigma, uniform norm, nowhere dense.
   - FineMath provenance remains weak and does not force a label alone.

3. **Formula-heavy science support**
   - Adds science support when science terms/units appear with math notation or symbol-heavy text.
   - Keeps math notation as a surface signal, not a hard `stem` decision.

4. **Broader science vocabulary**
   - Adds general botany, animal, physics, chemistry, unit, and habitat terms.

5. **Noisy/single-keyword abstention**
   - Abstains when noisy/residue-heavy text is supported only by one weak keyword such as `http`, `encyclopedia`, `article`, or `resource`.

Intentionally not changed:

- broad reference recall;
- education as generic function;
- climate/social-science boundary behavior;
- thresholds;
- allowed domains.

## Outputs

Generated v2.1 prediction files:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_topic_domain_v2_1.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_topic_domain_v2_1.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_topic_domain_v2_1.jsonl`

Generated patched-gold evaluations:

- `data_samples/real_hf_benchmark_v1_topic_domain_v2_1_eval_patched_fineweb.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_1_eval_patched_fineweb_edu.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_1_eval_patched_finemath.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_1_eval_patched_summary.json`

Diagnostic original-gold evaluation:

- `data_samples/real_hf_benchmark_v1_topic_domain_v2_1_eval_original_summary.json`

## Overall Metrics on Patched Gold

| Metric | v2.0 patched | v2.1 patched | Change |
| --- | ---: | ---: | ---: |
| records_total | 120 | 120 | 0 |
| answered_count | 96 | 109 | +13 |
| abstained_count | 24 | 11 | -13 |
| coverage_rate | 0.8000 | 0.9083 | +0.1083 |
| correct_answered | 72 | 89 | +17 |
| accuracy_on_answered | 0.7500 | 0.8165 | +0.0665 |
| accuracy_counting_abstain_wrong | 0.6000 | 0.7417 | +0.1417 |

The improvement comes from both higher coverage and more correct answers. This is exactly the intended role of v2.1: reduce obvious keyword/profile gaps without changing the benchmark target.

## Per-Dataset Metrics

| Dataset | v2.0 coverage | v2.0 answered accuracy | v2.0 strict accuracy | v2.1 coverage | v2.1 answered accuracy | v2.1 strict accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 0.8500 | 0.7647 | 0.6500 | 0.9000 | 0.9167 | 0.8250 |
| FineWeb-Edu | 0.7750 | 0.7742 | 0.6000 | 0.9000 | 0.7500 | 0.6750 |
| FineMath | 0.7750 | 0.7097 | 0.5500 | 0.9250 | 0.7838 | 0.7250 |

Interpretation:

- FineWeb improved strongly due to the patent/POS technology guard and media/news additions.
- FineMath improved strongly due to broader stem/math terms and better science support.
- FineWeb-Edu gained coverage and strict accuracy, but answered accuracy dipped slightly. That is a useful warning: some new science/media terms answer previously abstained ambiguous records incorrectly.

## Fixed Error Groups

Major improvements on patched gold:

| Error group | v2.0 patched | v2.1 patched | Change |
| --- | ---: | ---: | ---: |
| `technology -> commercial` | 5 | 1 | -4 |
| `technology -> ABSTAIN` | 2 | 1 | -1 |
| `stem -> ABSTAIN` | 7 | 2 | -5 |
| `science -> ABSTAIN` | 4 | 0 | -4 |
| `media -> ABSTAIN` | 3 | 2 | -1 |
| `unknown -> reference` | 1 | 0 | -1 |

These are exactly the targeted clusters from the manual review.

## Regressions and Cautions

New or worsened issues:

1. **Some science vocabulary is too broad**
   - `tree` in B-tree software text caused `software -> science`.
   - `leaves` and `energy` caused a reflective humanities passage to become `science`.
   - This suggests v2.1 science additions should be context-aware if tuned further.

2. **Formula-heavy science still sometimes becomes stem**
   - `science -> stem` dropped from 4 to 3, but did not disappear.
   - Physics/mirror/force/gecko examples still expose the surface-vs-topic tension.

3. **Education/media boundary remains fragile**
   - Some educational civic/current-events text became `media`.
   - This was intentionally not aggressively tuned in v2.1.

4. **Software vs stem remains hard**
   - Maple/matrix/B-tree examples are still often pulled to `stem`.
   - This may require a specific software/docs guard later, or embedding comparison.

5. **One noisy unknown still answers**
   - `unknown -> software` remains for an `http`/spam-like FineMath chunk.
   - The noisy/single-keyword policy helped one case but not all noisy cases.

## Original-Gold Diagnostic

v2.1 was also evaluated against the original unpatched pseudo-gold for diagnostics only:

| Metric | v2.1 original gold |
| --- | ---: |
| coverage_rate | 0.9083 |
| accuracy_on_answered | 0.7798 |
| accuracy_counting_abstain_wrong | 0.7083 |

This is lower than patched-gold evaluation because the manual review found several original labels were function/genre-like rather than semantic `topic.domain` labels. The patched gold remains the better dev target for v2.1.

## Should v2.1 Replace v2 as Current MVP Baseline?

Yes, with a caveat.

v2.1 should become the current MVP weak `topic.domain` rule baseline because it improves:

- coverage: `0.8000 -> 0.9083`;
- accuracy_on_answered: `0.7500 -> 0.8165`;
- strict accuracy counting abstain wrong: `0.6000 -> 0.7417`;
- correct answered records: `72 -> 89`.

Caveat:

- This is still a dev-benchmark result on `real_hf_benchmark_v1`.
- v2.1 should not be tuned further on this same 120-record benchmark without either a held-out v2-test or a clear manual-review target.

## What Should Come Next?

Recommended next step:

**Create a held-out v2-test before further tuning.**

Reason:

- v2.1 already fixed the obvious, manually reviewed rule gaps.
- Further keyword tuning on v1 risks overfitting.
- A small held-out benchmark can tell whether v2.1 generalizes before any embedding bake-off or v2.2 work.

After held-out testing:

- If v2.1 generalizes, keep it as the transparent MVP baseline.
- If failures remain semantic rather than lexical, run topic-only embedding bake-off.
- Do not tune reference/education broadly until the schema boundary is clearer.

## Checks Run

- `py_compile` for `scripts/classify_topic_domain_v2.py`
- v2.1 classifier on the three tokenized feature files
- evaluator on patched gold per dataset and combined
- evaluator on original gold combined for diagnostics
- output method integrity check confirmed all v2.1 outputs use `weak_topic_domain_v2_1_keyword_surface_prior`

