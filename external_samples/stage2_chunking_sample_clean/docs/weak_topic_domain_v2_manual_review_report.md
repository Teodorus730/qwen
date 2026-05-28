# Weak topic.domain v2 manual review report

## Purpose

This report reviews targeted failure cases from `weak_topic_domain_v2` before implementing any v2.1 changes. The goal is to decide:

- which current pseudo-gold labels should stay;
- which cases are better marked ambiguous or abstained;
- which classifier errors are safe to address with conservative rules;
- which cases should wait for a future embedding-model bake-off.

No pseudo-gold file, classifier, evaluator, taxonomy, feature extractor, tokenizer stats, HF sample, MiniLM run, or NLL step was changed.

## Inputs

Gold:

- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold.jsonl`

Predictions:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_topic_domain_v2.jsonl`

Previous error analysis:

- `data_samples/real_hf_benchmark_v1_topic_domain_v2_error_analysis.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_error_examples.jsonl`
- `docs/weak_topic_domain_v2_error_analysis.md`

Generated review files:

- `data_samples/real_hf_benchmark_v1_topic_domain_v2_manual_review_candidates.jsonl`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_manual_review_labeled.jsonl`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_manual_review_summary.json`

## Candidate Selection

The targeted review set contains 36 unique chunks selected from the main ambiguous/error groups:

| Error group | Reviewed |
| --- | ---: |
| `technology->commercial` | 5 |
| `science->social_sciences` | 3 |
| `reference->not_reference_or_abstain` | 5 |
| `high_confidence_wrong>=0.70` | 6 |
| `gold_abstained_classifier_answered` | 2 |
| `stem->ABSTAIN` | 5 |
| `science->ABSTAIN` | 5 |
| `education_media->ABSTAIN` | 5 |

The selection intentionally focuses on cases where the previous analysis suggested rule-fixable clusters, pseudo-gold ambiguity, or abstention-policy issues.

## Review Decisions

| Decision | Count | Meaning |
| --- | ---: | --- |
| `keep_gold` | 19 | Current pseudo-gold label looks acceptable; classifier should change or abstain behavior should be adjusted. |
| `revise_gold_domain` | 9 | Current pseudo-gold domain likely points to genre/function or an overly broad interpretation rather than primary topic. |
| `mark_gold_abstained` | 2 | Topic is too mixed or context is too thin for a reliable domain label. |
| `ambiguous_keep_for_embedding_bakeoff` | 6 | Current label can stay for now, but the case should not drive keyword tuning. |

Recommended actions:

| Action | Count | Interpretation |
| --- | ---: | --- |
| `v2_1_rule_fix` | 15 | Safe candidates for conservative weak-topic v2.1 changes. |
| `pseudo_gold_patch` | 11 | Current annotation_v2 pseudo-gold should be patched in a separate, explicit step. |
| `embedding_bakeoff_later` | 8 | Ambiguous semantic cases should wait for embedding-model comparison. |
| `v2_1_abstain_policy` | 2 | Classifier should abstain on noisy/single-keyword cases. |

The review found 11 cases where the revised topic/abstention differs from the current pseudo-gold.

## Cases Confirming Classifier Errors

### Patent/POS technology vs commercial

All 5 `technology->commercial` cases should keep gold `technology`.

Diagnosis:

- The text is patent-like or POS-system technical documentation.
- Commercial vocabulary such as `rental`, `sale`, `invoice`, `customer`, and `tax` describes the system being processed.
- It should not override patent/POS/technical context.

Safe v2.1 fix:

- Add a conservative POS/patent technology guard.
- Boost or protect `technology` when `POS`, `transaction`, `display`, `circuit`, `terminal`, `function key`, `apparatus`, `embodiment`, or patent-like flow language appears.
- Cap commercial evidence when it appears inside such technical context.

Risk:

- Moderate overfit risk because benchmark v1 contains a visible POS patent cluster. Keep terms broad and technical, not exact-chunk specific.

### Stem abstentions

All 5 sampled `stem->ABSTAIN` cases should keep gold `stem`.

Examples include:

- AMC math problem;
- inches/BMI/measurement formula text;
- MAPE/forecasting error metrics;
- holdout-set model evaluation;
- nowhere dense/functions/uniform norm.

Safe v2.1 fix:

- Add broad math/statistics/measurement keywords such as `problem`, `formula`, `measurement`, `inches`, `BMI`, `MAPE`, `forecast`, `holdout`, `model evaluation`, `norm`, `continuous function`.
- Let FineMath provenance help only when paired with math/statistics/surface evidence, not as a hard label.

### Science abstentions and science vs stem

Several reviewed science cases should keep `science`:

- botany/species description;
- great ape behavioral science article;
- force/physics lesson;
- Newton/unit calculation.

Safe v2.1 fix:

- Add broader science vocabulary: botany/tree/leaves/bark/acorn, animal/ape/chimpanzee/orangutan, force/Newton/weight/mass/adhesion.
- Treat formulas and symbol-heavy text as surface signals, not automatic `stem` topic.

### Media/news misses

The accident report case should keep `media`.

Safe v2.1 fix:

- Add conservative news-event cues such as `authorities`, `troopers`, `injured`, `killed`, `crash`, `reported`, `officials`.

This should be small and guarded, because "media" can become a catch-all if over-expanded.

## Cases Showing Pseudo-Gold Ambiguity

### Reference as genre/function

Most `reference` failures reveal that reference is partly genre/function, not a clean topic domain.

Patch candidates:

- A biography of a sociology professor should become `social_sciences`.
- A war/book review passage should become `humanities`.
- A Spanish political history passage should become `humanities`.
- A short trailing UN/book-review paragraph should be abstained.

Keep one true reference case:

- The numbered `Yodhājīva Sutta` catalogue entry should stay `reference`, but should not drive aggressive keyword tuning.

Implication:

- `reference` should be used narrowly for dictionary/encyclopedia/catalogue/bibliographic content.
- Broader "reference-like page" behavior belongs to optional future `genre_or_function`, not primary `topic.domain`.

### Education as function vs topic

Some old `education` labels look like teaching style or educational function rather than primary topic.

Patch candidates:

- service-quality explanation should become `commercial`;
- personal art/quilting narrative should become `humanities`;
- reflective seasons/life passage should become `humanities`;
- Mark Twain/loyalty passage should become `humanities`.

Ambiguous keep-for-bakeoff:

- force lesson text can stay `science` for now, but it mixes education function and physics topic.

Implication:

- MVP topic.domain should prefer the underlying subject where visible.
- Education should be reserved for cases where education itself is the topic or the best coarse grouping.

### Science vs social_sciences

Patch candidates:

- food shortages, rural economies, politics, corruption, and planning should become `social_sciences`;
- disaster trap/development/social aspects should become `social_sciences`;
- flood/logging/local government responsibility should be abstained.

Ambiguous case:

- climate adaptation plus tribal land management and public-agency policy can stay `science` for now but should be held for embedding bake-off.

Implication:

- Climate/environment/policy text is a difficult boundary.
- v2.1 should avoid forcing these cases unless one side has strong evidence.

## Abstention Policy Cases

Two gold-abstained records were answered by the classifier:

- `FineWeb-Edu_000008_000`: noisy encyclopedia/footer/license residue predicted as `reference`.
- `FineMath_000012_001`: spammy triangle/URL text predicted as `software` because of `http`.

Recommended v2.1 abstain policy:

- If a chunk is noisy or residue-heavy and the topic evidence is based on a single weak keyword, abstain.
- Do not let `http`, `encyclopedia`, or one isolated keyword override poor quality context.

## What Is Safe for v2.1

Safe, conservative changes:

1. Patent/POS technology guard.
2. Broader generic stem/math/statistics/measurement keywords.
3. Broader generic science vocabulary for botany, animals, force/physics, units.
4. Formula-heavy science guard: surface math should not automatically steal topic from science.
5. News-event cues for obvious media/news chunks.
6. Stronger abstention for noisy single-keyword cases.

Avoid in v2.1:

- changing allowed domains;
- adding field/subfield;
- making `reference` broad;
- globally lowering thresholds;
- using FineWeb-Edu provenance to force `education`;
- tuning directly around exact benchmark strings.

## What Should Wait for Embedding Bake-Off

Keep these for later semantic-model comparison:

- climate adaptation vs social policy;
- education function vs underlying subject;
- reference-like pages with real semantic topics;
- food/creative/media cases where the allowed domain set lacks a clean food/lifestyle category;
- personal historical/media/memoir cases.

These are exactly where keyword profiles are least trustworthy.

## Should Pseudo-Gold Be Patched Now?

Not inside this session. But the review recommends a separate explicit pseudo-gold patch step for 11 records:

- 9 `revise_gold_domain`;
- 2 `mark_gold_abstained`.

Patch before evaluating v2.1, otherwise v2.1 will be partly judged against labels this review now considers questionable.

## Recommendation

Do not implement v2.1 immediately in this same step.

The exact next step should be:

1. Apply a small annotation_v2 pseudo-gold patch for the 11 reviewed records.
2. Re-run only the existing evaluator to establish corrected v2.0 baseline metrics.
3. Then implement conservative `weak_topic_domain_v2.1` rule fixes.

This keeps the work clean:

- first fix benchmark labels where review found they are wrong or ambiguous;
- then tune the weak classifier;
- then compare v2.0 vs v2.1 without silently moving the target.

## Generated Artifacts

- `data_samples/real_hf_benchmark_v1_topic_domain_v2_manual_review_candidates.jsonl`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_manual_review_labeled.jsonl`
- `scripts/summarize_topic_domain_v2_manual_review.py`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_manual_review_summary.json`
- `docs/weak_topic_domain_v2_manual_review_report.md`

