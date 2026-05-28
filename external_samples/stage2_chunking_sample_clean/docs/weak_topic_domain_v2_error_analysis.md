# Weak topic.domain v2 error analysis

## Purpose

This report analyzes the current `weak_topic_domain_v2` baseline on `real_hf_benchmark_v1` before any tuning. The goal is to separate:

- rule or keyword-profile weaknesses that can be fixed conservatively;
- ambiguous pseudo-gold or domain-boundary cases;
- cases where abstention is better than guessing;
- cases that should wait for a future embedding-model bake-off.

No classifier, taxonomy, threshold, feature-extractor, tokenizer, HF, MiniLM, or NLL step was changed for this analysis.

## Current Metrics Recap

Input predictions:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_topic_domain_v2.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_topic_domain_v2.jsonl`

Gold reference:

- `data_samples/real_hf_benchmark_v1_annotation_v2_pseudo_gold.jsonl`

Overall metrics:

| Metric | Value |
| --- | ---: |
| records_total | 120 |
| answered_count | 96 |
| abstained_count | 24 |
| coverage_rate | 0.8000 |
| correct answered | 67 |
| incorrect or abstained | 53 |
| accuracy_on_answered | 0.6979 |
| accuracy_counting_abstain_wrong | 0.5583 |
| gold_abstained_count | 3 |

Per-domain precision/recall highlights:

| Domain | Precision | Recall | Diagnosis |
| --- | ---: | ---: | --- |
| technology | 1.0000 | 0.7083 | High precision, misses patent/POS cases to commercial or abstain. |
| science | 0.9375 | 0.4839 | High precision, low recall; many science chunks abstain or shift to stem/social_sciences. |
| stem | 0.7273 | 0.6957 | Decent but surface math sometimes steals software/science cases. |
| education | 0.7143 | 0.4545 | Too many abstentions and boundary cases with humanities/commercial/media. |
| reference | 0.5000 | 0.1667 | Weakest clear domain; often acts like genre/function rather than topic. |
| commercial | 0.0000 | 0.0000 | Commercial predictions are mostly false positives from product/service/POS language. |
| social_sciences | 0.2857 | 1.0000 | Over-predicts social_sciences when policy/community/gender words appear. |

## Main Failure Groups

Top mismatch groups:

| Mismatch | Count | Main dataset(s) | Short diagnosis |
| --- | ---: | --- | --- |
| `stem -> ABSTAIN` | 7 | FineMath | Stem evidence often only comes from weak FineMath prior and stays below `min_score`. |
| `technology -> commercial` | 5 | FineWeb | POS/patent transaction text triggers commercial keywords more strongly than technology. |
| `science -> ABSTAIN` | 5 | FineWeb-Edu, FineMath | Science keyword coverage is too narrow, especially educational science and formula-heavy physics. |
| `science -> stem` | 4 | FineMath | Math notation/symbol features overpower science content in physics/measurement chunks. |
| `education -> ABSTAIN` | 3 | FineWeb-Edu | Education signal is weak when text is educational but not classroom/lesson vocabulary. |
| `science -> social_sciences` | 3 | FineWeb-Edu | Climate/community/policy text sits on science vs social-science boundary. |
| `media -> ABSTAIN` | 3 | FineWeb | News/media chunks without explicit media vocabulary are missed. |
| `reference -> ABSTAIN` | 2 | FineWeb-Edu | Reference is a weak topic label and often overlaps with genre/function. |
| `technology -> ABSTAIN` | 2 | FineWeb | Technology profile still misses some patent-style technical chunks. |
| `software -> stem` | 2 | FineMath | Algorithmic/math-heavy software explanations are pulled to stem. |

Abstention by gold domain:

| Gold domain | Abstentions |
| --- | ---: |
| stem | 7 |
| science | 5 |
| education | 3 |
| media | 3 |
| reference | 2 |
| technology | 2 |
| unknown | 1 |
| commercial | 1 |

## Evidence Cluster Diagnosis

The analysis script groups evidence into rough clusters:

| Cluster | Count | Meaning |
| --- | ---: | --- |
| keyword | 101 | Most predictions are driven by keyword profiles. |
| provenance_prior | 76 | FineMath/FineWeb-Edu priors are frequently present, but usually weak. |
| surface_feature | 42 | Math/code/symbol/token-density features affect many predictions. |
| abstain | 24 | Abstention is a large part of error accounting. |
| low_margin | 16 | Some predictions are close enough that abstention may be preferable. |

Most common keyword evidence included `program`, `school`, `function`, `store`, `community`, `display`, `learning`, `pos`, `data tap`, `social`, `sale`, `circuit`, `service`, `computer`, `mean`, `training`, `teacher`, `political`, `invoice`, and `policy`. This confirms that v2 is mostly a transparent keyword/surface/prior baseline, not a semantic model.

## Failure Group Inspection

### 1. Technology predicted as commercial

Representative example:

- `FineWeb_000009_038`
- gold: `technology`
- predicted: `commercial`
- confidence: `0.7088`
- evidence: `customer`, `sale`, `rental`, `tax` vs `transaction`, plus API/command surface signal
- note: technology/patent topic; primary text appears usable

Diagnosis:

- Likely classifier weakness.
- POS/patent transaction text contains commercial words, but the actual topic is technical/patent-like.
- Current commercial keywords are too eager when patent/POS context is present.

Safe fix candidate:

- Add a conservative patent/POS context guard: when `pos`, `data tap`, `lan adapter`, `apparatus`, `embodiment`, `circuit`, `terminal`, `transaction`, or patent-style language appears together, boost technology or cap commercial evidence.

Overfitting risk:

- Moderate. The v1 benchmark contains a visible POS/patent cluster. Fixes should use broad patent/technical context, not exact benchmark phrases only.

### 2. Stem abstained

Diagnosis:

- Likely classifier weakness plus conservative abstention.
- Many FineMath chunks only get `FineMath weak provenance prior (+0.9)`, below the default `min_score = 1.4`.
- This causes abstention even when pseudo-gold topic is clearly `stem`.

Safe fix candidate:

- Add a small set of general STEM/math terms that are not benchmark-specific: `ratio`, `percent`, `measurement`, `units`, `formula`, `estimate`, `average`, `proportion`, `integer`, `graph`, `sequence`.
- Consider allowing FineMath prior to answer only when paired with math/symbol/number-heavy surface features, not by provenance alone.

Overfitting risk:

- Low to moderate if the terms are broad and the FineMath prior remains weak.

### 3. Science predicted as stem

Representative examples:

- `FineMath_000026_023`: gold `science`, predicted `stem`; strong math notation, symbol density, token density, and FineMath prior.
- Physics/measurement examples with Newtons, force, units, and formula-heavy text.

Diagnosis:

- Mixed classifier weakness and domain-boundary issue.
- Surface math features correctly detect notation but incorrectly treat formula-heavy science as `stem`.
- Pseudo-gold treats physical science as `science`, while the surface format is mathematical.

Safe fix candidate:

- Add natural-science context terms that can override pure stem surface evidence: `force`, `newton`, `mass`, `weight`, `pressure`, `energy`, `organism`, `species`, `cell`, `chemical`, `reaction`, `molecule`.
- Keep `has_math_notation` as surface evidence, not a hard topic decision.

Overfitting risk:

- Low if implemented as broad science vocabulary. Medium if tuned around only FineMath physics examples.

### 4. Science predicted as social_sciences

Representative pattern:

- Climate/community/policy/disaster text where science, government, social-science, and media signals all appear.

Diagnosis:

- Mostly domain-boundary ambiguity.
- Climate impacts and climate justice are valid science-adjacent and social-science-adjacent content.
- Pseudo-gold selected `science` for several mixed chunks, but the classifier finds social/policy/community terms.

Safe fix candidate:

- For climate/disaster/environment chunks, abstain when science and social-sciences are close unless strong natural-science terms dominate.
- Alternatively split later into a cross-axis topic plus policy/genre/function axis, but that is beyond v2.0.

Overfitting risk:

- High if forced into one label. Abstention is safer than aggressive keyword changes.

### 5. Reference confused or abstained

Representative examples:

- `FineWeb-Edu_000014_000`: gold `reference`, predicted `media`; journalist/television/published words dominate.
- Other reference chunks are predicted as software, humanities, media, or abstained.

Diagnosis:

- Schema/gold ambiguity more than simple classifier bug.
- `reference` behaves partly like genre/function, not a semantic topic domain.
- The same text can be reference-style while semantically about war, history, media, software, or science.

Safe fix candidate:

- Do not aggressively tune `reference` as a primary topic.
- Prefer abstention or later optional `genre_or_function=reference`.
- Keep `reference` for clearly dictionary/encyclopedia/catalog/bibliography chunks only.

Overfitting risk:

- High. Improving reference recall by keywords may damage semantic topic grouping.

### 6. High-confidence wrong predictions

Examples:

| chunk_id | gold | predicted | confidence | Diagnosis |
| --- | --- | --- | ---: | --- |
| `FineWeb_000023_000` | education | humanities | 1.0000 | Art/history terms dominate; gold says educational function. Boundary issue. |
| `FineWeb_000007_000` | media | science | 1.0000 | Accident/news chunk contains medical terms; media/news signal too weak. |
| `FineWeb_000009_038` | technology | commercial | 0.7088 | POS patent text uses commercial transaction vocabulary. |
| `FineMath_000020_001` | software | stem | 0.7241 | Algorithm complexity text has math notation and FineMath prior. |
| `FineMath_000026_005` | science | education | 0.7639 | Lesson/teacher/student words overpower force/science context. |

Diagnosis:

- Confidence is a normalized score share, not a calibrated probability.
- A single strong keyword group can produce high confidence even when the coarse domain boundary is wrong.

Safe fix candidate:

- v2.1 should report "confidence" as heuristic confidence only, and possibly add a `decision_reason` or `risk_flags` field later.
- Add conservative guards for known cross-domain patterns rather than globally lowering confidence.

### 7. Low-confidence correct predictions

Example:

- `FineWeb-Edu_000000_000`
- gold: `science`
- predicted: `science`
- confidence: `0.4269`
- evidence: climate/health vs government/education/media/reference

Diagnosis:

- Some correct answers are low-confidence because the text is mixed.
- Raising thresholds or increasing abstention globally would remove useful correct predictions.

Safe fix candidate:

- Keep main KPI as `accuracy_on_answered + coverage`.
- Avoid global threshold tuning before inspecting held-out data.

### 8. Gold unknown / gold abstained records

Examples:

- `FineMath_000012_002`: gold abstained, classifier abstained. Good behavior.
- `FineWeb-Edu_000008_000`: gold abstained, classifier predicted `reference` from `encyclopedia`.
- `FineMath_000012_001`: gold abstained, classifier predicted `software` from `http`.

Diagnosis:

- Classifier still sometimes answers on noisy chunks when a keyword appears.
- The `mostly_noise` quality gate helps, but not all noisy chunks are caught or enough to force abstention.

Safe fix candidate:

- Prefer abstention when quality/noise evidence is strong and topic evidence is based on only one weak keyword.
- Do not solve this with semantic taxonomy changes.

## What Looks Fixable by Rules

Good candidates for conservative v2.1:

1. Patent/POS technology vs commercial:
   - Add patent/POS context guard.
   - Cap commercial keywords when strong patent/technology context is present.

2. Stem abstention:
   - Add broad math/statistics/measurement terms.
   - Allow FineMath prior to answer only with supporting math/surface/numeric evidence.

3. Science vs stem:
   - Add broader physics/chemistry/biology unit/context terms.
   - Prevent `has_math_notation` from overpowering science context alone.

4. News/media recall:
   - Add conservative news-event terms such as `authorities`, `injured`, `killed`, `reported`, `officials`, `police`, but only if validated against examples.

5. Noisy gold-abstained cases:
   - Strengthen abstention when quality residue/noise is present and domain score is based on one keyword.

## What Should Be Handled by Abstention

Cases where guessing is risky:

- climate justice / policy / community / disaster texts with both science and social_sciences evidence;
- reference-like pages whose semantic topic is a different domain;
- educational-function text about humanities/science/business where "education" is not the semantic topic;
- noisy chunks where a single keyword triggers a confident topic.

For these, v2.1 should prefer abstention or lower-confidence output over forced labels.

## What Should Wait for Embedding Bake-Off

Do not solve these with more keyword tuning yet:

- broad semantic differences between reference, media, education, and humanities;
- mixed climate/social/science cases;
- long patent or technical chunks where semantic intent is not clear from keywords alone;
- deciding whether "education" is topic, genre/function, or both.

These are good candidates for later MiniLM vs stronger embedding-model comparison for `topic.domain` only, using the annotation_v2 pseudo-gold as dev data.

## Manual Review Candidates

Before implementing v2.1, manually inspect at least these groups:

- all `technology -> commercial` examples;
- all `science -> social_sciences` examples;
- all `reference -> *` errors;
- high-confidence wrong predictions above `0.70`;
- gold-abstained records where the classifier answered.

Manual review should decide whether the pseudo-gold domain is still right under the v2 policy or whether the chunk is genuinely ambiguous and should be marked abstained in future benchmark revisions.

## Recommendation for v2.1

Do not implement v2.1 immediately from metrics alone. The next immediate step should be:

**B. Manual review of ambiguous pseudo-gold cases.**

Reason:

- There are clear rule-fixable clusters, but several largest errors are boundary cases, not pure classifier bugs.
- `reference`, `education`, `media`, and climate/social-science cases expose schema ambiguity.
- Benchmark v1 is already a dev benchmark; tuning directly against it without a short review risks overfitting.

After that review, implement a small v2.1 with only conservative changes:

1. patent/POS technology guard;
2. broader but generic stem/science keywords;
3. formula-heavy science override;
4. stronger abstention for noisy/single-keyword cases;
5. no changes to taxonomy, feature extractor, tokenizer, or old classifiers.

## Generated Artifacts

- `scripts/analyze_topic_domain_v2_errors.py`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_error_analysis.json`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_error_examples.jsonl`
- `docs/weak_topic_domain_v2_error_analysis.md`

