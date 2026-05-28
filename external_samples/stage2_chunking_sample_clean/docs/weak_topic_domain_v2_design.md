# Weak topic.domain v2 design

## Purpose

Weak `topic.domain` v2 is a coarse semantic baseline for annotation schema v2. It is meant to support future NLL/logprob/perplexity profiling by grouping chunks into broad topics with confidence and abstention.

This is not:

- `source_type`;
- field/subfield classification;
- MiniLM tuning;
- embedding model bake-off;
- NLL scoring;
- training data selection by itself.

## Allowed domains

The MVP domain set is intentionally coarse:

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

## Why coarse domain only

The previous benchmark showed that strict full-label evaluation was dominated by ambiguous `source_type` and fine-grained taxonomy coverage. For annotation schema v2, coarse `topic.domain` is the useful first semantic grouping:

- broad enough for NLL/probability profile buckets;
- easier to evaluate with pseudo-gold;
- compatible with abstention;
- less fragile than field/subfield labels.

Field/subfield can remain traceability or diagnostic metadata, but it is not the MVP target.

## Evaluation policy

Primary metrics:

- `coverage_rate = answered / total`;
- `accuracy_on_answered = correct_answered / answered`.

Secondary metric:

- `accuracy_counting_abstain_wrong`.

Gold records with `review_topic_abstained=true` should be reported separately because the pseudo-gold itself says topic is unreliable.

Do not use old full-label accuracy for this classifier. The classifier predicts only one axis: `topic.domain`.

## Signals used

### Lexical keyword profiles

Each allowed domain has a small keyword profile. The classifier scores exact-ish lexical indicators from text. This keeps the baseline transparent and dependency-free.

### Annotation v2 surface features

Surface features can support a topic but should not force it alone:

- `has_math_notation` and `is_symbol_heavy` support `stem`;
- `has_code` and `has_api_or_command_syntax` support `software`;
- `has_scientific_formula` supports `science` and sometimes `stem`;
- token-density features can provide weak support for formula-heavy STEM content.

### Provenance priors

Priors are deliberately weak:

- FineMath weakly supports `stem`;
- FineWeb-Edu weakly supports `education`, `science`, `humanities`, or `reference` depending on text;
- FineWeb does not force any topic.

Provenance is never treated as a label. It only breaks weak ties when the text is compatible.

### Abstention

The classifier abstains when:

- the record is `mostly_noise`;
- the top score is too low;
- the top two domains are too close;
- there is no meaningful text.

Abstention is a feature, not a failure. For future NLL/profiling, low-confidence chunks should not be forced into misleading groups.

## Confidence

Confidence is a simple normalized heuristic score. It is not a calibrated probability.

Each prediction stores:

- `domain`;
- `confidence`;
- `method`;
- `abstained`;
- `abstain_reason`;
- `top_k`;
- `evidence`.

## Risks

- Keyword bias: vocabulary chosen from benchmark v1 may overfit this dev set.
- Mixed chunks: one chunk can contain education, science, reference, and UI residue.
- Surface/topic leakage: formulas and code-like syntax are surface signals, not topics by themselves.
- Weak labels are not truth: the output should guide grouping and review, not replace pseudo-gold.

## Recommended next use

Use this baseline to establish a transparent lower bound for `topic.domain` grouping. Then inspect failure clusters before considering a stronger embedding model bake-off.
