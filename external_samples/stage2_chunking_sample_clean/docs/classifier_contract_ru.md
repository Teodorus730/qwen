# Classifier contract

Дата: 2026-05-26

This contract defines how rule-based, lexical, and future embedding classifiers should write labels for stage2 chunks.

## Source metadata is not predicted label metadata

Input/source metadata:

- `dataset`: dataset or local sample identity.
- `source_type`: broad origin/type, for example `web_general`, `educational`, `math`, `unknown`.
- source-specific ids or run metadata when available.

Predicted labels:

- `domain`;
- `field`;
- `subfield`;
- `confidence`;
- `label_method`.

Classifiers should preserve `dataset`. They should not silently reinterpret a source as another dataset.

`source_type` should normally be preserved as source metadata. If a classifier needs to propose a different source-like category later, that should be a separate field or explicitly documented behavior, not a hidden overwrite.

## Label tuple

The classifier label is:

```text
(domain, field, subfield)
```

Examples:

- `stem / mathematics / calculus`
- `education / general_education / article`
- `web / boilerplate_or_navigation / page_noise`

`subfield` may be `null` when taxonomy intentionally has no deeper label.

`domain` and `field` may be `null` only when:

- chunk has not been classified yet;
- real-sample mode allows unknown/low-confidence output;
- classifier explicitly abstains;
- validation is running on raw chunks, not labeled outputs.

Synthetic benchmark labeled outputs should generally have labels.

## Confidence

`confidence` is classifier-specific.

It is not a calibrated probability.

Interpretation:

- rule-based: heuristic score or confidence assigned by transparent rules;
- lexical: lexical similarity/overlap score;
- embedding: cosine similarity or nearest-label score.

Do not compare raw confidence values across classifier families as if they were the same probability scale. Compare agreement, rank, low-confidence counts, and manual review examples instead.

## label_method

`label_method` identifies how labels were produced.

Recommended values:

- `rule_based`;
- `lexical_nearest_label`;
- `embedding_nearest_label_minilm` or a model-specific variant;
- `manual_review` if a human later changes labels.

The method should be stable enough for comparison scripts and review reports.

## Synthetic benchmark mode

Synthetic benchmark inputs have expected labels.

In this mode:

- strict validation is appropriate;
- `--require-labels` is appropriate for labeled outputs;
- accuracy/mismatch reports are meaningful;
- classifier changes can be regression-tested.

Main files:

- `examples/local_docs_classifier_benchmark.jsonl`
- `data_samples/classifier_benchmark_chunks.jsonl`
- `data_samples/classifier_benchmark_labeled.jsonl`
- `data_samples/classifier_benchmark_lexical_labeled.jsonl`

## Real sample mode

Real samples do not have gold labels.

In this mode:

- `--require-labels` is usually not appropriate for raw chunks;
- null or unknown labels may be valid;
- agreement/disagreement is a review signal, not accuracy;
- manual inspection is required before treating labels as evidence.

Main files:

- `data_samples/real_samples/fineweb_edu_sample10bt_*`
- `data_samples/real_samples/finemath_*`

## Relationship between classifiers

Rule-based classifier:

- transparent baseline;
- useful for predictable rules and sanity checks;
- may miss semantic similarity.

Lexical nearest-label classifier:

- cheap taxonomy similarity baseline;
- useful before MiniLM;
- can confuse related labels when keyword overlap is weak.

Embedding nearest-label classifier:

- next milestone;
- should use the same chunks and taxonomy;
- should write comparable output fields;
- should not overwrite rule-based or lexical files.

## Expected MiniLM output schema

Future MiniLM outputs should keep the base chunk schema and add/update:

```json
{
  "domain": "stem",
  "field": "mathematics",
  "subfield": "algebra",
  "confidence": 0.73,
  "label_method": "embedding_nearest_label_minilm"
}
```

Recommended optional fields:

- `model_name`;
- `taxonomy_path`;
- `top_k_labels`;
- `low_confidence`;
- `embedding_threshold`;
- `text_truncation`.

If optional fields are added, they should be documented before outputs are promoted.
