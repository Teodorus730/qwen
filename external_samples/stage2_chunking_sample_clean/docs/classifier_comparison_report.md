# Classifier comparison report

## Purpose

Compare rule-based, lexical nearest-label, and optional embedding nearest-label approaches on the local synthetic benchmark.

## Methods

- `rule_based_keyword_v1`: transparent hand-written rules over chunk text.
- `lexical_nearest_label_v1`: no-dependency bag-of-words nearest-label baseline over taxonomy descriptions and keywords.
- `embedding_nearest_label_v1`: optional future baseline if local dependencies and model files are available.

## Current inputs

- `data_samples/classifier_benchmark_chunks.jsonl`
- `data_samples/classifier_benchmark_labeled.jsonl`
- `data_samples/classifier_benchmark_lexical_labeled.jsonl`
- `taxonomy/simple_domain_labels.json`

## Results

Rule-based benchmark accuracy:

- source_type_accuracy: 1.0000
- domain_accuracy: 1.0000
- field_accuracy: 1.0000
- subfield_accuracy: 1.0000
- full_label_accuracy: 1.0000

Lexical benchmark accuracy:

- source_type_accuracy: 0.9524
- domain_accuracy: 0.9524
- field_accuracy: 0.9524
- subfield_accuracy: 0.9524
- full_label_accuracy: 0.9524

Embedding benchmark accuracy:

- not run; `sentence-transformers` and `torch` are not available locally.

Rule-based vs lexical agreement:

- records in each run: 42
- common chunk_ids: 42
- matching full labels: 40
- differing full labels: 2
- agreement rate: 0.9524

Main disagreement types:

- educational media-literacy text mentioning retail words was mapped by lexical baseline to `commercial/product_page/retail`;
- useful environmental article with footer/cookie terms was mapped by lexical baseline to `web/boilerplate_or_navigation/page_noise`.

## Interpretation

The rule-based classifier is high on the synthetic benchmark because the benchmark and rules share explicit signals. This is useful for smoke/regression testing but does not prove real FineWeb quality.

The lexical baseline tests whether taxonomy descriptions and keywords are enough for nearest-label matching. It performs well on direct domain labels, but it is weak when misleading keywords appear in the wrong context.

The embedding baseline is the next real step, but it requires local model/dependency readiness. It was not run in this session.

None of these prove real FineWeb quality.

## Recommendation

Keep the lexical baseline as a cheap comparison point. Use it to detect taxonomy wording problems and to sanity-check embedding output later. The next substantial classifier step should be a local embedding run only after dependencies and model files are explicitly prepared.
