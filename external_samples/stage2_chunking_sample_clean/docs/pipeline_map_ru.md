# Stage2 pipeline map

Дата: 2026-05-26

## High-level flow

```text
raw/local/HF rows
  -> row adapter / source metadata
  -> logical chunks JSONL
  -> validation
  -> rule-based labels
  -> lexical nearest-label labels
  -> comparison and disagreement review
  -> future MiniLM embedding labels
```

## Current pipeline stages

| Stage | Inputs | Outputs | Scripts |
| --- | --- | --- | --- |
| Local chunking | `examples/*.jsonl` | `data_samples/*_chunks*.jsonl`, stats | `sample_fineweb_chunks.py` |
| Validation | chunks/labeled JSONL | pass/fail diagnostics | `validate_chunks.py` |
| Inspection | chunks/labeled JSONL | terminal preview | `inspect_chunks.py` |
| Rule-based labeling | chunks JSONL | `*_labeled_rule_based.jsonl` or canonical labeled output | `classify_chunks_rule_based.py` |
| Lexical labeling | chunks JSONL + taxonomy | `*_labeled_lexical.jsonl` | `classify_chunks_lexical_baseline.py` |
| Evaluation | synthetic labeled output | accuracy/mismatch report | `evaluate_chunk_labels.py`, `check_label_consistency.py` |
| Comparison | two labeled outputs | agreement/disagreement report | `compare_label_runs.py`, `review_label_mismatches.py` |
| Source planning | registry | planned commands only | `inspect_dataset_sources.py`, `plan_real_sample_run.py`, `plan_real_source_pipeline.py` |
| Future embedding | chunks JSONL + taxonomy + local model | `*_labeled_embedding_*.jsonl` | `classify_chunks_embedding_baseline.py` |

## Inputs

Stable local inputs:

- `examples/local_docs_edge_cases.jsonl`
- `examples/local_docs_classifier_benchmark.jsonl`
- `taxonomy/simple_domain_labels.json`
- `config/dataset_sources.json`

Real tiny sample inputs already generated:

- `data_samples/real_samples/fineweb_edu_sample10bt_chunks.jsonl`
- `data_samples/real_samples/finemath_chunks.jsonl`

## Outputs

Synthetic/local outputs:

- `data_samples/edge_case_chunks_sample.jsonl`
- `data_samples/edge_case_chunks_labeled.jsonl`
- `data_samples/classifier_benchmark_chunks.jsonl`
- `data_samples/classifier_benchmark_labeled.jsonl`
- `data_samples/classifier_benchmark_lexical_labeled.jsonl`

Real tiny outputs:

- `data_samples/real_samples/fineweb_edu_sample10bt_labeled_rule_based.jsonl`
- `data_samples/real_samples/fineweb_edu_sample10bt_labeled_lexical.jsonl`
- `data_samples/real_samples/finemath_labeled_rule_based.jsonl`
- `data_samples/real_samples/finemath_labeled_lexical.jsonl`

Future MiniLM outputs should use a distinct suffix and must not overwrite existing rule-based or lexical files.

## Safe vs approval-required

Safe local:

- validation;
- inspection;
- synthetic benchmark pipeline;
- comparison of existing outputs;
- planning scripts that only print commands.

Requires approval:

- HF streaming;
- new real sample generation;
- dependency installation;
- model download;
- MiniLM inference;
- NLL/logprob scoring.
