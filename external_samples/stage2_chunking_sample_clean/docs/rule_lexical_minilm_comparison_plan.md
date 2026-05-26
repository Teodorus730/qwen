# Rule-based vs lexical vs MiniLM comparison plan

Date: 2026-05-27

## Purpose

Compare three classifier families on the same stage2 chunks:

- transparent rule-based labels;
- no-dependency lexical nearest-label baseline;
- MiniLM/Sentence-Transformers embedding nearest-label baseline.

The goal is not to prove MiniLM is universally better. The goal is to understand where semantic embeddings help, where they overgeneralize, and whether the taxonomy/output contract is ready for tiny real samples.

## Inputs

Synthetic benchmark:

- `data_samples/classifier_benchmark_chunks.jsonl`
- `data_samples/classifier_benchmark_labeled.jsonl`
- `data_samples/classifier_benchmark_lexical_labeled.jsonl`
- future `data_samples/classifier_benchmark_minilm_labeled.jsonl`

Tiny real samples later:

- `data_samples/real_samples/fineweb_edu_sample10bt_chunks.jsonl`
- `data_samples/real_samples/fineweb_edu_sample10bt_labeled_rule_based.jsonl`
- `data_samples/real_samples/fineweb_edu_sample10bt_labeled_lexical.jsonl`
- future `data_samples/real_samples/fineweb_edu_sample10bt_labeled_embedding_minilm.jsonl`
- `data_samples/real_samples/finemath_chunks.jsonl`
- `data_samples/real_samples/finemath_labeled_rule_based.jsonl`
- `data_samples/real_samples/finemath_labeled_lexical.jsonl`
- future `data_samples/real_samples/finemath_labeled_embedding_minilm.jsonl`

## Metrics

Synthetic benchmark:

- source_type accuracy against expected labels;
- domain accuracy;
- field accuracy;
- subfield accuracy;
- full label accuracy;
- agreement rate between classifier outputs;
- disagreements by field;
- manual review examples.

Real samples:

- agreement rate;
- full label agreement;
- disagreements by `source_type`, `domain`, `field`, `subfield`;
- low-confidence count;
- top-k ambiguity from MiniLM output;
- manual review examples.

Do not report real-sample agreement as accuracy because real samples do not have gold labels.

## Commands

Run commands from:

```powershell
cd C:\Users\pervo\PycharmProjects\qwen\external_samples\stage2_chunking_sample_clean
```

### Rule-based pipeline

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\run_local_benchmark_pipeline.py
```

This refreshes:

- `data_samples/classifier_benchmark_chunks.jsonl`;
- `data_samples/classifier_benchmark_labeled.jsonl`;
- `data_samples/classifier_benchmark_run_stats.json`.

### Lexical baseline

Dry-run:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\classify_chunks_lexical_baseline.py `
  --input data_samples\classifier_benchmark_chunks.jsonl `
  --labels taxonomy\simple_domain_labels.json `
  --output data_samples\classifier_benchmark_lexical_labeled.jsonl `
  --dry-run
```

Real lexical output, only if regeneration is needed:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\classify_chunks_lexical_baseline.py `
  --input data_samples\classifier_benchmark_chunks.jsonl `
  --labels taxonomy\simple_domain_labels.json `
  --output data_samples\classifier_benchmark_lexical_labeled.jsonl
```

### Future MiniLM baseline

Only after dependency/model approval and local model availability:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\classify_chunks_embedding_baseline.py `
  --input data_samples\classifier_benchmark_chunks.jsonl `
  --labels taxonomy\simple_domain_labels.json `
  --output data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --batch-size 32 `
  --top-k 3
```

If using the current `.venv` after explicit approval, replace `.venv-embedding\Scripts\python.exe` with `.venv\Scripts\python.exe`.

### Validate outputs

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\validate_chunks.py --input data_samples\classifier_benchmark_labeled.jsonl --require-labels
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\validate_chunks.py --input data_samples\classifier_benchmark_lexical_labeled.jsonl --require-labels
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\validate_chunks.py --input data_samples\classifier_benchmark_minilm_labeled.jsonl --require-labels
```

### Evaluate against expected labels

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\evaluate_chunk_labels.py --input data_samples\classifier_benchmark_labeled.jsonl
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\evaluate_chunk_labels.py --input data_samples\classifier_benchmark_lexical_labeled.jsonl
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\evaluate_chunk_labels.py --input data_samples\classifier_benchmark_minilm_labeled.jsonl
```

### Compare classifier outputs

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\compare_label_runs.py `
  --left data_samples\classifier_benchmark_labeled.jsonl `
  --right data_samples\classifier_benchmark_lexical_labeled.jsonl `
  --left-name rule_based `
  --right-name lexical `
  --limit 10
```

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\compare_label_runs.py `
  --left data_samples\classifier_benchmark_labeled.jsonl `
  --right data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --left-name rule_based `
  --right-name minilm `
  --limit 10
```

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\compare_label_runs.py `
  --left data_samples\classifier_benchmark_lexical_labeled.jsonl `
  --right data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --left-name lexical `
  --right-name minilm `
  --limit 10
```

### Review mismatches

Use after MiniLM output exists:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\review_label_mismatches.py `
  --input data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --limit 20
```

If the review script expects expected-label fields and a different mode is needed later, document the mismatch before changing code.

## Decision criteria

MiniLM is useful if:

- it improves real/generalization cases, not just synthetic benchmark;
- it produces sensible labels on FineWeb-Edu tiny samples;
- it produces sensible labels on FineMath tiny samples;
- disagreements are explainable;
- low-confidence behavior is useful rather than noisy;
- top-k alternatives expose real ambiguity;
- it does not collapse to broad labels;
- it does not rewrite source metadata.

MiniLM is not ready to promote if:

- it mostly predicts one broad domain;
- it has many unexplained high-confidence mistakes;
- it fails the synthetic benchmark badly;
- it produces labels that are hard to compare with rule-based/lexical outputs;
- runtime/setup friction outweighs value for the current MVP.

## Risks

- synthetic benchmark may be too easy;
- label descriptions may be too short;
- taxonomy may be too coarse or uneven;
- confidence calibration is weak because cosine similarity is not probability;
- local CPU speed may be slow;
- dependency/model download friction may delay reproducibility;
- real samples have no gold labels, so manual review remains necessary.
