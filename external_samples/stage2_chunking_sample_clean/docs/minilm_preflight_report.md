# MiniLM preflight report

Date: 2026-05-26

Scope: stage2-local update of `scripts/classify_chunks_embedding_baseline.py` before any real MiniLM/Sentence-Transformers inference.

No HF streaming, dependency installation, model download, real MiniLM inference, commit, or push was performed.

## What changed

The embedding baseline is now closer to the classifier contract:

- preserves input `dataset`, `source_type`, `text`, `token_count`, and other existing fields;
- fills/updates only predicted label fields: `domain`, `field`, `subfield`, `confidence`, `label_method`;
- uses `label_method = embedding_nearest_label_minilm`;
- adds `low_confidence` instead of using a separate low-confidence method name;
- records per-output metadata:
  - `embedding_model`;
  - `taxonomy_path`;
  - `min_confidence`;
  - `batch_size`;
  - `device_requested`;
  - `device_actual`;
  - `text_chars`;
  - `top_k_labels`;
- supports `--batch-size`;
- supports optional `--device`;
- encodes chunk texts in batches;
- uses matrix cosine similarity when `numpy` is available, with a small pure-Python fallback for smoke tests;
- keeps dry-run model-free and output-free;
- fails closed when `sentence-transformers` or a local model is unavailable.

## Safe dry-run

This command does not load a model and does not write output:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\classifier_benchmark_labeled_embedding_minilm.jsonl --dry-run --batch-size 8 --top-k 3
```

## Future local-only run

Run only after explicit approval and after confirming that dependencies and model files are already local:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\classifier_benchmark_labeled_embedding_minilm.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --batch-size 32 --top-k 3
```

Expected behavior:

- request `local_files_only=True`;
- fail if the model is missing;
- write comparable JSONL output;
- do not overwrite rule-based or lexical outputs unless the user chooses the same output path by mistake.

## Remaining blockers

- No dependency/model approval yet.
- No verified local MiniLM path/cache recorded yet.
- Real MiniLM output quality is untested.
- Top-k is stored per record, but there is no separate aggregate top-k report yet.
- No ANN/index optimization; not needed for the current tiny MVP.

## Checks to run before real inference

- dry-run embedding baseline;
- smoke test embedding baseline;
- validate benchmark inputs;
- run local benchmark pipeline;
- confirm local model path/cache;
- run first on `classifier_benchmark_chunks.jsonl`;
- validate output;
- compare embedding vs rule-based and lexical;
- then run FineWeb-Edu/FineMath tiny samples.
