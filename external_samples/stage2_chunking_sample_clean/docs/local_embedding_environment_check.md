# Local embedding environment check

## Purpose

Check whether the embedding baseline can run locally without installing packages, using the network, or downloading models.

## Results

- Python executable: `C:\Users\pervo\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- Python version: `3.14.1`
- `sentence-transformers` import: unavailable
- `torch` import: unavailable
- local model/cache status:
  - `C:\Users\pervo\.cache\huggingface\hub` exists;
  - matching cache entry found: `models--sentence-transformers--LaBSE`;
  - no local `sentence-transformers/all-MiniLM-L6-v2` cache entry was found;
  - `C:\Users\pervo\.cache\torch\sentence_transformers` was not found;
  - `C:\Users\pervo\.cache\sentence_transformers` was not found.
- real embedding run safe now: no.

## Decision

`dependencies_missing`

The environment is not ready for a real embedding run because both `sentence-transformers` and `torch` are unavailable. A related model cache entry for LaBSE exists, but the requested MiniLM model was not found and the dependency is missing.

## Next command

Safe dry-run, already tested:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\classifier_benchmark_embedding_labeled.jsonl --dry-run
```

Later, after explicit approval to install dependencies and prepare local model files:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\classifier_benchmark_embedding_labeled.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --min-confidence 0.35
```

Do not run the real command until dependencies and model files are available locally.
