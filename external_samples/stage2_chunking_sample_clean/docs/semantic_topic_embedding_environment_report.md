# Semantic topic embedding environment report

## Purpose

This report records the current local embedding environment and model-cache state for the `semantic_topic_domain_v1` bake-off scaffold.

No embedding inference, model download, dependency installation, HF streaming, v2-test access, NLL/logits, or classifier tuning was performed.

## Minimal status

Repository root:

```text
C:\Users\pervo\PycharmProjects\qwen
```

Branch:

```text
feature/review-stage2-chunking
```

The stage2-local bake-off scaffold already exists under:

- `external_samples/stage2_chunking_sample_clean/scripts/classify_semantic_topic_embedding.py`
- `external_samples/stage2_chunking_sample_clean/scripts/evaluate_semantic_topic_predictions.py`
- `external_samples/stage2_chunking_sample_clean/scripts/run_semantic_topic_embedding_bakeoff.py`
- `external_samples/stage2_chunking_sample_clean/docs/semantic_topic_embedding_bakeoff_implementation_plan.md`

## Python environment

The embedding environment was found at:

```text
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe
```

Python version:

```text
3.11.9
```

Checked executable:

```text
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe
```

Package import check:

```text
ok
torch: 2.12.0+cpu
sentence_transformers: 5.5.1
transformers: 5.9.0
huggingface_hub: 1.16.4
numpy: 2.4.6
```

Conclusion:

- `.venv-embedding` exists in the repository root, not inside the stage2 folder.
- The environment can import the required embedding packages.
- CPU Torch is available.

## Cache inventory

Stage2 embedding cache:

```text
external_samples/stage2_chunking_sample_clean/.hf_embedding_cache
exists: false
```

Stage2 tokenizer cache:

```text
external_samples/stage2_chunking_sample_clean/.hf_tokenizer_cache
exists: true
contains Qwen tokenizer/model cache material for tokenizer infrastructure
```

Global Hugging Face hub cache:

```text
C:\Users\pervo\.cache\huggingface\hub
exists: true
```

Observed model directories:

- `models--sentence-transformers--all-MiniLM-L6-v2`
- `models--BAAI--bge-m3`
- `models--BAAI--bge-reranker-v2-m3`
- `models--intfloat--multilingual-e5-large-instruct`
- `models--Qwen--Qwen3-4B-Instruct-2507`
- `models--Qwen--Qwen3-8B`
- `models--sentence-transformers--LaBSE`

## Model snapshot status

### `sentence-transformers/all-MiniLM-L6-v2`

Cache root exists:

```text
C:\Users\pervo\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2
```

Snapshot status:

```text
snapshots: missing
refs/main: c9745ed1d9f207416be6d2e6f8de32d1f16199bf
```

Conclusion:

- MiniLM is not currently usable as a fully cached local model.
- Running MiniLM now would likely require a model download.

### `BAAI/bge-m3`

Cache root exists:

```text
C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3
```

Snapshots observed:

```text
C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181
C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\9a0624b896d81da7492a910ffa53731274b6cf3d
```

`refs/main` points to:

```text
5617a9f61b028005a4858fdac845db406aefb181
```

Snapshot `5617a9f61b028005a4858fdac845db406aefb181` contains expected usable files:

- `config.json`
- `config_sentence_transformers.json`
- `modules.json`
- `pytorch_model.bin`
- `sentencepiece.bpe.model`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `1_Pooling`

Snapshot `9a0624b896d81da7492a910ffa53731274b6cf3d` contains only:

- `model.safetensors`

Conclusion:

- BGE-M3 appears locally available through the full `5617...` snapshot.
- The later `9a06...` snapshot appears incomplete on its own.
- For a no-download BGE-M3 run, prefer the explicit full snapshot path or update cache resolution to follow `refs/main` rather than mtime.

### `intfloat/multilingual-e5-large-instruct`

Cache root exists:

```text
C:\Users\pervo\.cache\huggingface\hub\models--intfloat--multilingual-e5-large-instruct
```

Snapshot status:

```text
snapshots: missing
refs/main: 274baa43b0e13e37fafa6428dbc7938e62e5c439
```

Conclusion:

- E5-large-instruct is not currently usable as a fully cached local model.
- Running it would likely require a download.

### `intfloat/multilingual-e5-base`

Cache root:

```text
missing
```

Conclusion:

- E5-base is not cached.

### `Qwen/Qwen3-Embedding-0.6B`

Cache root:

```text
missing
```

Conclusion:

- Qwen3 Embedding 0.6B is not cached.

## Script checks through `.venv-embedding`

From:

```text
C:\Users\pervo\PycharmProjects\qwen\external_samples\stage2_chunking_sample_clean
```

`py_compile` passed for:

- `scripts/classify_semantic_topic_embedding.py`
- `scripts/evaluate_semantic_topic_predictions.py`
- `scripts/run_semantic_topic_embedding_bakeoff.py`

Classifier dry-run with MiniLM passed input/taxonomy validation:

```text
records: 120
domains: 11
cached_snapshot: not found
```

Classifier dry-run with `BAAI/bge-m3` passed input/taxonomy validation and detected a local snapshot, but selected the mtime-latest snapshot:

```text
cached_snapshot: C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\9a0624b896d81da7492a910ffa53731274b6cf3d
```

This selected snapshot appears incomplete, so a real BGE-M3 run should use the explicit full snapshot path or fix cache selection first.

## Can MiniLM run without download?

No.

Blocker:

- the MiniLM cache root exists, but there is no `snapshots` directory;
- only `refs/main` was observed.

MiniLM should not be run until either:

- the full snapshot is present locally; or
- the user explicitly approves downloading it.

## Can BGE-M3 run without download?

Probably yes, with a caveat.

Usable local snapshot appears to be:

```text
C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181
```

However, the current scaffold cache resolver may choose the incomplete `9a06...` snapshot because it sorts snapshots by modification time. A safe no-download BGE-M3 run should either:

1. pass the full `5617...` snapshot path as `--model`; or
2. update the classifier cache resolver to follow `refs/main` and prefer snapshots with `config.json` and `modules.json`.

No real BGE-M3 inference was run in this session.

## Recommended cache policy

For now, use the global HF cache:

```text
C:\Users\pervo\.cache\huggingface\hub
```

The stage2 `.hf_embedding_cache` does not exist yet. Keeping the scaffold's `--cache-dir .hf_embedding_cache` is fine for future controlled downloads, but current local no-download runs should rely on the global cache or explicit snapshot paths.

## Next user action

Recommended immediate next step:

1. Decide whether to patch `classify_semantic_topic_embedding.py` cache resolution to follow `refs/main`.
2. If yes, run a BGE-M3 dry-run and then a small v1-dev prediction/evaluation using the full local snapshot.
3. If MiniLM is still desired as the lightweight baseline, explicitly approve downloading or restoring its full snapshot first.

No download or install is needed for BGE-M3 if using the full cached snapshot path.
