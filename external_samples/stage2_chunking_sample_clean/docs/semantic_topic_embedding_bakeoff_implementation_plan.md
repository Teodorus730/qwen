# Semantic topic embedding bake-off implementation plan

## Purpose

This plan defines the controlled embedding bake-off scaffold for `semantic_topic_domain_v1`.

The bake-off should target the cleaned semantic axis, not the old mixed `topic.domain`, because the old field combined semantic subject labels with function-like labels such as `reference`, `education`, `media`, `commercial`, and `unknown`.

The new target asks one question:

> What is the text about?

`genre_function_v1` is intentionally out of scope for the primary embedding task. It can be handled separately by rules, LLM-assisted review, or manual labels later.

## Protocol

Development split:

- use `real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl`;
- use v1-dev for label-card design, model comparison, and threshold exploration;
- report results as dev pseudo-gold only.

Held-out policy:

- do not use v2-test for this bake-off implementation;
- do not tune thresholds or model choices on v2-test;
- before held-out claims under cleaned axes, decide whether to relabel v2-test under the new schema or create a fresh v3 split.

Prediction method:

1. Build one label card per `semantic_topic_domain_v1` value from:
   - domain name;
   - description;
   - positive examples;
   - negative notes.
2. Embed each label card.
3. Embed each chunk text.
4. Use cosine similarity against label cards.
5. Predict top-1 when score and margin pass thresholds.
6. Abstain when confidence or margin is too low.
7. Preserve all input fields and write prediction metadata under `semantic_topic_embedding`.

## Candidate models

Lightweight baseline:

- `sentence-transformers/all-MiniLM-L6-v2`

Larger multilingual / retrieval-oriented candidates:

- `BAAI/bge-m3`
- `intfloat/multilingual-e5-large-instruct`
- `intfloat/multilingual-e5-base`
- `Qwen/Qwen3-Embedding-0.6B`

## Model size and download caveats

No new models should be downloaded without explicit approval.

The scripts default to local-cache-only behavior. If a requested model is missing from local cache, classification exits nonzero and prints a clear blocker. A future run may pass `--allow-download`, but only after approval.

Expected rough caveats:

- MiniLM is small and useful as a quick baseline, but may underperform on nuanced domain distinctions.
- BGE-M3 and multilingual E5 models are larger and may improve label-card matching, especially for mixed web content.
- Qwen embedding models are more aligned with the broader Qwen-family infrastructure direction, but require careful resource planning.

## Current scaffold

Classifier:

```powershell
python scripts\classify_semantic_topic_embedding.py `
  --input data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --domain-descriptions taxonomy\semantic_topic_domain_v1_descriptions.json `
  --output data_samples\real_hf_benchmark_v1_semantic_topic_embedding_minilm.jsonl `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --cache-dir .hf_embedding_cache `
  --threshold 0.0 `
  --margin-threshold 0.0 `
  --top-k 3 `
  --text-field text `
  --dry-run
```

Evaluator:

```powershell
python scripts\evaluate_semantic_topic_predictions.py `
  --predictions data_samples\real_hf_benchmark_v1_semantic_topic_embedding_minilm.jsonl `
  --gold data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --output-json data_samples\real_hf_benchmark_v1_semantic_topic_embedding_minilm_eval.json `
  --dry-run
```

Runner:

```powershell
python scripts\run_semantic_topic_embedding_bakeoff.py `
  --gold data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --domain-descriptions taxonomy\semantic_topic_domain_v1_descriptions.json `
  --output-dir data_samples `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --dry-run
```

## Future real run commands

After environment setup and explicit approval for any missing downloads:

```powershell
python scripts\classify_semantic_topic_embedding.py `
  --input data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --domain-descriptions taxonomy\semantic_topic_domain_v1_descriptions.json `
  --output data_samples\real_hf_benchmark_v1_semantic_topic_embedding_minilm.jsonl `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --cache-dir .hf_embedding_cache `
  --threshold 0.0 `
  --margin-threshold 0.0 `
  --top-k 3 `
  --text-field text
```

```powershell
python scripts\evaluate_semantic_topic_predictions.py `
  --predictions data_samples\real_hf_benchmark_v1_semantic_topic_embedding_minilm.jsonl `
  --gold data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --output-json data_samples\real_hf_benchmark_v1_semantic_topic_embedding_minilm_eval.json
```

For multi-model comparison:

```powershell
python scripts\run_semantic_topic_embedding_bakeoff.py `
  --gold data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --domain-descriptions taxonomy\semantic_topic_domain_v1_descriptions.json `
  --output-dir data_samples `
  --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-m3 intfloat/multilingual-e5-large-instruct
```

Use `--allow-download` only after explicit approval and only when model size/resource implications are acceptable.

## Expected outputs

Prediction JSONL:

- preserves input fields;
- adds `semantic_topic_embedding.domain`;
- adds confidence, abstention, margin, top-k, model, method, and evidence.

Evaluation JSON:

- records total;
- answered/abstained counts;
- coverage;
- accuracy on answered;
- strict accuracy;
- top-k contains gold;
- macro-F1;
- per-domain precision/recall;
- per-dataset metrics;
- confusion matrix;
- mismatch examples.

## Recommended next step

First run MiniLM if the local environment can import `sentence_transformers` and `torch` while using the already-cached model. Then compare against BGE/E5/Qwen embedding candidates only after explicit approval for any missing environment setup or downloads.
