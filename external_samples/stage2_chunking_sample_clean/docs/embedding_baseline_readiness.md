# Embedding baseline readiness

> Current status note, 2026-05-26: this is an older readiness note. Use `minilm_readiness_plan_ru.md` and `classifier_contract_ru.md` as the current MiniLM preparation docs. No dependency install, model download, or embedding inference is approved by this note.

## Current script

`scripts/classify_chunks_embedding_baseline.py` is an optional nearest-label baseline. It reads chunk JSONL, reads `taxonomy/simple_domain_labels.json`, builds label texts from domain/field/subfield/description/keywords, and can write embedding-based labels.

It is not a replacement for the rule-based baseline.

## Dependency needed

The real run needs `sentence-transformers` and a locally available model, for example `sentence-transformers/all-MiniLM-L6-v2` or a local path to the same kind of model.

## Why it is not run now

This session does not install packages, use the network, download models, or run model inference. The dry-run path is safe because it does not load a model and does not write output.

## Local files used

- input chunks: `data_samples/classifier_benchmark_chunks.jsonl`
- labels: `taxonomy/simple_domain_labels.json`
- possible output: `data_samples/classifier_benchmark_embedding_labeled.jsonl`

## Dry-run command

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\classifier_benchmark_embedding_labeled.jsonl --dry-run
```

## Real run later

Only run this if the dependency and model files are already available locally:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\classifier_benchmark_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\classifier_benchmark_embedding_labeled.jsonl --model sentence-transformers/all-MiniLM-L6-v2
```

## Risks

- accidental model download if the environment is not configured carefully;
- CPU speed may be slow;
- label descriptions may be too short for robust nearest-label matching;
- long chunks may need truncation decisions;
- cosine confidence is not calibrated probability;
- embedding labels may blur source_type and domain if interpreted too broadly.
