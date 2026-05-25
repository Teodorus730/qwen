# MiniLM readiness plan

Дата: 2026-05-26

This is a plan for the next stage2 milestone. It does not implement MiniLM and does not approve dependency or model downloads.

## Goal

Add an embedding-based nearest-label classifier using MiniLM/Sentence-Transformers style embeddings.

The classifier should label chunks with the same taxonomy used by the lexical baseline and produce outputs comparable with rule-based and lexical labels.

## Why MiniLM-style nearest-label now

MiniLM/Sentence-Transformers nearest-label classification is a good next step because:

- it is simpler than fine-tuning BERT;
- it can work with tiny samples;
- it reuses the current taxonomy;
- it gives a semantic baseline beyond keyword overlap;
- it supports disagreement review before heavier modeling.

Do not start with BERT fine-tuning in the current MVP because it needs labeled training data, training code, validation split decisions, more dependencies, and a larger project scope.

## Current script status

`scripts/classify_chunks_embedding_baseline.py` already exists as an optional embedding scaffold.

Known gaps before a real MiniLM run:

- dependency/model approval is not documented as a strict preflight;
- model loading must be local/no-download;
- batching and device settings need a final decision;
- output schema should match `classifier_contract_ru.md`;
- top-k/low-confidence policy should be explicit;
- mock tests would be useful before model inference;
- generated output names must not overwrite rule-based or lexical files.

## Inputs

Use existing stage2 inputs:

- chunks JSONL;
- `taxonomy/simple_domain_labels.json`;
- local MiniLM/Sentence-Transformers model path or verified local cache;
- classifier settings such as threshold, batch size, top-k.

Recommended first inputs:

1. `data_samples/classifier_benchmark_chunks.jsonl`
2. `data_samples/real_samples/fineweb_edu_sample10bt_chunks.jsonl`
3. `data_samples/real_samples/finemath_chunks.jsonl`

## Outputs

Recommended naming:

```text
data_samples/classifier_benchmark_labeled_embedding_minilm.jsonl
data_samples/real_samples/fineweb_edu_sample10bt_labeled_embedding_minilm.jsonl
data_samples/real_samples/finemath_labeled_embedding_minilm.jsonl
```

Recommended output metadata:

- `label_method`: `embedding_nearest_label_minilm`;
- `confidence`: cosine similarity or nearest-label score, not probability;
- `model_name` or model path;
- taxonomy path/version;
- threshold;
- top-k labels if implemented;
- low-confidence flag if implemented.

## No-download workflow

Before real inference:

1. Confirm dependency approval.
2. Confirm model approval.
3. Confirm model is already local or explicitly approved for download.
4. Configure the script to fail closed if the model is missing.
5. Run `--dry-run` first.
6. Run a tiny local benchmark before real samples.

No command should implicitly download a model during routine cleanup.

## Comparison plan

Run the same chunks through:

- rule-based classifier;
- lexical nearest-label classifier;
- embedding nearest-label classifier.

Compare:

- benchmark accuracy on synthetic data;
- agreement/disagreement on FineWeb-Edu;
- agreement/disagreement on FineMath;
- low-confidence examples;
- cases where embedding helps lexical;
- cases where embedding overgeneralizes.

## Pre-run checklist

- classifier contract reviewed;
- validation modes documented;
- source status table current;
- output naming agreed;
- local model/dependency approval explicit;
- no HF streaming involved;
- no NLL/logprob scoring involved;
- old outputs protected from overwrite.

## Post-run checks

- validate output schema;
- compare with rule-based outputs;
- compare with lexical outputs;
- inspect top disagreements;
- write a short report;
- decide whether outputs are temporary or promoted/tracked.
