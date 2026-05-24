# Real sample next plan

## Goal

Move from synthetic benchmark checks to tiny real samples while keeping the pipeline small, inspectable, and safe.

## Candidate datasets

- FineWeb
- FineWeb-Edu
- FineMath / OpenWebMath
- Cosmopedia / SmolLM-Corpus

## First safe target

Later, use only 20-100 documents per dataset for the first real sample. Start with one dataset and inspect outputs manually before expanding.

## Why not now

- local stage just stabilized;
- network/HF access needs explicit user approval;
- dataset names and configs need verification before use;
- avoid unexpected downloads;
- generated samples may be noisy and should be reviewed before committing.

## Proposed future commands

Not yet run:

```bash
python scripts\sample_fineweb_chunks.py --use-hf-streaming --dataset HuggingFaceFW/fineweb --config sample-10BT --max-docs 50 --out data_samples\real_small_chunks.jsonl --stats-out data_samples\real_small_run_stats.json
python scripts\classify_chunks_rule_based.py --input data_samples\real_small_chunks.jsonl --output data_samples\real_small_labeled_rule_based.jsonl
python scripts\inspect_chunks.py --input data_samples\real_small_labeled_rule_based.jsonl --limit 30 --show-text
```

Optional embedding labeling later, only if dependencies/model are local:

```bash
python scripts\classify_chunks_embedding_baseline.py --input data_samples\real_small_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\real_small_labeled_embedding.jsonl --model sentence-transformers/all-MiniLM-L6-v2
```

## Expected outputs

- `data_samples/real_small_chunks.jsonl`
- `data_samples/real_small_run_stats.json`
- `data_samples/real_small_labeled_rule_based.jsonl`
- `data_samples/real_small_labeled_embedding.jsonl` later

## Review questions before running

- Which datasets exactly?
- Which configs/splits?
- Maximum documents per dataset?
- Is network/HF access allowed for this run?
- Should real samples be committed or kept local?
- Should embedding labels wait until rule-based labels are manually reviewed?
