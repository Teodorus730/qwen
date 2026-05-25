# Real sample next plan

## Goal

Move from synthetic benchmark checks to tiny real samples while keeping the pipeline small, inspectable, and safe.

## Candidate datasets

- FineWeb
- FineWeb-Edu
- FineMath
- OpenWebMath later only as optional comparison/backup, not an active current MVP source
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

Use `config/dataset_sources.json` as the planning registry. Inspect planned sources first:

```bash
python scripts\inspect_dataset_sources.py --registry config\dataset_sources.json
python scripts\plan_real_sample_run.py --registry config\dataset_sources.json --all --max-docs 20
python scripts\plan_real_source_pipeline.py --registry config\dataset_sources.json --source fineweb_edu --max-docs 20
```

HF commands are not yet run:

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

## Related docs

- `docs/dataset_source_registry.md`
- `docs/hf_dataset_verification_plan.md`
- `docs/hf_row_adapter_design.md`
- `docs/future_hf_streaming_runbook.md`
- `docs/real_source_labeling_plan.md`
- `docs/fineweb_edu_tiny_sample_report.md`
- `docs/finemath_tiny_sample_plan.md`
- `docs/real_sample_readiness_checklist.md`
- `docs/real_samples_output_structure.md`
- `docs/local_real_like_sample_report.md`

## First FineWeb-Edu sample status

A controlled `FineWeb-Edu` tiny sample completed with `max_docs=20`. It produced `data_samples\real_samples\fineweb_edu_sample10bt_chunks.jsonl`, rule-based labels, lexical labels, and a short report. Review those outputs manually before trying FineMath. OpenWebMath remains optional later only.

## FineMath next plan

The FineMath tiny sample completed with `max_docs=20`. It produced `data_samples\real_samples\finemath_chunks.jsonl`, rule-based labels, lexical labels, and a short report. See `docs/finemath_tiny_sample_report.md`. OpenWebMath remains optional later only.
