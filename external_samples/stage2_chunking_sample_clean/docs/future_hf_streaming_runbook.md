# Future HF streaming runbook

## Preconditions

- User explicitly allows network/HF.
- Dataset id, config, split, and text field are verified.
- `max_docs` is chosen.
- Output directory is chosen.
- Disk space is roughly checked.
- No push unless explicitly requested.

## Step 1: verify source

Use `config/dataset_sources.json` together with `docs/hf_dataset_verification_plan.md`. Confirm dataset id, config, split, license, and row schema before running anything.

## Step 2: dry-run command planning

```bash
python scripts\inspect_dataset_sources.py --registry config\dataset_sources.json
python scripts\plan_real_sample_run.py --registry config\dataset_sources.json --source fineweb_edu --max-docs 20
python scripts\plan_real_source_pipeline.py --registry config\dataset_sources.json --source fineweb_edu --max-docs 20
```

## Step 3: first tiny run

Run one source only with `max_docs=20`. Write outputs under `data_samples\real_samples`.

## Step 4: validate and inspect

```bash
python scripts\validate_chunks.py --input data_samples\real_samples\<source_id>_chunks.jsonl
python scripts\inspect_chunks.py --input data_samples\real_samples\<source_id>_chunks.jsonl --limit 30 --show-text
python scripts\classify_chunks_rule_based.py --input data_samples\real_samples\<source_id>_chunks.jsonl --output data_samples\real_samples\<source_id>_labeled_rule_based.jsonl
python scripts\classify_chunks_lexical_baseline.py --input data_samples\real_samples\<source_id>_chunks.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\real_samples\<source_id>_labeled_lexical.jsonl
python scripts\compare_label_runs.py --left data_samples\real_samples\<source_id>_labeled_rule_based.jsonl --right data_samples\real_samples\<source_id>_labeled_lexical.jsonl --left-name rule_based --right-name lexical
```

## Step 5: decide whether to keep sample

Commit a tiny sample only if size, privacy, and licensing are acceptable. Otherwise, commit only stats and a short report.

## Abort conditions

- Unexpected huge download.
- Missing text field.
- Bad schema.
- Too much boilerplate/noise.
- Licensing or privacy concern.
