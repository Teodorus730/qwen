# Real samples output structure

Future tiny real samples should live under:

```text
data_samples/real_samples/
  fineweb_edu_chunks.jsonl
  fineweb_edu_labeled_rule_based.jsonl
  fineweb_edu_labeled_lexical.jsonl
  fineweb_edu_run_stats.json
```

## Naming convention

- `<source_id>_chunks.jsonl`
- `<source_id>_run_stats.json`
- `<source_id>_labeled_rule_based.jsonl`
- `<source_id>_labeled_lexical.jsonl`
- `<source_id>_labeled_embedding.jsonl` later, only if embedding is available locally

## Rules

- Keep real samples tiny.
- Include `source_id` indirectly through stable `dataset`/`dataset_label` and file names.
- Include `dataset` and `source_type` in every chunk record.
- Label outputs should not overwrite raw chunk outputs.
- Generated real samples may or may not be committed depending on size, privacy, licensing, and reproducibility.
- Never commit large data dumps.
- Prefer reports/stat summaries over large raw web samples if size grows.

## Commit policy

Commit tiny samples only when they are useful for reproducible tests and safe to share. Otherwise keep generated real samples local and commit only scripts, configs, stats, and reports.

## Privacy and licensing caution

Real web samples may contain personal data, copyrighted text, or license restrictions. Review before committing or sharing.

## Reproducibility notes

Record:

- source id;
- dataset id;
- config;
- split;
- max docs;
- command used;
- date;
- output paths.
