# HF row adapter design

## Problem

Different HF datasets may expose document text under different fields, for example `text`, `content`, `markdown`, `html`, `document`, `generated_text`, or nested fields. The current local JSONL convention uses `text`, while `sample_fineweb_chunks.py` currently guesses among a small set of flat fields.

## Future design

- Add `--text-field`, defaulting to `text`.
- Add `--id-field`, defaulting to no explicit id field.
- Optionally add `--fallback-text-fields text,content,body`.
- Allow `config/dataset_sources.json` to store `text_field` and `id_field`.
- Preserve output metadata: `dataset`, `source_type`, and later `source_id` if added to the chunk schema.
- Fail loudly if the configured text field is missing or empty for all rows.
- Do not silently concatenate unrelated fields.
- Keep local JSONL support simple: local rows continue to use `text`.

## Pseudo-interface

```bash
python scripts\sample_fineweb_chunks.py --use-hf-streaming --dataset ... --config ... --split train --text-field text --id-field id --max-docs 20 --dataset-label ... --source-type ...
```

## Notes

This is design-only. The adapter should be implemented only after one online schema verification pass confirms the actual row fields for the selected dataset.
