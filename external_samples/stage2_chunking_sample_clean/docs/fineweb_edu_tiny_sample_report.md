# FineWeb-Edu tiny sample report

## Purpose

First real HF streaming sample for FineWeb-Edu with `max_docs=20`.

## Source

- dataset id: `HuggingFaceFW/fineweb-edu`
- config: `sample-10BT`
- split: `train`
- text_field: `text`
- max_docs: 20
- command used:

```bash
python scripts\sample_fineweb_chunks.py --use-hf-streaming --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --split train --dataset-label FineWeb-Edu --source-type educational --text-field text --max-docs 20 --out data_samples\real_samples\fineweb_edu_sample10bt_chunks.jsonl --stats-out data_samples\real_samples\fineweb_edu_sample10bt_run_stats.json
```

## Chunking results

- docs_seen: 20
- docs_with_text: 20
- docs_missing_text_field: 0
- chunks_written: 44
- token_count min/mean/max: 83 / 321.07 / 416
- stats path: `data_samples\real_samples\fineweb_edu_sample10bt_run_stats.json`

The run used a local Hugging Face cache under `data_samples\hf_cache_test` because the default user-home cache path was not writable in this environment.

## Labeling results

Rule-based output:

- records: 44
- source_type distribution: `educational=40`, `commercial_product=4`
- domain distribution: `null=25`, `education=14`, `commercial=4`, `science=1`
- label_method distribution: `rule_based_unknown=25`, `rule_based_keyword_v1=19`

Lexical output:

- records: 44
- source_type distribution: `educational=44`
- domain distribution: `null=36`, `software=3`, `education=2`, `science=2`, `commercial=1`
- label_method distribution: `lexical_nearest_label_low_confidence=36`, `lexical_nearest_label_v1=8`

Rule-based vs lexical comparison:

- common chunk_ids: 44
- matching full labels: 24
- differing full labels: 20
- agreement rate: 0.5455
- source_type disagreements: 4
- domain disagreements: 20
- field disagreements: 20
- subfield disagreements: 18

Strict `--require-labels` validation fails for both labeled outputs because real chunks can remain `domain/field=null` under `rule_based_unknown` or `lexical_nearest_label_low_confidence`. Non-strict schema validation passes.

## Manual observations

- The sample contains real educational web/article-like text, including literature commentary, health education, science/news-style articles, and software licensing guidance.
- FineWeb-Edu is not purely classroom text: some chunks look like news, public health guidance, legal/software FAQ material, and commercial or donation/program pages.
- The `text` field worked for all 20 streamed rows; no missing text field issue appeared.
- Chunks are longer than the synthetic benchmark average because real paragraphs often exceed the target chunk size; max observed chunk size was 416 fallback tokens.
- Some rule-based commercial labels are plausible but fragile on software licensing, product/program, pricing, or vendor-like language.
- Lexical nearest-label is conservative on real text: most chunks are low confidence.

## Limitations

- Only 20 documents.
- First streaming sanity check, not a representative benchmark.
- No human gold labels.
- No embedding classifier.
- No NLL/logprob scoring.
- Some outputs contain null domain/field/subfield by design when confidence is low or rules are unknown.

## Next recommendation

Review 20-30 chunks manually before changing rules. FineMath can be the next MVP source after review and explicit approval. OpenWebMath should remain optional later only. Do not start NLL/logprob scoring until the real-label behavior is accepted.
