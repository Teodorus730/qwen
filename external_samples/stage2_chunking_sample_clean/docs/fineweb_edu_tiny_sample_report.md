# FineWeb-Edu tiny sample report

## Purpose

First controlled HF streaming sample attempt for FineWeb-Edu with `max_docs=20`.

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

## Result

The run did not reach dataset streaming. It failed locally because the `datasets` package is not installed in the active Python environment.

```text
ModuleNotFoundError: No module named 'datasets'
```

No alternative HF sources or configs were tried. No `pip install` was run.

## Chunking results

- docs_seen: not available
- docs_with_text: not available
- chunks_written: 0
- token_count min/mean/max: not available
- missing text field count: not available

The failed run left an empty output file at `data_samples\real_samples\fineweb_edu_sample10bt_chunks.jsonl`. No stats file was written.

## Labeling results

Rule-based and lexical labeling were not run for this HF sample because no chunks were created.

## Manual observations

No content was streamed, so there are no observations about text quality, boilerplate, chunk length, or schema.

## Limitations

- Only one controlled HF attempt was made.
- The attempt failed before network/dataset streaming.
- The environment needs `datasets` before HF streaming can run.
- No NLL/logprob scoring was run.

## Next recommendation

Install or provide `datasets` only after explicit approval, then retry the same single FineWeb-Edu command before trying FineMath, OpenWebMath, or any other source.
