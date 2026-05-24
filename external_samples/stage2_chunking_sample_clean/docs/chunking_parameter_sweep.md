# Chunking parameter sweep

Small local sweep on `examples/local_docs_classifier_benchmark.jsonl`. No network or HF streaming was used.

## Available chunker parameters

`scripts/sample_fineweb_chunks.py` currently supports:

- `--min-chunk-tokens`
- `--target-chunk-tokens`
- `--max-chunk-tokens`
- `--max-docs`

## Runs

### Default, max-docs 100

Command:

```bash
python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_classifier_benchmark.jsonl --max-docs 100 --out data_samples\classifier_benchmark_chunks.jsonl --stats-out data_samples\classifier_benchmark_run_stats.json
```

Result:

- docs_seen: 30
- chunks_written: 30
- min/mean/max token_count: 129 / 144.87 / 180
- expected labels preserved: yes
- classifier accuracy after rule updates: full_label_accuracy 1.0000
- comments: default settings produce one chunk per synthetic document after benchmark text expansion; no tiny chunks.

### max-docs 20

Command:

```bash
python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_classifier_benchmark.jsonl --max-docs 20 --out data_samples\classifier_benchmark_chunks_maxdocs20.jsonl --stats-out data_samples\classifier_benchmark_run_stats_maxdocs20.json
```

Result:

- docs_seen: 20
- chunks_written: 20
- min/mean/max token_count: 129 / 144.60 / 180
- expected labels preserved: yes
- comments: useful for faster spot checks, but excludes later hard cases.

### max-docs 40

Command:

```bash
python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_classifier_benchmark.jsonl --max-docs 40 --out data_samples\classifier_benchmark_chunks_maxdocs40.jsonl --stats-out data_samples\classifier_benchmark_run_stats_maxdocs40.json
```

Result:

- docs_seen: 30 because the benchmark has 30 rows
- chunks_written: 30
- min/mean/max token_count: 129 / 144.87 / 180
- expected labels preserved: yes
- comments: same coverage as max-docs 100 for the current benchmark.

### Smaller target chunks

Command:

```bash
python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_classifier_benchmark.jsonl --max-docs 100 --min-chunk-tokens 60 --target-chunk-tokens 120 --max-chunk-tokens 220 --out data_samples\classifier_benchmark_chunks_target120.jsonl --stats-out data_samples\classifier_benchmark_run_stats_target120.json
```

Result:

- docs_seen: 30
- chunks_written: 54
- min/mean/max token_count: 60 / 75.72 / 118
- expected labels preserved: yes, present on all 54 chunks
- classifier accuracy: source_type_accuracy 0.6481, full_label_accuracy 0.4630
- comments: this setting splits benchmark context paragraphs away from the category-specific paragraphs. That is useful as a stress test, but many second chunks become generic benchmark boilerplate while still carrying document-level expected labels.

## Notes

The CLI already has the requested chunk-size parameters, so no refactor was needed. The target-120 run shows a benchmark limitation: document-level expected labels are copied to every chunk, which can be unfair when generic trailing paragraphs are split into separate chunks. Future benchmark rows should avoid generic repeated filler or add chunk-level expected labels after splitting.
