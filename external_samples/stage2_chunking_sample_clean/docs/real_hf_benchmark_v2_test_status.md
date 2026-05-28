# real_hf_benchmark_v2_test status

## Purpose

`real_hf_benchmark_v2_test` is a held-out test split for the annotation_v2
topic-domain baseline. The earlier `real_hf_benchmark_v1` split is now a dev
benchmark because it was used for pseudo-gold review, taxonomy/source-type
analysis, patched annotation_v2 gold, and weak_topic_domain_v2.1 rule choices.

This split must not be used for further tuning. Results may be reported, but
classifier changes after inspecting v2-test results require a later independent
holdout.

## Sampling status

HF streaming succeeded for the three MVP sources using small streaming samples:

| Dataset | HF dataset/config | Split | Seed | Skip | Stream limit | Docs written |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FineWeb | `HuggingFaceFW/fineweb`, `sample-10BT` | train | 2026 | 1000 | 1000 | 30 |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | train | 2026 | 1000 | 1000 | 30 |
| FineMath | `HuggingFaceTB/finemath`, `finemath-4plus` | train | 2026 | 1000 | 1000 | 30 |

Sampling used `source_type=unknown` for all datasets. The dataset label is
provenance only, not a topic or source-type label.

The sampler used the local stage2 HF cache directory `.hf_dataset_cache/`.
That cache is an environment artifact and should not be committed.

## Generated files

Sampled docs:

- `examples/real_hf_benchmark_v2_test_fineweb_docs.jsonl`
- `examples/real_hf_benchmark_v2_test_fineweb_edu_docs.jsonl`
- `examples/real_hf_benchmark_v2_test_finemath_docs.jsonl`

Pipeline outputs:

| Dataset | Chunks | Refined features | Tokenized features | Topic v2.1 predictions | Review candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 52 | 52 | 52 | 52 | 30 |
| FineWeb-Edu | 119 | 119 | 119 | 119 | 30 |
| FineMath | 245 | 245 | 245 | 245 | 30 |

Output paths:

- `data_samples/real_samples/real_hf_benchmark_v2_test_*_chunks.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_*_run_stats.json`
- `data_samples/real_samples/real_hf_benchmark_v2_test_*_features_v2_refined.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_*_features_v2_tokenized.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_*_topic_domain_v2_1.jsonl`
- `data_samples/real_hf_benchmark_v2_test_*_features_v2_tokenized_validation.json`
- `data_samples/real_hf_benchmark_v2_test_annotation_review_candidates.jsonl`

## Validation

The three tokenized feature files passed `validate_annotation_features_v2.py`
with zero errors and zero warnings.

Noise-level counts:

- FineWeb: `clean=43`, `partial_noise=9`
- FineWeb-Edu: `clean=112`, `partial_noise=7`
- FineMath: `clean=236`, `partial_noise=9`

Topic v2.1 predictions were generated for review hints only. They must not be
treated as pseudo-gold:

- FineWeb: 52 records, 14 abstained
- FineWeb-Edu: 119 records, 24 abstained
- FineMath: 245 records, 53 abstained

## Overlap check

Overlap check output:

- `data_samples/real_hf_benchmark_v2_test_overlap_check.json`

Result:

- v1 doc ids: 90
- v2-test doc ids: 90
- doc id overlap: 0
- v1 chunk ids: 396
- v2-test chunk ids: 416
- chunk id overlap: 0

The split is ID-distinct from `real_hf_benchmark_v1`.

## Next step

Create annotation_v2 pseudo-gold labels for
`data_samples/real_hf_benchmark_v2_test_annotation_review_candidates.jsonl`.
The review file contains 90 balanced records, 30 per dataset, with v2.1
prediction hints and empty review fields.

Do not tune `weak_topic_domain_v2.1` on this split. After pseudo-gold labeling,
evaluate once and report held-out metrics.
