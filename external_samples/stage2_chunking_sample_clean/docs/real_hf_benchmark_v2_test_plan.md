# Real HF benchmark v2-test plan

## Purpose

`real_hf_benchmark_v2_test` is an independent held-out test split for annotation schema v2, especially the current weak `topic.domain` baseline.

It is needed because `real_hf_benchmark_v1` has become a dev benchmark:

- annotation_v2 pseudo-gold was created on v1;
- ambiguous v1 cases were manually reviewed;
- patched v1 pseudo-gold was created;
- `weak_topic_domain_v2.1` was improved after looking at v1 errors.

Therefore v1 metrics are useful for development history, but they are no longer an independent quality estimate.

## Datasets

Current MVP sources:

- FineWeb: `HuggingFaceFW/fineweb`, config `sample-10BT`, split `train`
- FineWeb-Edu: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split `train`
- FineMath: `HuggingFaceTB/finemath`, config `finemath-4plus`, split `train`

OpenWebMath remains optional later, not a required current source.

## Sample Size

Recommended held-out review target:

- minimum: 20 review records per dataset, 60 total;
- preferred if cheap: 30 review records per dataset, 90 total.

The sampled docs can be larger than the final review set because chunking may create uneven numbers of chunks per document.

## Sampling Policy

Held-out sampling should:

- use a different seed from v1, e.g. `2026` instead of `42`;
- use a different stream window/offset when possible;
- avoid reusing v1 document ids and chunk ids;
- keep `streaming=True`;
- keep stream windows small;
- never download full datasets.

Implementation policy:

- preserve dataset provenance labels (`FineWeb`, `FineWeb-Edu`, `FineMath`);
- use v2-test specific document ids and chunk id prefixes to avoid accidental overlap with v1;
- run overlap checks against v1 docs/chunks before labeling.

## Generated Files

Expected docs:

- `examples/real_hf_benchmark_v2_test_fineweb_docs.jsonl`
- `examples/real_hf_benchmark_v2_test_fineweb_edu_docs.jsonl`
- `examples/real_hf_benchmark_v2_test_finemath_docs.jsonl`

Expected local pipeline outputs:

- `data_samples/real_samples/real_hf_benchmark_v2_test_*_chunks.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_*_features_v2_refined.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_*_features_v2_tokenized.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v2_test_*_topic_domain_v2_1.jsonl`

Expected review candidates:

- `data_samples/real_hf_benchmark_v2_test_annotation_review_candidates.jsonl`

Expected overlap report:

- `data_samples/real_hf_benchmark_v2_test_overlap_check.json`

## Review Policy

The review candidate file should include:

- text and text preview;
- dataset and chunk id;
- deterministic annotation_v2 features;
- tokenization stats;
- `weak_topic_domain_v2.1` prediction as a hint;
- empty review fields for future pseudo-gold labeling.

Predictions are hints, not truth.

## No-Tuning Policy

Results on v2-test may be reported, but the classifier must not be tuned on v2-test in the same loop.

If classifier changes are made after seeing v2-test results, a later `v3-test` or final holdout should be created. This keeps at least one independent test set available.

