# Real HF benchmark plan

Date: 2026-05-27

Scope: prepare a small real-data benchmark for checking rule-based, lexical, MiniLM, and hybrid labels on real chunks.

This plan does not make OpenWebMath a current source. OpenWebMath remains optional_later only.

## Purpose

Synthetic benchmark results are useful for regression checks, but they are not enough to claim that MiniLM or hybrid labeling works on real data.

The real HF benchmark should answer:

- do chunks from real datasets look usable;
- do rule-based labels overfit the synthetic benchmark;
- does MiniLM help on real text;
- does hybrid labeling preserve source/type metadata while improving semantic labels;
- which disagreements need manual pseudo-gold review.

## Why synthetic benchmark is not enough

The synthetic benchmark was designed around known labels and rule coverage. It is intentionally small and controlled.

Real data has:

- noisier formatting;
- boilerplate;
- mixed topics;
- short or fragmented text;
- metadata gaps;
- content that does not map cleanly to the current taxonomy.

Therefore benchmark accuracy on synthetic data should be treated as a local regression check, not as proof of real-world classifier quality.

## Datasets

Current MVP datasets:

| Dataset | Role | Current status |
| --- | --- | --- |
| FineWeb | small web-general comparison sample | current MVP real benchmark source |
| FineWeb-Edu | educational web sample | current MVP real benchmark source |
| FineMath | current MVP math source | current MVP real benchmark source |

Optional later only:

- OpenWebMath.

## Sampling strategy

Use Hugging Face streaming only for small samples:

- `streaming=True`;
- no full dataset download;
- inspect only the first `--stream-limit` examples;
- reservoir sample eligible rows from that prefix;
- fixed `--seed`;
- minimum text length filter;
- output local JSONL docs first;
- run the local pipeline only on those sampled docs.

Initial smoke sample:

```text
stream-limit: 50
sample-size: 3
seed: 42
```

Candidate first real benchmark after smoke:

```text
stream-limit: 500
sample-size: 30
seed: 42
```

Do not increase sample sizes before reviewing smoke outputs.

## Output files

Sampled docs:

```text
examples/real_hf_fineweb_docs.jsonl
examples/real_hf_fineweb_edu_docs.jsonl
examples/real_hf_finemath_docs.jsonl
```

Pipeline outputs under `data_samples/real_samples/`:

```text
real_hf_benchmark_<source>_chunks.jsonl
real_hf_benchmark_<source>_run_stats.json
real_hf_benchmark_<source>_labeled_rule_based.jsonl
real_hf_benchmark_<source>_labeled_lexical.jsonl
real_hf_benchmark_<source>_labeled_embedding_minilm.jsonl
real_hf_benchmark_<source>_labeled_hybrid.jsonl
```

Manual review candidates:

```text
data_samples/real_hf_review_candidates.jsonl
```

## Dev/test split

For the current MVP, keep the split simple:

- smoke: 3 docs per source, only to verify mechanics;
- dev: first reviewed small benchmark, about 30 sampled docs per source;
- test: another fixed-seed sample later, only after labels and criteria stabilize.

Do not tune thresholds on the final test sample.

## Pseudo-gold labels

Real samples do not have gold labels. They need pseudo-gold review fields:

- `expected_source_type`;
- `expected_domain`;
- `expected_field`;
- `expected_subfield`;
- `review_note`;
- `review_confidence`.

Pseudo-gold labels should be filled manually after looking at the text, not copied blindly from rule-based, lexical, MiniLM, or hybrid predictions.

## Label semantics

`dataset` means provenance:

- FineWeb;
- FineWeb-Edu;
- FineMath.

`source_type` means format/content type:

- web_general;
- educational;
- math;
- code;
- forum_qa;
- boilerplate_or_noise;
- unknown;
- etc.

`domain/field/subfield` means semantic topic:

- stem / mathematics / calculus;
- science / biology / article;
- web / boilerplate_or_navigation / page_noise;
- etc.

Hybrid labels should be evaluated against pseudo-gold labels. Agreement with rule-based or MiniLM alone is not accuracy.

## Risks and limitations

- HF streaming can fail because of network, dataset config, auth, or schema changes.
- First rows of a stream may not represent the full dataset.
- Reservoir sampling over the first `N` rows reduces order bias but does not remove all sampling bias.
- Tiny samples are for inspection, not final statistics.
- MiniLM confidence is cosine similarity, not calibrated probability.
- Pseudo-gold labels can be subjective.
- Current taxonomy may be too small for real data.
- FineMath may contain educational pages, formula-heavy pages, or web boilerplate around math content.
- FineWeb and FineWeb-Edu can overlap in style.

## Required sequence

1. Sample tiny docs with `sample_real_hf_docs.py`.
2. Inspect sampled docs.
3. Run local pipeline with `run_real_hf_benchmark_pipeline.py`.
4. Build review candidates.
5. Fill pseudo-gold labels.
6. Evaluate classifier/hybrid behavior against pseudo-gold labels.
7. Only then decide whether MiniLM/hybrid improves real labeling.
