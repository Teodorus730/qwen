# Annotation schema v2 domain layer readiness report

## Executive summary

The annotation_v2 domain layer is ready for pilot NLL/probability profiling, with clear caveats.

Ready:

- provenance fields;
- deterministic text/surface features;
- refined quality/noise features;
- Qwen3.5 tokenizer-aware text statistics;
- `weak_topic_domain_v2.1` as a transparent baseline with confidence and abstention;
- v1-dev and v2-test pseudo-gold references for measuring coarse `topic.domain`.

Not final:

- `topic.domain` is not reliable enough to treat as ground truth for all chunks;
- FineWeb topic labels are especially weak;
- old single-label `source_type` should remain exploratory/deprecated as a central KPI;
- v2-test must not be used for more rule tuning.

## Project assumptions

The project trains a 0.8B Qwen-like / Qwen-architecture model from scratch. It does not use Qwen pretrained weights, continued pretraining from Qwen weights, or fine-tuning of an existing Qwen checkpoint.

The tokenizer is a separate frozen infrastructure choice. The current MVP tokenizer is:

```text
Qwen/Qwen3.5-0.8B-Base
revision: dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
tokenizer_class: Qwen2Tokenizer
vocab_size: 248044
len_tokenizer: 248077
eos_token_id: 248044
pad_token_id: 248044
```

The downstream goal is NLL/logprob/perplexity profiling over known data groups. Stage2 should therefore provide reproducible corpus metadata and weak labels, not final scientific claims about model behavior.

## What was built

Annotation schema v2 now includes:

- `provenance`: dataset/source metadata;
- `text_stats`: deterministic character/byte/line/rough word stats;
- tokenizer stats: Qwen3.5 token count and density fields;
- `surface`: math/code/formula/symbol/API/table/link/boilerplate features;
- `quality`: noise level, noise score, explicit residue/noise reasons;
- `topic.domain`: weak coarse domain labels with confidence, abstention, top-k, and evidence.

Benchmark and review artifacts:

- `real_hf_benchmark_v1`: dev benchmark, later used for pseudo-gold, manual review, patched gold, and v2.1 rule selection;
- `real_hf_benchmark_v2_test`: held-out test benchmark, 90 reviewed records with zero v1 overlap by doc_id/chunk_id;
- annotation_v2 pseudo-gold for v1-dev and v2-test;
- patched v1-dev pseudo-gold after targeted manual review;
- held-out v2-test evaluation of `weak_topic_domain_v2.1`.

## Metrics

Main metrics use `accuracy_on_answered + coverage`. Strict accuracy counts abstentions as wrong and is useful as a conservative diagnostic.

| Split | Records | Coverage | Accuracy on answered | Strict accuracy |
| --- | ---: | ---: | ---: | ---: |
| v1 patched dev | 120 | 0.9083 | 0.8165 | 0.7417 |
| v2-test held-out | 90 | 0.7889 | 0.6197 | 0.4889 |

Per-dataset v2-test metrics:

| Dataset | Records | Coverage | Accuracy on answered | Strict accuracy |
| --- | ---: | ---: | ---: | ---: |
| FineWeb | 30 | 0.7333 | 0.3182 | 0.2333 |
| FineWeb-Edu | 30 | 0.8667 | 0.7692 | 0.6667 |
| FineMath | 30 | 0.7667 | 0.7391 | 0.5667 |

## Interpretation

`weak_topic_domain_v2.1` is useful as a transparent MVP baseline. It gives interpretable evidence, exposes abstention, and works reasonably on FineWeb-Edu and FineMath.

The held-out drop from v1-dev to v2-test is substantial:

- coverage dropped from 0.9083 to 0.7889;
- accuracy on answered dropped from 0.8165 to 0.6197;
- strict accuracy dropped from 0.7417 to 0.4889.

That gap means the keyword/surface/prior baseline overfits or undergeneralizes. The v2-test result is the healthier number to cite when discussing generalization.

FineWeb is the main failure zone. It contains commercial pages, legal/service terms, media snippets, adult/SEO spam, product listings, rescue pages, and lifestyle content. Keyword-only rules are brittle in this mixture.

FineWeb-Edu and FineMath are more usable:

- FineWeb-Edu works well for science/humanities/education-like material, but still confuses legal/government, history/social content, and science/stem boundaries.
- FineMath works well for math/stem passages, but code snippets and technology/electronics passages are often pulled into `stem` by matrix/function/provenance evidence.

## Recommended usage for NLL/profiling

Use confidently:

- `provenance.dataset` and related source metadata;
- deterministic text statistics;
- Qwen3.5 tokenization statistics;
- surface flags such as math notation, code, symbol-heavy, formula/API signals;
- quality/noise fields, with review caveats;
- `topic.domain` only together with confidence, abstention status, and method metadata.

Use cautiously:

- `topic.domain` for answered records only;
- FineWeb topic labels, especially commercial/media/unknown/social boundaries;
- noise level as a grouping axis, not as perfect cleanliness truth.

Do not use as final truth:

- `topic.domain` for all chunks without confidence/abstention filtering;
- FineWeb topic labels as a high-confidence corpus truth source;
- old `source_type` as the central quality metric;
- full-label accuracy from the old source_type/domain/field/subfield schema as a primary KPI.

## MVP readiness

The domain layer is ready for pilot NLL/probability profiling if the pilot treats labels as weak metadata:

- group by dataset/provenance first;
- use deterministic features and token stats as stable axes;
- use `topic.domain` as a weak coarse grouping signal;
- report coverage and abstention whenever using topic labels;
- separate FineWeb from FineWeb-Edu/FineMath in analysis.

It is not ready for final claims about domain behavior or corpus quality. Any result depending heavily on FineWeb topic labels should be framed as exploratory.

## Next research steps

1. Embedding-model bake-off for `topic.domain`.
   Compare the current weak keyword baseline against embedding classifiers on the same coarse domains. Do not tune on v2-test; use v2-test only as reported held-out evidence.

2. Larger held-out sample or v3 final holdout.
   If the project needs stronger claims, create a fresh holdout after the next classifier design step. Do not recycle v2-test as a tuning set.

3. FineWeb-specific noise/commercial detection.
   FineWeb failures suggest a separate robustness layer for spam, adult SEO, legal/service pages, and commercial product content. This should be treated as a new dev experiment, not v2-test tuning.

4. NLL/probability profiling interface.
   Once grouping policy is accepted, add a small interface that consumes annotation_v2 metadata and emits profiling groups without running model logits yet.

## Commit and reproducibility notes

Important references:

- `docs/tokenizer_policy_from_scratch.md`
- `docs/annotation_schema_v2_feature_refinement_report.md`
- `docs/annotation_schema_v2_tokenization_stats_report.md`
- `docs/annotation_schema_v2_pseudo_gold_patch_report.md`
- `docs/weak_topic_domain_v2_1_eval_report.md`
- `docs/real_hf_benchmark_v2_test_topic_domain_eval_report.md`
- `data_samples/real_hf_benchmark_v1_topic_domain_v2_1_eval_patched_summary.json`
- `data_samples/real_hf_benchmark_v2_test_topic_domain_v2_1_eval_summary.json`

`real_hf_benchmark_v1` is dev. `real_hf_benchmark_v2_test` is held-out and must remain held-out. If classifier rules change after reading v2-test results, a later v3/final holdout is required for independent quality claims.
