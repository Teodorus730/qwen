# Probability profile schema

This document sketches future outputs for reference/teacher model scoring. It does not imply model inference is implemented now.

## Chunk-level scoring fields

- `chunk_id`
- `dataset`
- `source_type`
- `domain`
- `field`
- `subfield`
- `token_count`
- `model_name`
- `tokenizer_name`
- `window_mode`: `effective`, `full`, or `grid`
- `window_size`
- `stride`
- `tokens_scored`
- `mean_nll`
- `median_nll`
- `p90_nll`
- `p95_nll`
- `mean_logprob`
- `perplexity`
- `nll_histogram_bins`
- `nll_histogram_counts`
- `scoring_method`

## Comparison fields

- `chunk_id`
- `model_name`
- `effective_window_size`
- `full_window_cap`
- `nll_eff_mean`
- `nll_full_mean`
- `delta_mean`
- `delta_median`
- `delta_p90`
- `perplexity_ratio`
- `js_distance_hist`
- `wasserstein_distance`
- `notes`

## Important notes

- Do not store full softmax distributions for the MVP.
- Store observed-token logprobs/NLL.
- Bootstrap confidence intervals should be by chunk or document, not by token.
- Effective window comes from task 1.
- Full window means full prefix capped by model context.
- Perplexity should be interpreted domain-wise.
- Very low perplexity can mean boilerplate or repetition.
- Very high perplexity can mean noise or useful rare domain text.
