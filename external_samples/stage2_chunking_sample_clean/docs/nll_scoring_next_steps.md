# Observed-token NLL scoring next steps

## Goal

The next stage should score labeled chunks with a reference model and store observed-token logprob/NLL aggregates. This remains separate from classification.

## Proposed input

- `data_samples/classifier_benchmark_labeled.jsonl` for local smoke tests.
- Later, small real sampled chunks only after the local path is stable.
- First real pilot candidate manifest: `data_samples/real_samples/nll_pilot_candidates.jsonl`.

## Reference model

Likely first candidate: `Qwen/Qwen2.5-0.5B` base or another small Qwen-like base model. Base vs instruct should be decided explicitly because instruction tuning can affect perplexity profiles.

## Output

Store aggregate observed-token metrics:

- `mean_nll`
- `median_nll`
- `p90_nll`
- `p95_nll`
- `mean_logprob`
- `perplexity`
- histogram bins/counts

Do not store full softmax distributions for the MVP.

## Window plan

Start with grid windows:

- 256
- 512
- 1024
- 2048

Later, the effective window should come from task 1. Full/adaptive window means full prefix capped by the model context limit.

## First implementation scope

The first real scoring run should use only 10-50 chunks. It should not run on full FineWeb.
For the first real pilot, use `data_samples/real_samples/nll_pilot_candidates.jsonl`: it includes both FineWeb-Edu and FineMath chunks, with agreement and disagreement cases from rule-based vs lexical labels.

Classification and model scoring should stay separate:

- classifier decides source_type/domain/field/subfield;
- scorer computes observed-token NLL/logprob for already selected chunks.

## Pseudo-CLI

```bash
python scripts/score_observed_token_nll.py --input data_samples/classifier_benchmark_labeled.jsonl --output data_samples/nll_scores_sample.jsonl --model Qwen/Qwen2.5-0.5B --window-sizes 256,512,1024 --max-chunks 20
python scripts/score_observed_token_nll.py --input data_samples/real_samples/nll_pilot_candidates.jsonl --output data_samples/real_samples/nll_pilot_scores.jsonl --model Qwen/Qwen2.5-0.5B --window-sizes 256,512,1024 --max-chunks 20
```

Do not use the candidate manifest as final evaluation; it is a small pilot set for checking the scoring path and comparing FineWeb-Edu vs FineMath behavior.

## Risks

- CPU inference may be too slow;
- tokenizer/model download must be avoided unless explicitly allowed;
- context length and memory can become limiting;
- dependency versions may differ across machines;
- base vs instruct model choice can change interpretation;
- high NLL can mean rare useful domain text or noise;
- low NLL can mean boilerplate/repetition rather than high-quality data.
