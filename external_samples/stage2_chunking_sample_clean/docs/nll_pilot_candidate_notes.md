# NLL pilot candidate notes

## Purpose

`data_samples/real_samples/nll_pilot_candidates.jsonl` is a small manifest for a future observed-token NLL/logprob pilot. It is not a scoring output and no model inference has been run.

## Selection

The manifest contains 20 chunks total:

- 5 FineWeb-Edu chunks where rule-based and lexical labels agree.
- 5 FineWeb-Edu chunks where rule-based and lexical labels disagree.
- 5 FineMath chunks where rule-based and lexical labels agree.
- 5 FineMath chunks where rule-based and lexical labels disagree.

Each record keeps the chunk text because future scoring needs text input. It also stores both label tuples for later comparison.

## Why include disagreements

Agreement chunks provide stable baseline examples. Disagreement chunks are useful stress cases: they can reveal whether probability profiles separate clean educational/math text from noisy, commercial, boilerplate, or ambiguous text.

## Scoring constraints

The first pilot should score observed-token NLL/logprob only. Do not store full softmax distributions for the MVP. Keep the first run small and local, and do not treat these 20 chunks as a final evaluation set.
