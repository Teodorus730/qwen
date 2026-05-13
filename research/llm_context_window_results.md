# LLM Context Window Research Implementation Results

## Artifacts

- `research/llm_context_window_mecw.ipynb`
- `research/llm_context_window_results.md`

All generated files are inside `research`. The existing repository-root `hypothesis.md` is not modified. The notebook's SATLUTION demo writes only to `research/hypothesis.md` when it is executed and a MECW breach is detected.

## What Was Implemented

The notebook converts the supplied research report into five executable modules:

1. **Pimentel & Meister word probability correction**
   - Detects beginning-of-word vocabulary entries for GPT-style `Ġ`, SentencePiece `▁`, decoded whitespace prefixes, and WordPiece fallback conventions.
   - Implements the correction:
     `p(w | w_<t) = p(sw | sw_<t) * sum_s p(s | sw_<t o sw) / (sum_s p(s | sw_<t) + eps)`.
   - Uses vectorized PyTorch softmax, gather, product, and masked BoW probability sums.
   - Adds `1e-9` denominator protection and finite-value cleanup.

2. **Sampler-free confidence metrics**
   - Avoids stochastic sampling entirely.
   - Extracts logit entropy, normalized entropy, top-1 margin, logit IQR, hidden-state IQR, hidden trajectory energy, and inter-layer variance.
   - Combines these into a deterministic `ci_width_proxy` aligned with the report's Perception Tunnel and Structural Confidence discussion.

3. **Dynamic MECW detection**
   - Expands context in fixed `window_step` increments.
   - Runs deterministic forward passes with `output_hidden_states=True`.
   - Tracks moving average confidence-width proxy values.
   - Flags MECW breach when the variance of recent moving widths exceeds `epsilon`.
   - Can return either a boolean or a detailed diagnostic dictionary.

4. **N-gram perplexity elbow method**
   - Tokenizes domain strings with a portable regex tokenizer.
   - Filters stop words and builds unique n-grams, defaulting to trigrams.
   - Computes increasing-context cross entropy and perplexity.
   - Defaults to corrected word-level cross entropy using the Pimentel & Meister correction, with a faster token-level fallback available.
   - Finds the elbow using normalized numerical gradients and maximum curvature.
   - Plots the perplexity curve with a vertical dashed elbow marker.

5. **SATLUTION-style L2 state management**
   - Implements `AgentContextManager`.
   - Maintains active context tokens and notes.
   - Calls MECW detection on the active context.
   - On breach, writes a structured Markdown record with JSON payload to `research/hypothesis.md`.
   - Flushes active tokens and notes after externalizing the compressed state.

## Mapping to the Research Report

- The report's BoW marginalization error is represented directly by `compute_corrected_word_probability`.
- The report's warning about raw-logit overconfidence and the Perception Tunnel is reflected in `extract_confidence_metrics`, where logits are not treated as ground-truth confidence.
- The report's Structural Confidence preference is approximated by hidden-state IQR, token-to-token trajectory energy, and inter-layer hidden-state variance.
- The report's MECW definition is implemented as a dynamic degradation detector based on confidence-width instability rather than declared architectural context length.
- The report's domain adaptation procedure is represented by stop-word-filtered n-gram extraction, perplexity measurement over increasing context sizes, and second-derivative elbow localization.
- The report's SATLUTION/Kairos L0-L1-L2 pattern is implemented through a local L2 `hypothesis.md` buffer and clean-context reset behavior.

## Expected Notebook Outputs

When run top to bottom, the notebook should:

- Install missing dependencies if required.
- Load `gpt2` through HuggingFace.
- Print corrected word probability diagnostics, including BoW token count, BoW mass before and after the word, base subword probability, correction factor, and corrected probability.
- Print confidence metrics such as `ci_width_proxy`, entropy, margin, hidden IQR, and trajectory energy.
- Print MECW breach diagnostics, including breach flag, breach index, max variance, and threshold.
- Print n-gram counts, perplexity values, and the detected elbow point.
- Render a matplotlib perplexity-elbow plot.
- Write `research/hypothesis.md` only if the demo's MECW detector triggers.

## Operational Notes

- The notebook is deterministic with fixed seeds and no stochastic decoder.
- The dependency bootstrap installs `torch`, `transformers`, `numpy`, `scipy`, and `matplotlib` only when missing.
- GPT-2 download requires HuggingFace/network access on first run.
- Corrected word-level perplexity is mathematically closer to the report but slower than token-level loss because it evaluates words with aligned before/after BoW logits.
- The confidence interval is a structural proxy, not a trained ECE or quantile-regression calibration model.
- `AgentContextManager` refuses to write outside the resolved `research` directory.
- The notebook should be launched from the repository root or from `research`; otherwise it fails fast rather than writing artifacts to an unrelated directory.

## Validation Performed

- Parsed `research/llm_context_window_mecw.ipynb` as valid JSON.
- Confirmed notebook format `nbformat=4`, `nbformat_minor=5`.
- Confirmed the notebook contains 16 cells with the intended Markdown/code sequence.
- Compiled all code cells with Python AST parsing to catch syntax errors without importing unavailable dependencies or downloading models.
