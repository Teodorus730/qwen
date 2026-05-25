# Real source labeling plan

## Goal

Label future tiny real samples in a controlled way before any embedding or NLL work.

## Per-source workflow

1. Chunk the raw sample.
2. Run the rule-based classifier.
3. Run the lexical nearest-label baseline.
4. Compare rule-based and lexical outputs.
5. Manually review at least 20 examples.
6. Consider embedding labels only after the simple labels look sane.
7. Consider observed-token NLL/logprob scoring only after labels and source metadata are stable.

## Expected source-specific problems

- FineWeb: boilerplate, navigation, commercial pages, forum text, and useful articles are mixed.
- FineWeb-Edu: text should be educational, but domain may vary widely.
- FineMath/OpenWebMath: formulas, LaTeX, code snippets, and tokenization quirks can confuse simple rules.
- Cosmopedia/SmolLM: synthetic educational text may be smoother than real web data and may overstate classifier quality.
- General issue: `source_type` from the dataset is not the same thing as `domain/field/subfield`.

## Review standard

The first real sample should be treated as an inspection set, not a benchmark. Disagreements between rule-based and lexical labels should be reviewed manually before changing taxonomy or classifier rules.
