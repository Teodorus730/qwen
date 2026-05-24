# Local real-like sample report

## Purpose

`examples/local_real_like_mini_sample.jsonl` is a tiny local dry run for the future real-sample pipeline. It is not a real HF sample and does not use the network. It tests whether the current chunking, rule-based labeling, lexical labeling, validation, inspection, and comparison tools can handle unlabeled real-like metadata.

## Data

- source documents: 10
- chunks written: 10
- token_count min/mean/max: 80 / 91.00 / 107
- dataset label: `local_real_like`
- expected labels: not included, to mimic unlabeled real data

## Source types in input

- `web_general`
- `educational`
- `math`
- `code`
- `commercial_product`
- `forum_qa`
- `news`
- `wiki_reference`
- `legal_government`
- `unknown`

## Rule-based labels summary

Rule-based output source_type counts:

- `educational`: 2
- `code`: 1
- `commercial_product`: 1
- `forum_qa`: 1
- `legal_government`: 1
- `math`: 1
- `news`: 1
- `unknown`: 1
- `wiki_reference`: 1

Rule-based output domain counts:

- `education`: 2
- `commercial`: 1
- `government`: 1
- `media`: 1
- `multilingual`: 1
- `reference`: 1
- `software`: 1
- `stem`: 1
- `web`: 1

## Lexical labels summary

Lexical output source_type counts:

- `code`: 1
- `commercial_product`: 1
- `educational`: 1
- `forum_qa`: 1
- `legal_government`: 1
- `math`: 1
- `news`: 1
- `unknown`: 1
- `web_general`: 1
- `wiki_reference`: 1

Lexical output domain counts:

- `web`: 2
- `commercial`: 1
- `education`: 1
- `government`: 1
- `media`: 1
- `multilingual`: 1
- `reference`: 1
- `software`: 1
- `stem`: 1

## Rule-vs-lexical agreement

- common chunk_ids: 10
- matching full labels: 8
- differing full labels: 2
- agreement rate: 0.8000

Disagreements:

- `local_real_like_000000_000`: rule-based labels a general community article as `education/general_education/article`; lexical maps the footer-heavy web article to `web/boilerplate_or_navigation/page_noise`.
- `local_real_like_000005_000`: rule-based labels Q&A as `web/forum_qa/discussion`; lexical maps it to `web/boilerplate_or_navigation/page_noise` because the question discusses cookie/footer noise.

## What this tells us before real HF sample

- The local real-like pipeline works end to end without expected labels.
- Rule-based and lexical labels can disagree in useful ways.
- Lexical nearest-label is sensitive to boilerplate words in otherwise useful text.
- Manual review is needed before interpreting real sample labels as data quality signals.

## Limitations

- This is still hand-authored local text, not real web data.
- Only 10 documents are included.
- No embedding model was run.
- No HF streaming, dataset download, or model inference was used.
