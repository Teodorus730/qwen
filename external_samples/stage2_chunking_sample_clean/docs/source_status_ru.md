# Source status

Дата: 2026-05-26

This file is the current stage2 source-status reference. It summarizes which sources are active, completed, optional, or downstream.

## Current source table

| Source | Status | Role | Notes |
| --- | --- | --- | --- |
| `local_edge_cases` | active local fixture | smoke testing | Synthetic/local edge cases, safe to run without network |
| `local_classifier_benchmark` | active synthetic benchmark | classifier evaluation | Has expected labels; accuracy is meaningful |
| FineWeb-Edu | tiny sample completed | educational real-sample review | Completed with `max_docs=20`; use agreement/disagreement, not accuracy |
| FineMath | tiny sample completed | current MVP math source | The only current MVP math dataset |
| OpenWebMath | optional_later | possible later comparison/backup | Do not run in current cleanup/MVP stage |
| FineWeb general | not current | possible later web-general sample | Requires approval and source verification |
| Cosmopedia/SmolLM | optional/later | possible synthetic educational source | Requires approval and source verification |
| NLL/effective context | downstream | not a data source for current stage2 | Keep separate from current corpus labeling milestone |

## FineWeb-Edu

Current status:

- tiny real sample completed;
- outputs live under `data_samples/real_samples/`;
- intended for review of chunk quality and classifier disagreements;
- not a gold-labeled accuracy benchmark.

Main files:

- `data_samples/real_samples/fineweb_edu_sample10bt_chunks.jsonl`
- `data_samples/real_samples/fineweb_edu_sample10bt_labeled_rule_based.jsonl`
- `data_samples/real_samples/fineweb_edu_sample10bt_labeled_lexical.jsonl`
- `data_samples/real_samples/fineweb_edu_sample10bt_run_stats.json`
- `docs/fineweb_edu_tiny_sample_report.md`

## FineMath

Current status:

- tiny real sample completed;
- current MVP math source;
- use this instead of OpenWebMath for the current math path;
- not a gold-labeled accuracy benchmark.

Main files:

- `data_samples/real_samples/finemath_chunks.jsonl`
- `data_samples/real_samples/finemath_labeled_rule_based.jsonl`
- `data_samples/real_samples/finemath_labeled_lexical.jsonl`
- `data_samples/real_samples/finemath_run_stats.json`
- `docs/finemath_tiny_sample_report.md`

## Optional later sources

OpenWebMath, FineWeb general, Cosmopedia, and SmolLM should not be promoted by accident. They need:

- explicit approval;
- dataset id/config/split verification;
- text field verification;
- license/privacy review;
- tiny max-docs run first;
- documented output naming.

## Source metadata vs predicted labels

`dataset` and `source_type` describe where a chunk came from.

`domain`, `field`, and `subfield` are predicted labels from classifiers.

Classifiers should not silently rewrite dataset identity. The detailed classifier rules are in `classifier_contract_ru.md`.
