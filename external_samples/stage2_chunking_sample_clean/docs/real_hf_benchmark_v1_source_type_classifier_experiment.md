# Real HF benchmark v1 source_type classifier experiment

## Purpose

Taxonomy v2 improved semantic `domain` / `field` / `subfield` labels, but full-label accuracy stayed low because `source_type` remained weak.

This experiment treats `source_type` as its own genre/format/function classification task. It does not change the semantic taxonomy, existing rule-based classifier, semantic MiniLM pipeline, thresholds, chunks, or pseudo-gold labels.

## Why source_type is separate

`dataset` is provenance, such as FineWeb, FineWeb-Edu, or FineMath.

`domain` / `field` / `subfield` are semantic topic labels.

`source_type` is document genre or function: math problem, code/help page, educational lesson, reference text, news, product page, forum Q&A, legal/government text, web-general prose, or noise.

This separation matters for future NLL/probability profiling because model behavior may differ by format even when semantic topic is similar.

## Label set

Created:

- `taxonomy/source_type_labels.json`

Labels:

- `math`
- `code`
- `educational`
- `reference`
- `wiki_reference`
- `web_general`
- `news`
- `forum_qa`
- `commercial_product`
- `boilerplate_or_noise`
- `legal_government`
- `unknown`

Each label has a description, positive keywords, negative keywords, and examples. Descriptions focus on genre/format/function, not semantic topic.

## Scripts

Created:

- `scripts/classify_source_type_baseline.py`
- `scripts/evaluate_source_type_predictions.py`

Classifier modes:

- `lexical`: standard-library lexical nearest-label over source_type descriptions.
- `minilm`: MiniLM nearest-label over source_type descriptions.
- `hybrid`: conservative regex overrides for high-precision source types, otherwise MiniLM.

Output fields:

- `source_type`
- `source_type_confidence`
- `source_type_method`
- `source_type_top_k`

The script preserves all input fields and does not alter `domain`, `field`, or `subfield`.

## Experiment setup

Inputs:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_chunks.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_chunks.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_chunks.jsonl`

Pseudo-gold:

- `data_samples/real_hf_benchmark_v1_pseudo_gold_v2.jsonl`

Model:

- `C:\models\sentence-transformers\all-MiniLM-L6-v2`

Common settings:

- `--min-confidence 0.25`
- `--top-k 3`
- `--batch-size 32` for MiniLM modes

No HF streaming, dataset download, NLL/logits, semantic taxonomy tuning, or existing classifier changes were run.

## Outputs

Lexical:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_source_type_lexical.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_source_type_lexical.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_source_type_lexical.jsonl`

MiniLM:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_source_type_minilm.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_source_type_minilm.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_source_type_minilm.jsonl`

Hybrid:

- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_source_type_hybrid.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_fineweb_edu_source_type_hybrid.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_finemath_source_type_hybrid.jsonl`

Evaluation JSON:

- `data_samples/real_hf_benchmark_v1_source_type_eval_rule_based.json`
- `data_samples/real_hf_benchmark_v1_source_type_eval_lexical.json`
- `data_samples/real_hf_benchmark_v1_source_type_eval_minilm.json`
- `data_samples/real_hf_benchmark_v1_source_type_eval_hybrid.json`

## Metrics

| Classifier | Accuracy | Low-confidence | FineWeb | FineWeb-Edu | FineMath |
| --- | ---: | ---: | ---: | ---: | ---: |
| original rule-based source_type | 0.1833 | 0 | 0.0500 | 0.2500 | 0.2500 |
| lexical source_type | 0.0083 | 112 | 0.0000 | 0.0000 | 0.0250 |
| MiniLM source_type | 0.0833 | 103 | 0.0000 | 0.0250 | 0.2250 |
| hybrid source_type | 0.1417 | 63 | 0.0250 | 0.0000 | 0.4000 |

## Interpretation

The separate source_type MiniLM experiment did not beat the original rule-based source_type baseline overall.

MiniLM helped most on FineMath and found more `math`/`code`-like cases than lexical, but it still left 103 / 120 reviewed records low-confidence.

The hybrid source_type experiment reduced low-confidence count substantially, from 103 for MiniLM to 63, and improved FineMath accuracy to 0.4000. However, it hurt FineWeb and FineWeb-Edu, mainly because regex overrides were too eager on patent/reference text.

The original rule-based source_type remains the best overall baseline in this experiment, but its absolute accuracy is still low.

## Confusion analysis

### Original rule-based

Top mismatches:

- `reference -> unknown`: 25
- `educational -> unknown`: 19
- `math -> unknown`: 10
- `reference -> educational`: 9
- `web_general -> unknown`: 6

Rule-based source_type has some useful high-level signals, but misses most reference, web_general, news, forum, product, code, and noise cases.

### Lexical source_type

Top mismatches:

- `educational -> unknown`: 39
- `reference -> unknown`: 33
- `math -> unknown`: 14
- `web_general -> unknown`: 8
- `news -> unknown`: 7

Lexical source_type is too conservative at the current threshold and label wording.

### MiniLM source_type

Top mismatches:

- `educational -> unknown`: 34
- `reference -> unknown`: 30
- `math -> unknown`: 9
- `web_general -> unknown`: 9
- `news -> unknown`: 7

MiniLM source_type improves over lexical and detects some math/code cases, but still does not have enough confidence on most genre labels.

### Hybrid source_type

Top mismatches:

- `educational -> unknown`: 22
- `reference -> unknown`: 17
- `reference -> code`: 12
- `educational -> math`: 9
- `web_general -> unknown`: 8

Hybrid improves some math/code/noise/forum cases, but the current overrides are not high precision enough. Patent/reference texts often look technical and are incorrectly captured as `code` or `commercial_product`.

## Where MiniLM helps

MiniLM source_type helps most on FineMath:

- FineMath source_type accuracy: 0.2250
- It correctly catches some `math` and `code` cases that lexical misses.

However, it does not handle FineWeb or FineWeb-Edu well enough yet:

- FineWeb: 0.0000
- FineWeb-Edu: 0.0250

## Where rule overrides help

Hybrid overrides help on:

- obvious code-like command/help content;
- explicit forum Q&A markers;
- clear math notation;
- some boilerplate/noise cases.

But current overrides are too broad for:

- patent/reference descriptions with terms like interface, memory, terminal, function, display;
- educational statistics/science chunks with formulas, which become `math`;
- product-adjacent wording in reference/news contexts.

## Remaining errors

The main unresolved source_type boundaries are:

- `reference` vs `educational`;
- `reference` vs technical/code-like text;
- `math` vs educational math/statistics lessons;
- `web_general` vs educational/news/reference;
- FineWeb-Edu chunks that are not educational;
- FineMath chunks that are not math.

These are format/function distinctions, not semantic domain distinctions.

## Recommended next source_type strategy

Do not merge this source_type experiment into the final hybrid yet.

Recommended path:

1. Keep source_type as a separate task and metric.
2. Use the original rule-based source_type as the current overall baseline.
3. Treat MiniLM source_type as promising only for math/code-like cases, not yet as a full replacement.
4. Redesign source_type labels or simplify them before tuning thresholds.
5. If hybrid is retried, make regex overrides narrower and add explicit `reference`/`patent_reference` handling instead of letting technical text fall into `code`.

Possible MVP simplification:

- `math`
- `code`
- `educational_or_reference`
- `news`
- `forum_qa`
- `commercial_product`
- `boilerplate_or_noise`
- `legal_government`
- `web_general`
- `unknown`

This may be more stable than forcing a hard boundary between `reference` and `educational` on small chunks.

## What not to tune yet

- Do not lower source_type MiniLM threshold from this experiment alone.
- Do not merge current hybrid overrides into production.
- Do not change semantic taxonomy to fix source_type.
- Do not judge MiniLM semantic classifier by source_type errors.
- Do not use full-label accuracy as the only headline metric.

## Relation to future NLL/probability profiling

For NLL/logprob profiling, source_type should describe the shape and function of text: lesson, reference, Q&A, product page, code, math derivation, noisy page, and so on.

That is different from semantic topic. A math-flavored educational lesson and a terse math proof may both be STEM/mathematics but may produce different probability profiles. Keeping source_type separate will make later data selection and profiling cleaner.
