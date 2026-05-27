# Real HF benchmark v1 decision report

## 1. What was tested

The stage2 benchmark work tested the corpus/domain-labeling stack on real small Hugging Face samples:

- FineWeb
- FineWeb-Edu
- FineMath

The benchmark path was:

1. sample real docs with small HF streaming/reservoir sampling;
2. chunk docs;
3. run rule-based, lexical, MiniLM, and hybrid labeling;
4. create 120 pseudo-gold review records;
5. audit pseudo-gold and evaluation logic;
6. run taxonomy v2 coverage experiment;
7. run separate source_type classifier experiment.

No NLL/logprob scoring or model training is part of this stage.

## 2. Synthetic vs Real Benchmark Lesson

The synthetic benchmark was useful for checking that the pipeline works, schemas are stable, and scripts can run end to end.

The real benchmark showed that synthetic performance did not transfer directly to real web chunks. Real data contained:

- patent-style technical reference text;
- statistics and probability;
- arithmetic and measurement conversions;
- chemistry tutorials;
- geometry and topology;
- art history and social issues;
- noisy or SEO-like math text;
- FineWeb-Edu chunks that were not actually educational;
- FineMath chunks that were not purely math.

The main lesson is that synthetic benchmark accuracy should be treated as a smoke/regression signal, not as evidence that the classifiers are ready for real corpus filtering.

## 3. Taxonomy v1 Failure

Taxonomy v1 was too small for the real benchmark.

Before taxonomy v2, the results looked like:

| Classifier | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule-based | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.1000 |
| lexical | 0.0833 | 0.0750 | 0.0667 | 0.1167 | 0.0583 |
| MiniLM | 0.0000 | 0.0000 | 0.0000 | 0.0250 | 0.0000 |
| hybrid | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.1000 |

This could have been misread as "MiniLM is useless" or "hybrid is useless".

The taxonomy audit showed a different cause: many chunks had no appropriate label to land on. Classifiers were forced into overly broad or unrelated labels, or they stayed low-confidence.

## 4. Taxonomy v2 Improvement

Taxonomy v2 added a controlled set of missing semantic labels, especially:

- `technology / patents / POS_systems`
- `technology / patents / hardware_networking`
- `stem / mathematics / statistics`
- `stem / mathematics / arithmetic_measurement`
- `science / chemistry / article`
- `stem / mathematics / geometry`
- `stem / mathematics / topology`
- `humanities / art_history / article`
- `social_sciences / social_issues / article`

After taxonomy v2, semantic metrics improved:

| Classifier | Source type | Domain | Field | Subfield | Full label |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule-based | 0.1833 | 0.1417 | 0.1417 | 0.2333 | 0.0750 |
| lexical v2 | 0.0750 | 0.3667 | 0.3667 | 0.3333 | 0.0583 |
| MiniLM v2 | 0.0000 | 0.1333 | 0.1333 | 0.1333 | 0.0000 |
| hybrid v2 | 0.1833 | 0.2667 | 0.2667 | 0.3250 | 0.0917 |

Low-confidence outputs decreased:

| Classifier | V1 low-confidence | V2 low-confidence |
| --- | ---: | ---: |
| lexical | 331 / 396 | 212 / 396 |
| MiniLM | 383 / 396 | 320 / 396 |

Hybrid also became meaningfully different from rule-based:

| Version | Rule fallback | MiniLM semantic label used |
| --- | ---: | ---: |
| v1 | 383 | 13 |
| v2 | 320 | 76 |

Decision: taxonomy coverage is a real bottleneck for semantic labels. The v1 result should not be interpreted as a complete MiniLM failure.

## 5. Source_type Bottleneck

Even after taxonomy v2, full-label accuracy stayed low because `full_label` requires all of these to match:

- `source_type`
- `domain`
- `field`
- `subfield`

The source_type-specific analysis showed:

| Classifier | Source_type accuracy |
| --- | ---: |
| rule-based | 0.1833 |
| lexical v2 | 0.0750 |
| MiniLM v2 semantic output | 0.0000 |
| hybrid v2 | 0.1833 |

A separate source_type classifier experiment then showed:

| Classifier | Source_type accuracy | Low-confidence |
| --- | ---: | ---: |
| original rule-based source_type | 0.1833 | 0 |
| lexical source_type | 0.0083 | 112 |
| MiniLM source_type | 0.0833 | 103 |
| hybrid source_type | 0.1417 | 63 |

The separate MiniLM source_type classifier did not beat original rule-based overall. Hybrid source_type improved FineMath but introduced false positives on FineWeb and FineWeb-Edu.

Decision: source_type is not solved by semantic MiniLM and should not be treated as part of the same classifier problem.

## 6. Why Source_type Is Ambiguous

`source_type` is a format/function/genre label, not a topic label.

Ambiguous cases in real chunks include:

- reference vs educational;
- reference vs technical/code-like text;
- math vs educational math/statistics lesson;
- web_general vs educational/news/reference;
- FineWeb-Edu chunks that are not educational;
- FineMath chunks that are not math;
- patent/POS descriptions that are technical but not code.

Examples of major source_type mismatches:

- `reference -> unknown`: 25
- `educational -> unknown`: 19
- `math -> unknown`: 10
- `reference -> educational`: 9
- `web_general -> unknown`: 6
- `news -> unknown`: 4
- `commercial_product -> unknown`: 4
- `forum_qa -> unknown`: 4

The current label boundaries are too subtle for small chunks and weak baselines. In particular, `reference`, `educational`, and `web_general` overlap heavily unless the labeling policy is very explicit.

## 7. Recommended MVP Evaluation Policy

For the current MVP, evaluate these tasks separately:

1. `source_type`
2. semantic `domain / field / subfield`

Do not use `full_label_accuracy` as the primary KPI.

Recommended reporting:

- Primary semantic metrics:
  - domain accuracy
  - field accuracy
  - subfield accuracy
  - semantic full tuple accuracy: `domain + field + subfield`
- Separate source_type metrics:
  - source_type accuracy
  - source_type confusion matrix
  - accuracy by dataset
- Diagnostic-only metric:
  - full label accuracy: `source_type + domain + field + subfield`

Reason:

Semantic labels and source_type have different failure modes. A classifier can correctly identify `stem / mathematics / statistics` while still failing to decide whether the chunk is `math`, `educational`, or `reference`.

Using full-label accuracy as the headline metric hides real semantic improvements and over-penalizes the system for unresolved source_type boundaries.

## 8. Recommended Next Stage

Freeze `real_hf_benchmark_v1` as a dev benchmark.

Do not keep tuning taxonomy, thresholds, or source_type rules against the same 120 records. Taxonomy v2 was already informed by v1 gaps, so further improvements on v1 risk overfitting.

Recommended sequence:

1. Keep taxonomy v2 as the current semantic taxonomy candidate.
2. Keep original rule-based source_type as the current weak baseline.
3. Do not merge the experimental source_type hybrid yet.
4. Report semantic metrics and source_type metrics separately.
5. Later create a held-out `real_hf_benchmark_v2_test` sample with a different seed or stream window.
6. Validate whether taxonomy v2 and current classifiers generalize.
7. Before NLL/probability profiling, use semantic labels as the primary grouping and source_type as auxiliary metadata.

For future NLL/profiling:

- semantic labels answer "what is this about?";
- source_type answers "what kind of text is this?";
- both can affect observed-token probability profiles, but they should remain separate controls.

## 9. Commit Grouping Recommendation

The working tree contains several logical groups. Commit them separately to keep the project history readable.

### Commit A: real HF benchmark tooling

Files:

- `scripts/build_review_candidates.py`
- `scripts/classify_chunks_embedding_baseline.py`
- `scripts/run_real_hf_benchmark_pipeline.py`

Suggested message:

```text
Improve real HF benchmark tooling
```

### Commit B: real HF benchmark v1 sampled data and candidates

Files:

- `examples/real_hf_benchmark_v1_*_docs.jsonl`
- `data_samples/real_samples/real_hf_benchmark_v1_*`
- `data_samples/real_hf_benchmark_v1_review_candidates.jsonl`
- `docs/real_hf_benchmark_v1_status.md`

Suggested message:

```text
Add real HF benchmark v1 sampled data and candidates
```

### Commit C: pseudo-gold and evaluation

Files:

- `data_samples/real_hf_benchmark_v1_pseudo_gold.jsonl`
- `data_samples/real_hf_benchmark_v1_eval_*.json`
- `docs/real_hf_benchmark_v1_pseudo_gold_summary.md`
- `docs/real_hf_benchmark_v1_evaluation.md`
- `docs/real_hf_benchmark_v1_pseudo_gold_audit.md`
- `scripts/evaluate_predictions_against_pseudo_gold.py`

Suggested message:

```text
Add real HF benchmark v1 pseudo-gold and evaluation
```

### Commit D: taxonomy v2 coverage experiment

Files:

- `taxonomy/simple_domain_labels.json`
- `data_samples/real_hf_benchmark_v1_pseudo_gold_v2.jsonl`
- `data_samples/real_hf_benchmark_v1_eval_*_v2.json`
- `data_samples/real_samples/*_v2.jsonl`
- `docs/real_hf_benchmark_v1_taxonomy_v2_experiment.md`

Suggested message:

```text
Add taxonomy v2 coverage experiment
```

### Commit E: source_type analysis and experiment

Files:

- `taxonomy/source_type_labels.json`
- `scripts/classify_source_type_baseline.py`
- `scripts/evaluate_source_type_predictions.py`
- `data_samples/real_hf_benchmark_v1_source_type_analysis.json`
- `data_samples/real_hf_benchmark_v1_source_type_eval_*.json`
- `data_samples/real_samples/*_source_type_*.jsonl`
- `docs/real_hf_benchmark_v1_source_type_error_analysis.md`
- `docs/real_hf_benchmark_v1_source_type_classifier_experiment.md`

Suggested message:

```text
Add source type benchmark analysis and classifier experiment
```

### Commit F: final decision report

Files:

- `docs/real_hf_benchmark_v1_decision_report.md`

Suggested message:

```text
docs: add real HF benchmark v1 decision report
```
