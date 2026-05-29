# BGE-M3 semantic topic dev evaluation report

## Purpose

This report summarizes the first controlled embedding dev bake-off run for `semantic_topic_domain_v1`.

This is a v1-dev-only experiment. It does not use v2-test and must not be cited as held-out quality.

## Setup

Gold/dev target:

- `data_samples/real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl`

Target field:

- `semantic_topic_domain`

Model:

- `BAAI/bge-m3`

Local snapshot path:

```text
C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181
```

Python:

```text
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe
```

No download was requested or needed. The model was loaded from the explicit local snapshot path.

Command:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\classify_semantic_topic_embedding.py `
  --input data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --domain-descriptions taxonomy\semantic_topic_domain_v1_descriptions.json `
  --output data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3.jsonl `
  --model "C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181" `
  --threshold 0.0 `
  --margin-threshold 0.0 `
  --top-k 3 `
  --text-field text
```

Runtime:

- approximately 103 seconds on CPU for 120 v1-dev records.

## Outputs

Created:

- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3.jsonl`
- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_eval.json`
- `data_samples/real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_threshold_sweep.json`

The threshold sweep was computed from saved BGE-M3 scores only. It did not run additional embedding inference.

## Baseline metrics

Gold records with `semantic_topic_abstained=true` are excluded by the evaluator. Therefore:

- source pseudo-gold records: 120;
- evaluated records: 118;
- gold-abstained excluded: 2.

Baseline thresholds:

- `threshold=0.0`;
- `margin_threshold=0.0`;
- `top_k=3`.

Metrics:

| Metric | Value |
| --- | ---: |
| records_total | 118 |
| answered_count | 118 |
| abstained_count | 0 |
| coverage_rate | 1.000000 |
| accuracy_on_answered | 0.381356 |
| strict_accuracy | 0.381356 |
| top_k_contains_gold | 0.796610 |
| macro_f1 | 0.294358 |

## Per-dataset metrics

| Dataset | Records | Coverage | Accuracy on answered | Strict accuracy |
| --- | ---: | ---: | ---: | ---: |
| FineMath | 38 | 1.000000 | 0.552632 | 0.552632 |
| FineWeb | 40 | 1.000000 | 0.225000 | 0.225000 |
| FineWeb-Edu | 40 | 1.000000 | 0.375000 | 0.375000 |

Interpretation:

- FineMath is the strongest subset.
- FineWeb is the weakest subset.
- FineWeb-Edu is mixed and exposes semantic/genre boundary ambiguity.

## Per-domain precision/recall highlights

| Domain | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `math` | 0.629630 | 0.739130 | 0.680000 | 23 |
| `natural_science` | 0.642857 | 0.391304 | 0.486486 | 23 |
| `lifestyle_everyday` | 0.416667 | 0.500000 | 0.454545 | 10 |
| `law_government_policy` | 0.428571 | 0.428571 | 0.428571 | 7 |
| `humanities_arts` | 0.750000 | 0.230769 | 0.352941 | 13 |
| `engineering_technology` | 0.833333 | 0.192308 | 0.312500 | 26 |
| `medicine_health` | 0.500000 | 0.200000 | 0.285714 | 5 |
| `computer_science_software` | 0.052632 | 0.200000 | 0.083333 | 5 |
| `social_sciences` | 0.125000 | 0.200000 | 0.153846 | 5 |
| `business_economics` | 0.000000 | 0.000000 | 0.000000 | 1 |

## Confusion matrix highlights

Largest confusions:

| Count | Gold | Predicted |
| ---: | --- | --- |
| 18 | `engineering_technology` | `computer_science_software` |
| 8 | `natural_science` | `math` |
| 3 | `math` | `other_unclear` |
| 3 | `lifestyle_everyday` | `other_unclear` |
| 3 | `humanities_arts` | `lifestyle_everyday` |
| 3 | `computer_science_software` | `other_unclear` |
| 2 | `social_sciences` | `business_economics` |
| 2 | `natural_science` | `social_sciences` |
| 2 | `humanities_arts` | `social_sciences` |
| 2 | `humanities_arts` | `other_unclear` |
| 2 | `humanities_arts` | `natural_science` |

Main failure pattern:

- patent/POS/video-rental system chunks labeled `engineering_technology` are often predicted as `computer_science_software`.

This is understandable: the chunks contain terminals, LAN adapters, programs, data capture, screens, and transaction systems. The label-card-only classifier is seeing software/computing vocabulary but not reliably preserving the intended engineering/patent topic.

Second failure pattern:

- `natural_science` sometimes becomes `math`, especially when science text is formula-heavy or quantitative.

Third failure pattern:

- humanities/social/policy/lifestyle boundaries remain soft for reflective, historical, or essay-like web content.

## Mismatch examples

Examples from the evaluator:

| Chunk | Dataset | Gold | Predicted | Confidence | Margin |
| --- | --- | --- | --- | ---: | ---: |
| `FineWeb-Edu_000014_000` | FineWeb-Edu | `law_government_policy` | `natural_science` | 0.484108 | 0.000181 |
| `FineWeb-Edu_000018_000` | FineWeb-Edu | `humanities_arts` | `law_government_policy` | 0.528633 | 0.029253 |
| `FineWeb_000009_051` | FineWeb | `engineering_technology` | `computer_science_software` | 0.372156 | 0.001685 |
| `FineWeb-Edu_000006_000` | FineWeb-Edu | `natural_science` | `social_sciences` | 0.472096 | 0.041031 |
| `FineWeb_000009_013` | FineWeb | `engineering_technology` | `computer_science_software` | 0.466978 | 0.027101 |

Several wrong predictions have very small margins, which suggests that margin-based abstention or reranking may be useful.

## Threshold/margin sweep

Sweep settings:

- thresholds: `0.0`, `0.2`, `0.3`, `0.4`;
- margin thresholds: `0.0`, `0.02`, `0.05`.

Best strict accuracy in this small sweep:

| threshold | margin_threshold | coverage | accuracy_on_answered | strict_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 1.000000 | 0.381356 | 0.381356 |

Best accuracy on answered:

| threshold | margin_threshold | coverage | accuracy_on_answered | strict_accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.05 | 0.093220 | 0.727273 | 0.067797 |

Interpretation:

- margin filtering can improve precision on answered records;
- the cost is very low coverage;
- this first sweep is too small and blunt to choose a final operating point.

## Conceptual comparison to weak_topic_domain_v2.1

Do not compare these numbers directly to `weak_topic_domain_v2.1` as final quality because the label spaces differ:

- BGE-M3 targets cleaned `semantic_topic_domain_v1`;
- `weak_topic_domain_v2.1` targets the old mixed `topic.domain`;
- the new target removes genre/function labels from the primary semantic task.

Conceptually, BGE-M3 gives useful semantic signal:

- top-3 contains gold for about 80% of evaluated records;
- top-1 is not strong enough yet;
- errors are interpretable and concentrated in known boundary zones.

## Is BGE-M3 promising?

Yes, but not as a drop-in top-1 classifier yet.

Promising signs:

- no-download local run succeeded;
- top-3 recall is high enough to support reranking/manual review workflows;
- math and natural science have meaningful signal;
- many errors have low margins.

Weak signs:

- top-1 strict accuracy is only 0.381356 on v1-dev;
- FineWeb accuracy is especially low at 0.225000;
- engineering technology is over-routed to software;
- some labels with low support are unstable.

## Recommended next step

1. Improve dev-only decision policy:
   - margin-based abstention;
   - top-k reranking;
   - domain-card wording variants;
   - explicit patent/engineering vs software contrast cards.

2. Compare another embedding model only after approval/download if needed:
   - MiniLM after its snapshot is restored/downloaded;
   - E5 or Qwen embedding candidates after explicit approval.

3. Prepare cleaned held-out evaluation later:
   - either relabel v2-test under cleaned axes with audit notes;
   - or create a fresh v3 held-out split.

Do not use this v1-dev run as held-out evidence.
