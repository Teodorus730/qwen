# Stage2 final handoff status

## Purpose

This status note summarizes the final packaging pass for Stage2 handoff.

No new research iteration was performed. No HF streaming, model download, MiniLM run, embedding tuning, NLL/logits, feature extraction changes, tokenization changes, pseudo-gold relabeling, or v2-test tuning was performed.

## What was packaged

Created unified chunk-level handoff outputs:

- `data_samples/handoff/stage2_v1_dev_annotations.jsonl`
- `data_samples/handoff/stage2_v2_test_annotations.jsonl`
- `data_samples/handoff/stage2_handoff_validation_summary.json`

Created packaging/validation scripts:

- `scripts/build_stage2_handoff_annotations.py`
- `scripts/validate_stage2_handoff_annotations.py`

Updated team handoff:

- `STAGE2_HANDOFF.md`

The unified outputs join existing component artifacts by `chunk_id`.

## Validation result

Validation passed.

| Output | Records | Duplicate chunk IDs | Valid |
| --- | ---: | ---: | --- |
| v1-dev chunk-level handoff | 396 | 0 | yes |
| v2-test chunk-level handoff | 416 | 0 | yes |

Dataset counts:

| Split | FineWeb | FineWeb-Edu | FineMath |
| --- | ---: | ---: | ---: |
| v1-dev handoff | 113 | 88 | 195 |
| v2-test handoff | 52 | 119 | 245 |

The reviewed pseudo-gold subsets are smaller than the chunk-level handoff files. The unified handoff outputs cover all available tokenized feature records for the relevant split and attach labels/predictions where available.

## Ready for team review

Ready:

- deterministic `annotation_v2` fields;
- Qwen3.5 tokenization stats;
- surface features;
- quality/noise fields;
- legacy `weak_topic_domain_v2.1` outputs;
- cleaned `semantic_topic_domain_v1` and `genre_function_v1` v1-dev pseudo-gold;
- BGE-M3 v1.1/reranked v1-dev semantic-topic predictions;
- unified handoff outputs for v1-dev and v2-test.

## Not ready / not final

Not final:

- topic classifier quality;
- FineWeb topic reliability;
- BGE-M3 held-out quality;
- cleaned semantic-topic held-out evaluation.

The v2-test split has not been labeled/evaluated under cleaned semantic/genre axes.

## v2-test tuning status

v2-test was not used for cleaned-axis tuning in this packaging pass.

Existing v2-test artifacts remain legacy/held-out evidence for the old mixed topic-domain baseline. They should not be used as tuning data.

## Recommended next step

Choose one of two next steps:

1. Create/evaluate a cleaned held-out split:
   - relabel v2-test under cleaned `semantic_topic_domain`/`genre_function` with audit notes; or
   - create a fresh v3 held-out split.
2. Build an NLL grouping interface using robust axes first:
   - provenance;
   - deterministic text stats;
   - Qwen3.5 tokenization stats;
   - surface features;
   - quality/noise fields.

Until cleaned held-out exists, use semantic topic labels and BGE-M3 predictions as exploratory/dev metadata only.
