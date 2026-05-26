# Stage2: corpus chunking and domain labeling

Stage2 is an isolated subproject for corpus preparation and domain labeling in the HSE mini Qwen-like LLM project.

Current scope:

- raw/local/HF sample rows -> logical chunks;
- dataset and `source_type` metadata;
- `domain` / `field` / `subfield` labels;
- rule-based classifier baseline;
- lexical nearest-label baseline;
- tiny real samples for FineWeb-Edu and FineMath;
- disagreement review and preparation for MiniLM/Sentence-Transformers nearest-label classification.

Out of current scope:

- training the model;
- NLL/logprob scoring;
- effective context window experiments;
- OpenWebMath production run;
- root repository restructuring.

Stage2 intentionally stays under `external_samples/stage2_chunking_sample_clean/` for now. This keeps the branch safe and avoids touching shared root-level project areas. Any root README/docs/scripts changes should be discussed separately with the team.

## Start here

Recommended Russian handoff docs:

- `docs/README.md` - documentation index.
- `docs/current_status_ru.md` - current status and safe entry point.
- `docs/source_status_ru.md` - source status table.
- `docs/data_policy_ru.md` - what should and should not be committed.
- `docs/pipeline_map_ru.md` - input/output pipeline map.
- `docs/validation_modes_ru.md` - synthetic vs real validation rules.
- `docs/classifier_contract_ru.md` - classifier output contract.
- `docs/minilm_readiness_plan_ru.md` - MiniLM readiness plan without model download.

Audit reports are in `docs/repo_audit/`.

## Repository map

```text
external_samples/stage2_chunking_sample_clean/
  README.md
  CHECK_HOW_TO.md
  MESSAGE_TO_TEAM.txt
  requirements.txt
  config/
    dataset_sources.json
  docs/
    *.md
    repo_audit/
  examples/
    local JSONL fixtures
  scripts/
    chunking, validation, classifiers, comparison, planning
  taxonomy/
    simple_domain_labels.json
  data_samples/
    small tracked samples, generated local outputs, tiny real samples
```

## Current data status

- FineWeb-Edu: tiny sample completed and tracked for review.
- FineMath: tiny sample completed and tracked; this is the only current MVP math source.
- OpenWebMath: optional_later comparison or backup only.
- FineWeb general: not current.
- Cosmopedia/SmolLM: later or optional, only after explicit approval.
- NLL/effective context artifacts: downstream notes, not part of the current stage2 milestone.

## Safe local checks

Run commands from this folder:

```bash
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_sample.jsonl
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_labeled.jsonl --require-labels
python scripts\validate_chunks.py --input data_samples\classifier_benchmark_labeled.jsonl --require-labels
python scripts\smoke_test_rule_based_classifier.py
python scripts\run_local_benchmark_pipeline.py
python scripts\inspect_dataset_sources.py --registry config\dataset_sources.json
```

Real tiny samples do not have gold labels, so validate them without `--require-labels`:

```bash
python scripts\validate_chunks.py --input data_samples\real_samples\fineweb_edu_sample10bt_chunks.jsonl
python scripts\validate_chunks.py --input data_samples\real_samples\fineweb_edu_sample10bt_labeled_rule_based.jsonl
python scripts\validate_chunks.py --input data_samples\real_samples\fineweb_edu_sample10bt_labeled_lexical.jsonl
python scripts\validate_chunks.py --input data_samples\real_samples\finemath_chunks.jsonl
python scripts\validate_chunks.py --input data_samples\real_samples\finemath_labeled_rule_based.jsonl
python scripts\validate_chunks.py --input data_samples\real_samples\finemath_labeled_lexical.jsonl
```

## Commands that require approval

Do not run these casually:

- HF streaming;
- new dataset downloads;
- dependency installation;
- model download;
- MiniLM/Sentence-Transformers inference;
- NLL/logprob scoring;
- large sweeps or output regeneration.

Use the planning scripts first:

```bash
python scripts\inspect_dataset_sources.py --registry config\dataset_sources.json
python scripts\plan_real_sample_run.py --registry config\dataset_sources.json --all --max-docs 20
python scripts\plan_real_source_pipeline.py --registry config\dataset_sources.json --source fineweb_edu --max-docs 20
python scripts\plan_real_source_pipeline.py --registry config\dataset_sources.json --source finemath --max-docs 20
```

## Next milestone

The next technical milestone is an embedding-based nearest-label classifier using MiniLM/Sentence-Transformers style embeddings.

Real MiniLM runs require optional embedding dependencies and an approved local/downloaded model. See `docs/minilm_dependency_strategy.md` and `docs/minilm_local_preflight_report.md` before installing anything or running inference.

Before running it, stage2 needs:

- stable classifier contract;
- no-download/local-model workflow;
- output naming policy;
- synthetic vs real validation split;
- comparison plan: rule-based vs lexical vs embedding;
- manual review of disagreements on FineWeb-Edu and FineMath tiny samples.
