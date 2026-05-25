# Cleanup stabilization log

Date: 2026-05-26

## Purpose

This cleanup pass stabilizes stage2 documentation and handoff materials before the MiniLM/Sentence-Transformers embedding classifier milestone.

The goal is stage2-local cleanup only. No root repo restructuring, MiniLM implementation, HF streaming, model download, commit, push, or deletion of old sweep outputs is part of this pass.

## Audit findings addressed

- Missing Russian entry point for team handoff.
- Unclear current scope vs downstream NLL/effective context work.
- Source status scattered across reports and planning docs.
- FineMath current MVP math role not visible enough from entry docs.
- OpenWebMath optional_later status not centralized enough.
- No single data policy for tracked samples, temporary sweeps, caches, and future embedding outputs.
- Classifier contract not explicit enough before MiniLM.
- Synthetic benchmark and real sample validation modes mixed together.
- MiniLM readiness needs a no-download/local-model workflow before any implementation.

## Files changed or added

Added:

- `docs/README.md`
- `docs/current_status_ru.md`
- `docs/source_status_ru.md`
- `docs/data_policy_ru.md`
- `docs/classifier_contract_ru.md`
- `docs/pipeline_map_ru.md`
- `docs/validation_modes_ru.md`
- `docs/minilm_readiness_plan_ru.md`
- `docs/repo_audit/07_cleanup_stabilization_log.md`

Updated:

- `README.md`
- `data_samples/README.md`
- `docs/next_steps_checklist.md`
- `docs/real_sample_next_plan.md`

## Decisions

- Stage2 remains isolated under `external_samples/stage2_chunking_sample_clean/`.
- Immediate cleanup stays inside stage2.
- Root README/docs/scripts remain protected.
- FineMath is the only current MVP math source.
- OpenWebMath is optional_later only.
- NLL/logprob/effective context are downstream and out of current scope.
- Real sample agreement/disagreement is review signal, not accuracy.
- MiniLM needs explicit dependency/model approval and local/no-download workflow.

## Not touched

- Root repo files.
- Root `.gitignore`.
- Stage2 classifier/chunking logic.
- Old untracked sweep outputs.
- HF caches or model caches.
- NLL/logprob code or scoring.
- MiniLM model inference.

## Blockers remaining

- No MiniLM dependency/model approval yet.
- Embedding classifier still needs local-model preflight, batching/top-k decisions, and mock tests before real inference.
- Old sweep outputs still create untracked `git status` noise.
- Privacy/license decision for tracked tiny real samples should be reviewed before final handoff.
- Some historical docs remain long and English; they are now appendix material rather than the primary entry point.

## Next steps

1. Review the new Russian handoff docs.
2. Decide whether to commit the docs-only cleanup.
3. Decide data policy for old sweep outputs and future embedding outputs.
4. Stabilize classifier code only after the contract is accepted.
5. Prepare MiniLM preflight without downloads.
6. Implement MiniLM nearest-label classifier in a separate code commit later.

## Checks

All requested safe local checks passed with `C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe`.

Passed:

- validate edge-case raw chunks;
- validate edge-case labeled chunks with `--require-labels`;
- validate classifier benchmark labeled output with `--require-labels`;
- run `smoke_test_rule_based_classifier.py`;
- run `run_local_benchmark_pipeline.py`;
- validate FineWeb-Edu chunks/rule/lexical outputs without `--require-labels`;
- validate FineMath chunks/rule/lexical outputs without `--require-labels`;
- compare FineWeb-Edu rule-based vs lexical outputs;
- compare FineMath rule-based vs lexical outputs;
- inspect dataset sources;
- run FineWeb-Edu and FineMath planners in planning mode only.

Observed comparison signals:

- FineWeb-Edu rule vs lexical agreement: `0.5455` on 44 common chunks.
- FineMath rule vs lexical agreement: `0.4851` on 101 common chunks.

These are review signals for real samples, not accuracy metrics.
