# Аудит 05: итоговая картина и готовность к MiniLM

Дата: 2026-05-26

Область аудита: итоговая сводка по stage2-подпроекту `external_samples/stage2_chunking_sample_clean` на основе предыдущих audit reports. Проблемы не исправлялись. MiniLM/Sentence-Transformers не реализовывался, модели не скачивались, HF/network не запускался, commit/push не выполнялись.

## Executive summary

Stage2 уже содержит реальную полезную работу: chunking, metadata propagation, rule-based baseline, lexical nearest-label baseline, validators, inspectors, comparison tools, tiny real samples FineWeb-Edu/FineMath и disagreement review. Это уже не пустой прототип.

Но для team handoff и перехода к MiniLM он пока готов только частично. Главная проблема не в отсутствии кода, а в том, что проект вырос итерациями и теперь нуждается в стабилизации входной карты, документации, data policy и classifier contract.

Readiness к MiniLM:

- Conceptual readiness: medium.
- Data readiness: medium.
- Code readiness: medium-low.
- Team handoff readiness: low-medium.
- Safe model-run readiness: low until dependencies/model/local-cache policy are explicitly prepared.

Главная рекомендация: не начинать MiniLM milestone с запуска модели. Сначала нужно зафиксировать schema/contract, Russian handoff docs, source status, output naming, validation modes и no-download policy.

## Branch isolation and safe boundaries

Important scope correction:

- Stage2 is intentionally isolated under `external_samples/stage2_chunking_sample_clean/` in this branch.
- The fact that this path hurts onboarding does not mean the immediate fix is moving stage2 to the repo root.
- Immediate cleanup should be stage2-local: docs, source status, data policy, pipeline map, MiniLM readiness, and audit follow-up inside the stage2 folder.
- Root-level changes are protected and should happen only later, optionally, after explicit team approval.

Protected for now:

- root `README.MD`;
- root `scripts/`;
- root `docs/`;
- `research/`, notebooks, meeting notes, dataset/distribution folders, and other team-owned areas.

Root-level pointer or moving/renaming stage2 is postponed. For the next iterations, the safer goal is to make stage2 self-contained and easy to hand off without disturbing the rest of the repository.

## Насколько репозиторий понятен для команды

Текущая понятность: частичная.

Новый участник сможет разобраться, если у него есть внешний контекст от автора. Без этого onboarding будет медленным:

- активный stage2 спрятан в `external_samples/stage2_chunking_sample_clean`;
- root README почти не объясняет проект;
- root `scripts/` содержит zero-byte stub, а реальные scripts лежат внутри stage2;
- docs stage2 плоские и многочисленные;
- подробные текущие reports в основном на английском;
- NLL/future probability материалы лежат рядом с current corpus-labeling work;
- `data_samples/` содержит разные типы artifacts без явной policy.

Команда увидит, что работа большая, но не сразу поймет:

- какой документ читать первым;
- что актуально;
- что является входом/выходом pipeline;
- какие файлы generated;
- что можно запускать локально;
- где граница между текущей задачей и downstream NLL/effective context window.

## Что уже хорошо

1. Есть рабочий stage2 pipeline для локального MVP.
2. Chunker поддерживает local JSONL и HF streaming mode, при этом локальный путь можно проверять без сети.
3. Есть dataset/source_type metadata и базовая schema.
4. Есть rule-based baseline, прозрачный и без зависимостей.
5. Есть lexical nearest-label baseline, полезный как cheap comparison перед MiniLM.
6. Есть optional embedding scaffold с `--dry-run` и no-download intent.
7. Есть validators, inspectors, comparison scripts и review helpers.
8. Есть synthetic benchmark и edge/local real-like fixtures.
9. Есть tiny real samples FineWeb-Edu и FineMath с rule-based/lexical labels.
10. FineMath уже представлен как current MVP math source в нескольких docs.
11. OpenWebMath не был запущен и остается optional_later.
12. HF cache не попал в tracked files.
13. Repo пока не раздут по размеру: stage2 data artifacts около 1 MB.

## Критичные проблемы

Критичные для handoff:

- нет единого русского current-state/quickstart документа;
- docs перегружены английскими reports и старыми планами;
- root и stage2 entry points не объясняют текущий scope;
- FineMath/OpenWebMath/NLL границы не видны сразу;
- source statuses размазаны между registry, reports и checklists;
- generated outputs и tiny samples лежат рядом без data policy.

Критичные перед MiniLM:

- не стабилизирован contract `source_type` vs `domain/field/subfield`;
- synthetic benchmark и real samples требуют разных validation/evaluation modes;
- lexical и embedding nearest-label outputs ведут себя не полностью одинаково;
- нет shared utilities для JSONL/taxonomy/label tuple;
- embedding scaffold не готов к реальному MiniLM run: нет batching, top-k, canonical output schema, local-model-path workflow, tests;
- no-download policy есть как intention, но не оформлена как строгий preflight workflow.

## Что мешает team handoff

1. Вход в проект не очевиден. Активная работа находится в `external_samples`, а root README говорит, что repo черновой.
2. Документация не иерархична. Нет `docs/README.md`, `PIPELINE.md`, `current_status.md`.
3. Большая часть meaningful docs на английском, хотя команда/преподаватель likely Russian-speaking.
4. Старые планы и completed reports выглядят одинаково актуальными.
5. NLL материалы создают ложное ощущение, что это current responsibility.
6. `data_samples/README.md` устарел и не описывает реальные artifacts.
7. Untracked sweep outputs шумят в `git status`.
8. Нет явного "do not run" / "safe local only" handoff для HF/model tasks.

## Что мешает MiniLM stage

MiniLM stage зависит не только от модели, а от сопоставимого classifier protocol.

Главные blockers:

- нужно определить, что embedding classifier обязан писать в JSONL;
- нужно решить, сохраняем ли top-k alternatives;
- нужно решить, меняет ли classifier `source_type` или только `domain/field/subfield`;
- нужно зафиксировать low-confidence/null label behavior;
- нужно сравнивать embedding с rule-based и lexical на тех же chunks;
- нужно отделить benchmark accuracy от real-sample agreement/review;
- нужно исключить accidental model download;
- нужно выбрать local model path/cache policy;
- нужно назвать outputs так, чтобы не перезаписать existing rule/lexical outputs.

## Что перевести и упростить в документации

High priority:

- stage2 `README.md`: переписать на русском вокруг current scope.
- новый `docs/README.md` или `docs/quickstart_ru.md`: один документ для команды.
- `chunk_schema.md`: дать русское краткое описание schema.
- `classifier_comparison_report.md`: перевести/summarize как evidence для baselines.
- `embedding_baseline_readiness.md` + `local_embedding_environment_check.md`: объединить в русский MiniLM readiness/plan.
- `fineweb_edu_tiny_sample_report.md` и `finemath_tiny_sample_report.md`: сделать русские summary.
- `dataset_source_registry.md`: заменить или дополнить source status table.
- `real_sample_disagreement_review.md`: сделать короткую summary, подробности оставить appendix.

Упростить:

- `next_steps_checklist.md`: заменить на current status board.
- `real_sample_next_plan.md`: rewrite or archive, потому что там смешаны future и completed states.
- `hf_dataset_verification_plan.md`: archive or replace, потому что часть данных уже known/sampled.
- `night_work_log.md`: archive как historical work log.

Перенести в future/archive:

- `probability_profile_schema.md`;
- `nll_pilot_candidate_notes.md`;
- `nll_scoring_next_steps.md`;
- NLL-related manifests, если они не нужны в current handoff.

## Что стабилизировать в коде

Перед MiniLM обязательно:

1. Shared JSONL utilities:
   - `iter_jsonl`;
   - `write_jsonl`;
   - line-number JSON errors;
   - common UTF-8 handling.

2. Shared label/taxonomy utilities:
   - label fields;
   - expected fields;
   - null normalization;
   - label tuple;
   - label text construction;
   - taxonomy loading/validation.

3. Classifier output contract:
   - stable source metadata vs predicted source_type;
   - null label behavior;
   - confidence meaning as similarity/heuristic, not probability;
   - `label_method` naming convention.

4. Real vs synthetic modes:
   - synthetic: expected labels, accuracy, strict validation;
   - real: null labels allowed, agreement/disagreement, manual review.

5. Embedding classifier scaffolding:
   - local-only model loading;
   - `--batch-size`;
   - optional `--device`;
   - top-k outputs or at least top-k report;
   - output summary with model name, threshold, text truncation, taxonomy path;
   - mock tests for deterministic embeddings.

Можно отложить:

- full package restructure;
- one unified CLI entrypoint;
- atomic writes;
- structured JSON summaries everywhere;
- full HF exception taxonomy;
- OpenAlex taxonomy migration;
- large-scale embedding optimization;
- NLL/effective context window tooling.

## Что делать с data artifacts

Keep tracked:

- `examples/*.jsonl` small fixtures;
- `config/dataset_sources.json`;
- `taxonomy/simple_domain_labels.json`;
- canonical small benchmark outputs if team wants reproducible examples;
- FineWeb-Edu/FineMath tiny sample reports and stats.

Keep tracked only after privacy/license decision:

- FineWeb-Edu raw chunks and labeled outputs;
- FineMath raw chunks and labeled outputs.

Do not commit:

- HF caches;
- model caches;
- `__pycache__`;
- old temporary sweeps;
- large raw dataset dumps;
- model-dependent embedding outputs unless promoted intentionally;
- OpenWebMath samples before optional_later approval;
- NLL/logprob outputs in current stage2 scope.

Clean up later:

- move old sweep outputs to ignored `data_samples/sweeps/` or delete after summary extraction;
- expand `data_samples/README.md` into data policy;
- add broader `.gitignore` patterns for caches, sweeps, temp outputs, model outputs;
- define naming for MiniLM outputs before running anything.

## Actions before MiniLM

Mandatory:

1. Create Russian stage2 handoff docs:
   - current scope;
   - current pipeline;
   - current artifacts;
   - safe commands;
   - next milestone.

2. Add source status table:
   - FineWeb-Edu: tiny sample completed;
   - FineMath: current MVP math source, tiny sample completed;
   - OpenWebMath: optional_later only;
   - NLL/effective context: downstream/out of current scope.

3. Define classifier output contract:
   - source metadata;
   - predicted label fields;
   - low-confidence/null behavior;
   - confidence interpretation.

4. Split synthetic vs real evaluation mode.

5. Unify lexical and embedding nearest-label behavior:
   - shared label text;
   - same taxonomy input;
   - same source_type policy;
   - same top-k/low-confidence semantics where possible.

6. Prepare MiniLM no-download workflow:
   - explicit approval for dependencies/model;
   - local model path or verified local cache;
   - fail closed if unavailable;
   - no implicit HF/model download.

7. Define output naming:
   - no overwriting existing rule/lexical outputs;
   - include source, model short name, threshold, run date/stats.

8. Add lightweight tests for embedding classifier with mock encoder.

## Actions that can be postponed

- Moving stage2 out of `external_samples`.
- Adding/changing root-level pointers or root README.
- Full root repo cleanup.
- Large refactor into Python package.
- Single CLI wrapper.
- Atomic writes for all scripts.
- Full docs translation of every historical file.
- Archiving every old report immediately.
- New taxonomy migration.
- OpenWebMath comparison.
- NLL/logprob/effective context implementation.
- Large-scale or GPU-optimized embedding inference.

## Proposed roadmap

### 1. Cleanup / Organization

Goal: make stage2 navigable without changing scientific behavior or touching protected root areas.

Actions:

- Add a stage2-local docs index.
- Add a stage2-local current status / pipeline map.
- Document, but do not immediately change, the root `scripts/` vs stage2 `scripts/` ambiguity.
- Define data artifact categories.
- Move/ignore temporary sweeps later.
- Keep NLL materials clearly future/downstream.
- Treat root-level pointers or moving stage2 as later/optional after team approval.

Deliverable:

- New contributor can find stage2 code, docs, inputs, outputs, and current status in under 5 minutes from inside the stage2 folder.

### 2. Russian Docs / Team Handoff

Goal: make the work explainable to team and преподаватель.

Actions:

- Rewrite stage2 README in Russian.
- Create `quickstart_ru.md` or `PIPELINE.md`.
- Add source status table.
- Add concise real sample summary.
- Add MiniLM readiness note.
- Add explicit scope banner:

```text
Current: chunking + domain labeling.
Next: MiniLM/Sentence-Transformers nearest-label classifier.
Downstream: NLL/logprob/effective context window.
```

Deliverable:

- Team can understand what is done, what is next, and what is out of scope.

### 3. Pipeline Stabilization

Goal: make rule-based, lexical, and embedding comparisons fair and repeatable.

Actions:

- Add shared JSONL/taxonomy/label helpers.
- Stabilize classifier output contract.
- Separate benchmark and real-sample validation modes.
- Define output naming and run stats.
- Update planners/commands later so they do not imply strict real labels.

Deliverable:

- Same input chunks can be labeled by rule-based, lexical, and embedding classifiers with comparable outputs.

### 4. MiniLM Classifier

Goal: implement embedding nearest-label classifier safely.

Actions:

- Use Sentence-Transformers with MiniLM only after explicit dependency/model approval.
- Prefer local model path or verified cache.
- Fail closed on missing model.
- Add batching and optional device control.
- Record model/settings in output.
- Add top-k or at least top-k diagnostics.
- Add mock tests.

Deliverable:

- `*_labeled_embedding_minilm*.jsonl` outputs for benchmark, FineWeb-Edu, and FineMath tiny samples, produced without accidental download.

### 5. Comparison: Rule-Based vs Lexical vs Embedding

Goal: decide whether MiniLM improves current labeling behavior.

Actions:

- Run all three methods on the same benchmark chunks.
- Run all three methods on FineWeb-Edu tiny chunks.
- Run all three methods on FineMath tiny chunks.
- Compare:
  - agreement rates;
  - low-confidence counts;
  - source_type/domain disagreements;
  - math-vs-education cases;
  - boilerplate/noise cases.
- Manually review selected disagreements.

Deliverable:

- Short report: where embedding helps, where it fails, and whether taxonomy/rules need adjustment.

### 6. Final Handoff To Team

Goal: make the stage self-contained for review and next contributor.

Actions:

- Produce Russian final summary.
- Include pipeline diagram/table.
- Include source status table.
- Include data policy.
- Include commands that are safe to run.
- Include commands that require approval.
- Include known limitations and out-of-scope notes.

Deliverable:

- Team can reproduce local checks, inspect tiny samples, and understand why MiniLM is the next step.

## Audit files

Created audit reports:

1. `docs/repo_audit/01_structure_organization_problems.md`
2. `docs/repo_audit/02_documentation_language_problems.md`
3. `docs/repo_audit/03_code_pipeline_problems.md`
4. `docs/repo_audit/04_data_artifacts_git_hygiene_problems.md`
5. `docs/repo_audit/05_final_audit_summary_and_minilm_readiness.md`
6. `docs/repo_audit/06_branch_scope_and_safe_boundaries.md`

## Top-10 problems

1. Active stage2 work is hidden under `external_samples`, while root repo does not explain current ownership; immediate fix should still be stage2-local documentation, not moving stage2.
2. No single Russian current-state quickstart or pipeline map exists.
3. Documentation is mostly English and scattered across reports, plans, runbooks, and historical logs.
4. Current scope is blurred by NLL/logprob/effective context window materials.
5. FineMath current MVP role and OpenWebMath optional_later status are not centralized in the main entry point.
6. `source_type` vs `domain/field/subfield` classifier contract is not stable enough for MiniLM.
7. Synthetic benchmark and real samples need separate validation/evaluation modes.
8. Lexical and embedding nearest-label classifiers are conceptually aligned but behaviorally inconsistent.
9. `data_samples/` mixes fixtures, generated outputs, real samples, old sweeps, future NLL artifacts, and stats.
10. Old sweep outputs are untracked but not ignored, and future model/embedding outputs need a clearer policy.

## Recommended next actions

Immediate, before any MiniLM code/model work:

1. Write Russian stage2 current-state/quickstart doc.
2. Add source status table with FineWeb-Edu, FineMath, OpenWebMath.
3. Add explicit current-scope banner separating chunking/domain labeling from NLL.
4. Define classifier output contract.
5. Define synthetic vs real validation modes.
6. Define MiniLM no-download/local-model workflow.
7. Define MiniLM output naming and data policy.
8. Decide privacy/license stance for tracked FineWeb-Edu/FineMath raw tiny samples.

Then, before real MiniLM run:

1. Extract small shared JSONL/taxonomy/label helpers.
2. Align lexical and embedding output behavior.
3. Add batching/top-k/settings summary to embedding classifier.
4. Add mock tests.
5. Run only safe dry-run/preflight checks until dependencies/model are approved.

After MiniLM outputs exist:

1. Compare rule-based vs lexical vs embedding on synthetic benchmark.
2. Compare all three on FineWeb-Edu tiny sample.
3. Compare all three on FineMath tiny sample.
4. Manually review disagreements.
5. Produce Russian handoff report.

## Suggested commit grouping for audit reports

Do not commit from this audit step. Suggested grouping if/when commits are requested later:

1. `docs/audit: add stage2 structure and documentation audit`
   - `01_structure_organization_problems.md`
   - `02_documentation_language_problems.md`

2. `docs/audit: add stage2 pipeline and data hygiene audit`
   - `03_code_pipeline_problems.md`
   - `04_data_artifacts_git_hygiene_problems.md`

3. `docs/audit: add MiniLM readiness summary`
   - `05_final_audit_summary_and_minilm_readiness.md`
   - `06_branch_scope_and_safe_boundaries.md`

Alternative simpler grouping:

- one docs-only commit: `docs/audit: add stage2 repo audit and MiniLM readiness reports`

Do not mix these audit reports with code cleanup, `.gitignore` changes, data deletion, MiniLM implementation, or generated outputs. Keeping audit separate will make later cleanup decisions easier to review.
