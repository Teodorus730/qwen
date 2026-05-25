# Аудит 02: проблемы документации и языка stage2

Дата: 2026-05-26

Область аудита: документация stage2-подпроекта в `external_samples/stage2_chunking_sample_clean`. Код, данные, существующие docs и generated outputs не исправлялись. HF/network/model download, pipeline scripts, commit и push не запускались.

## Краткий вывод

Документации много, но она не собрана в понятный слой для нового участника, команды или преподавателя. Сейчас это скорее рабочий журнал + набор отчетов после итераций, чем документация продукта/pipeline.

Главные проблемы:

- почти вся подробная stage2-документация написана на английском;
- нет одного русского quickstart/current-state документа;
- актуальные результаты, планы и ограничения размазаны по нескольким файлам;
- часть документов устарела после FineWeb-Edu/FineMath tiny samples;
- NLL/effective context window материалы лежат рядом с текущей задачей и размывают scope;
- FineMath как текущий единственный math source объяснен, но не в главном entry point;
- OpenWebMath как `optional_later` объяснен в нескольких местах, но не вынесен в один authoritative status.

## Что было просмотрено

Stage2 entry files:

- `README.md`
- `CHECK_HOW_TO.md`
- `MESSAGE_TO_TEAM.txt`

Stage2 docs:

- `chunk_schema.md`
- `chunking_first_findings.md`
- `chunking_parameter_sweep.md`
- `classifier_benchmark_audit.md`
- `classifier_benchmark_report.md`
- `classifier_comparison_report.md`
- `cleaning_and_boilerplate_notes.md`
- `dataset_source_registry.md`
- `embedding_baseline_readiness.md`
- `finemath_tiny_sample_plan.md`
- `finemath_tiny_sample_report.md`
- `fineweb_edu_tiny_sample_report.md`
- `future_hf_streaming_runbook.md`
- `hf_dataset_verification_plan.md`
- `hf_row_adapter_design.md`
- `local_classifier_benchmark_dataset_card.md`
- `local_embedding_environment_check.md`
- `local_real_like_sample_report.md`
- `next_steps_checklist.md`
- `night_work_log.md`
- `nll_pilot_candidate_notes.md`
- `nll_scoring_next_steps.md`
- `probability_profile_schema.md`
- `real_sample_disagreement_review.md`
- `real_sample_next_plan.md`
- `real_sample_readiness_checklist.md`
- `real_samples_output_structure.md`
- `real_source_labeling_plan.md`
- `taxonomy_coverage_notes.md`
- `taxonomy_notes.md`
- `repo_audit/01_structure_organization_problems.md`

## Language audit

### Russian or mostly Russian

- `README.md`: mixed; main early sections are Russian, later MVP section is English.
- `CHECK_HOW_TO.md`: Russian.
- `MESSAGE_TO_TEAM.txt`: mixed; first half Russian, second half English.
- `docs/chunking_first_findings.md`: mostly Russian.
- `docs/repo_audit/01_structure_organization_problems.md`: mostly Russian.

### English

Most detailed stage2 docs are English:

- schema and taxonomy docs;
- benchmark reports/audits;
- dataset registry docs;
- real sample plans/reports;
- embedding readiness docs;
- NLL/probability docs;
- runbooks/checklists;
- disagreement review.

### Why this is a problem

For internal engineering work English is fine, but this project is for a Russian-speaking HSE team and likely has a Russian-speaking instructor/reviewer. The current language split creates a strange experience:

- quick early notes are Russian;
- the actual current evidence and technical decisions are English;
- to understand current stage2 status, a teammate must read English reports;
- преподавателю сложнее быстро проверить, что именно сделано и где граница ответственности.

### What should be translated first

High priority for Russian translation:

- `README.md` as the main stage2 overview;
- a new `docs/README.md` or `PIPELINE.md` current map;
- `chunk_schema.md`;
- `classifier_comparison_report.md`;
- `embedding_baseline_readiness.md`;
- `fineweb_edu_tiny_sample_report.md`;
- `finemath_tiny_sample_report.md`;
- a shortened summary of `real_sample_disagreement_review.md`;
- `dataset_source_registry.md` or a derived source-status table.

Medium priority:

- `classifier_benchmark_report.md`;
- `local_classifier_benchmark_dataset_card.md`;
- `taxonomy_notes.md`;
- `real_source_labeling_plan.md`;
- `future_hf_streaming_runbook.md`.

Low priority / do not translate unless needed:

- `night_work_log.md`;
- old sweep notes;
- future NLL/probability notes;
- detailed appendix-style review dumps.

## Quickstart audit

### What exists

- `CHECK_HOW_TO.md` is a simple local smoke-test quickstart.
- `README.md` includes local MVP commands.
- `MESSAGE_TO_TEAM.txt` includes useful commands for the local benchmark.

### Main issue

There is no single current quickstart that matches the actual current responsibility:

```text
raw/local/HF datasets
-> logical chunks
-> dataset/source_type metadata
-> domain/field/subfield labels
-> rule-based baseline
-> lexical nearest-label baseline
-> real tiny FineWeb-Edu/FineMath samples
-> disagreement review
-> next: MiniLM/Sentence-Transformers embedding classifier
```

The existing quickstart is still centered on early `sample_fineweb_chunks.py --max-docs 5` and `fineweb_chunks_sample.jsonl`. It does not foreground:

- rule-based classifier;
- lexical baseline;
- real sample outputs;
- FineMath current MVP role;
- embedding baseline as next milestone;
- "no HF/network/model download without approval";
- "NLL/effective context window is downstream, not current work".

### Recommendation

Create a new Russian `README.md` or `docs/quickstart_ru.md` with:

- what this stage2 subproject does;
- what it does not do;
- current pipeline map;
- safe local commands only;
- current outputs;
- source status table;
- next milestone: MiniLM/Sentence-Transformers embedding classifier;
- explicit NLL/effective context window boundary.

Keep old smoke-test commands as a small "legacy/local sanity check" section, not as the main story.

## Актуальность документов

### Current and useful as source material

These documents are broadly current and should be preserved, but several need Russian summaries or integration into a main map:

- `chunk_schema.md`: current schema explanation; should be translated/summarized in Russian.
- `classifier_comparison_report.md`: important current comparison of rule-based, lexical, and future embedding baseline.
- `embedding_baseline_readiness.md`: directly relevant to the next milestone.
- `local_embedding_environment_check.md`: useful environment note, but should be merged with embedding readiness.
- `fineweb_edu_tiny_sample_report.md`: current real sample report.
- `finemath_tiny_sample_report.md`: current FineMath report and key evidence for current math source.
- `real_sample_disagreement_review.md`: important evidence, but too long for main docs.
- `dataset_source_registry.md`: important source registry explanation, but has outdated/contradictory parts.
- `next_steps_checklist.md`: useful current checklist, but too mixed and should be converted into a maintained status page.
- `taxonomy_notes.md`: useful compact explanation.
- `taxonomy_coverage_notes.md`: useful benchmark taxonomy coverage evidence.
- `real_source_labeling_plan.md`: useful workflow boundaries.
- `real_samples_output_structure.md`: useful data/output policy seed.
- `future_hf_streaming_runbook.md`: still useful as a runbook, but should be renamed/reframed because HF runs have already happened.

### Partially current but needs consolidation

- `classifier_benchmark_report.md` and `classifier_benchmark_audit.md`: both valuable, but overlapping. They should become one benchmark report plus one appendix if detailed coverage is needed.
- `local_classifier_benchmark_dataset_card.md`: useful fixture documentation; should be linked from benchmark docs, not compete with them.
- `local_real_like_sample_report.md`: useful transition evidence, but less central now that FineWeb-Edu and FineMath samples exist.
- `real_sample_next_plan.md`: started as a future plan, then got appended with completed statuses. It now mixes "not run yet" language with "completed" language.
- `real_sample_readiness_checklist.md`: useful as a pre-run checklist, but some parts are outdated after first HF runs.
- `hf_dataset_verification_plan.md`: useful as historical caution, but now conflicts with registry/report state because it still lists FineWeb-Edu/FineMath ids as null.
- `finemath_tiny_sample_plan.md`: completed plan; should be linked from report or moved to archive.
- `chunking_first_findings.md`: early but useful historical note; not current pipeline state.
- `chunking_parameter_sweep.md`: useful old sweep note, but appears stale because it refers to older 30-document benchmark while current benchmark docs say 42 documents.

### Future-only / should not be in current main path

- `probability_profile_schema.md`
- `nll_pilot_candidate_notes.md`
- `nll_scoring_next_steps.md`
- `cleaning_and_boilerplate_notes.md` partially, because it motivates later probability profiling.
- `hf_row_adapter_design.md` partially, because it is future design rather than current quickstart.

These documents are not bad, but they should be labeled as downstream/future so they do not imply that NLL/effective context window is part of the current responsibility.

### Archive candidates

- `night_work_log.md`: too long and iterative for current docs; preserve as historical appendix.
- `chunking_parameter_sweep.md`: archive or appendix after summarizing one key lesson elsewhere.
- `finemath_tiny_sample_plan.md`: archive after the report superseded it.
- `real_sample_next_plan.md`: either rewrite as current source status or archive after extracting useful parts.
- `hf_dataset_verification_plan.md`: archive or update later; currently conflicts with completed sample docs.
- `MESSAGE_TO_TEAM.txt`: convert useful parts into README, then archive as communication note.

## Дублирование

### Pipeline commands duplicated

Same or similar commands appear in:

- `README.md`;
- `CHECK_HOW_TO.md`;
- `MESSAGE_TO_TEAM.txt`;
- `classifier_benchmark_report.md`;
- `real_sample_next_plan.md`;
- `future_hf_streaming_runbook.md`;
- `embedding_baseline_readiness.md`;
- `local_embedding_environment_check.md`.

Problem: user can find multiple command sets and not know which one is safest/current.

Recommendation: keep canonical commands in one quickstart/pipeline map. Reports should cite the command used for reproducibility, but not act as primary run instructions.

### Benchmark docs overlap

Overlap:

- `classifier_benchmark_report.md`: data, pipeline, results, observations, limitations.
- `classifier_benchmark_audit.md`: coverage, easy/tricky cases, limitations, next improvements.
- `local_classifier_benchmark_dataset_card.md`: what benchmark is, categories, limitations, regeneration.
- `chunking_parameter_sweep.md`: benchmark sweep behavior and label limitation.

Recommendation: merge into:

- `docs/benchmark/local_classifier_benchmark.md` as the main benchmark doc;
- `docs/appendix/chunking_parameter_sweep.md` for sweep details;
- maybe a short generated-output status table.

### Real sample docs overlap

Overlap:

- `real_sample_next_plan.md`;
- `dataset_source_registry.md`;
- `future_hf_streaming_runbook.md`;
- `real_sample_readiness_checklist.md`;
- `real_samples_output_structure.md`;
- `fineweb_edu_tiny_sample_report.md`;
- `finemath_tiny_sample_report.md`;
- `real_sample_disagreement_review.md`.

Recommendation: create one current `docs/real_samples/current_real_samples.md` with:

- source status table;
- outputs table;
- FineWeb-Edu/FineMath summaries;
- pointer to detailed reports;
- OpenWebMath optional_later note.

### Embedding docs overlap

Overlap:

- `embedding_baseline_readiness.md`;
- `local_embedding_environment_check.md`;
- embedding section in `classifier_comparison_report.md`;
- embedding items in `next_steps_checklist.md`;
- optional embedding commands in `README.md` and `real_sample_next_plan.md`.

Recommendation: create one `docs/embedding_baseline_plan.md` in Russian:

- goal;
- current script;
- dependencies;
- local environment status;
- no-download rule;
- expected inputs/outputs;
- first safe command;
- risks.

### NLL/probability docs overlap current classifier docs

NLL/probability docs are referenced from:

- `README.md`;
- `MESSAGE_TO_TEAM.txt`;
- `chunk_schema.md`;
- `cleaning_and_boilerplate_notes.md`;
- `probability_profile_schema.md`;
- `nll_pilot_candidate_notes.md`;
- `nll_scoring_next_steps.md`;
- `night_work_log.md`.

Recommendation: move to `docs/future_nll/` or `docs/archive/future_nll/` and leave only one boundary note in current docs.

## Противоречия и confusing points

### 1. Stage2 README says the purpose is preparation for logprob/NLL

Severity: High

`README.md` says the prototype prepares texts for later logprob/NLL and mentions `NLL_eff`, `NLL_full`, `delta`. This is historically understandable, but it now overstates NLL as the destination of the current work.

Why it matters:

Current responsibility is chunking + domain labeling, with next major milestone being MiniLM/Sentence-Transformers embedding classifier. A new teammate or reviewer can think NLL/effective context window is owned here.

Recommended later fix:

Rewrite the main README around:

```text
corpus preparation + domain labeling
next: embedding nearest-label classifier
downstream: NLL/effective context window, owned separately
```

### 2. "HF commands are not yet run" conflicts with completed FineWeb-Edu/FineMath reports

Severity: High

`real_sample_next_plan.md` still says "HF commands are not yet run", but later sections in the same file say FineWeb-Edu and FineMath completed.

Why it matters:

This makes it unclear whether real samples are planned, already executed, or safe to use as current evidence.

Recommended later fix:

Archive this plan or rewrite it as a current status page.

### 3. HF dataset verification plan conflicts with registry and reports

Severity: High

`hf_dataset_verification_plan.md` lists FineWeb-Edu and FineMath `hf_dataset: null` / `hf_config: null`, while `config/dataset_sources.json`, `fineweb_edu_tiny_sample_report.md`, and `finemath_tiny_sample_report.md` contain concrete dataset/config values and completed runs.

Why it matters:

New contributors may not know whether IDs/configs are known, verified, sampled, or still unknown.

Recommended later fix:

Replace this with a source status table or move the old verification plan to archive.

### 4. Dataset source registry mixes planned and completed states

Severity: Medium

`dataset_source_registry.md` says the registry is planning-only and later notes that FineWeb-Edu/FineMath completed. It also has a verification-needed section that is not obviously synchronized with the JSON registry.

Why it matters:

The document is useful but overloaded: planning metadata, source layer definitions, safety rules, current source list, verification state, completed run notes.

Recommended later fix:

Split into:

- `dataset_sources_schema.md`: what fields mean;
- `source_status.md`: current source state and sample status.

### 5. FineMath current role is explained, but not in the main entry point

Severity: Medium

Clear statements exist in:

- `finemath_tiny_sample_plan.md`;
- `dataset_source_registry.md`;
- `real_sample_next_plan.md`;
- `next_steps_checklist.md`;
- `finemath_tiny_sample_report.md` indirectly.

But `README.md` and `CHECK_HOW_TO.md` do not make FineMath's current MVP role obvious.

Why it matters:

A reviewer entering through README can still see FineWeb/FineWeb-Edu first and miss that FineMath is the only active math source.

Recommended later fix:

Add a prominent Russian source-status table:

| Source | Current status | Role |
| --- | --- | --- |
| FineWeb-Edu | tiny sample completed | educational real sample |
| FineMath | tiny sample completed | only current MVP math source |
| OpenWebMath | optional_later | comparison/backup only |

### 6. OpenWebMath optional_later is stated, but not centralized

Severity: Medium

OpenWebMath is correctly described as optional/later in several docs, but it remains listed alongside active candidates in source docs.

Why it matters:

It is easy for a teammate to interpret OpenWebMath as a second current math dataset.

Recommended later fix:

Make `optional_later` a visible status in the source table and separate active MVP sources from later comparison sources.

### 7. NLL boundary is stated in some docs but contradicted by placement/prominence

Severity: Medium

Several files correctly say no NLL/logprob scoring was run. But NLL docs and manifests live in the same active docs/data area, and main README mentions NLL early.

Why it matters:

The documentation surface says "not now" in details but "NLL destination" in first impression.

Recommended later fix:

Keep one "downstream NLL boundary" note in current docs. Move detailed NLL docs to future/archive.

### 8. FineMath report ends with a possible NLL pilot recommendation

Severity: Medium

`finemath_tiny_sample_report.md` says after review, either improve taxonomy/rules lightly or proceed to a very small observed-token NLL pilot. For the current responsibility, this can be read as pulling the next step toward NLL instead of embedding classifier.

Why it matters:

The next major milestone should be embedding nearest-label classification, not NLL.

Recommended later fix:

In a future edit, reframe the next recommendation as:

1. finish manual disagreement review;
2. improve taxonomy/rules only if needed;
3. run MiniLM/Sentence-Transformers embedding baseline;
4. leave NLL as downstream.

## Too long or hard to read

### `real_sample_disagreement_review.md`

Length: about 122 lines but very dense, with many long text previews. It is valuable review evidence but too heavy for quick understanding.

Recommendation:

Keep a short summary in main docs:

- agreement rates;
- top disagreement types;
- implications for embedding classifier.

Move detailed per-chunk previews to appendix.

### `night_work_log.md`

Length: about 190 lines. It is an excellent work log but not onboarding documentation.

Recommendation:

Move to `archive/work_logs/` or `appendix/work_logs/`. Extract only durable decisions into current docs.

### `classifier_benchmark_audit.md` + `classifier_benchmark_report.md`

Together they repeat coverage, results, limitations, and risks.

Recommendation:

Merge into one benchmark doc; keep detailed adversarial-case list in appendix.

### `next_steps_checklist.md`

Useful, but it mixes done/current/immediate/later/do-not-do, including items that became outdated or belong to other areas.

Recommendation:

Replace with a current status board:

- Done;
- Current next;
- Blocked/needs approval;
- Later/out of scope.

## Is the current scope clear?

### Chunking + domain labeling

Partially clear.

Good signals:

- `chunk_schema.md` explains chunk/labeled schemas.
- `classifier_comparison_report.md` explains rule-based, lexical, embedding baseline relationship.
- `real_source_labeling_plan.md` says labels should be reviewed before embedding or NLL.

Weak signals:

- main README still frames the work around NLL.
- no single Russian current pipeline map exists.
- docs are too scattered for a reviewer to see the story quickly.

### MiniLM/Sentence-Transformers as next milestone

Partially clear.

Good signals:

- `embedding_baseline_readiness.md` is focused and relevant.
- `local_embedding_environment_check.md` clearly says real embedding run is not currently safe because dependencies/model are missing.
- `classifier_comparison_report.md` says embedding is the next real step.

Weak signals:

- embedding is called "optional" in many places, which was true earlier but now underplays it as the next major milestone.
- current README does not foreground MiniLM/Sentence-Transformers.
- NLL docs compete for attention.

### NLL/effective context window boundary

Not clear enough at first glance.

Good signals:

- multiple docs say no model inference/logprob/NLL was run;
- NLL docs include caution about downloads and first tiny pilots.

Weak signals:

- README mentions NLL in the opening;
- NLL docs are in active docs folder;
- `nll_pilot_candidates.jsonl` exists under active real samples;
- `night_work_log.md` and `MESSAGE_TO_TEAM.txt` include probability profiling as a prepared path.

Recommended later fix:

Use a visible banner in the new current docs:

```text
Current owner scope: chunking + domain labeling.
Next milestone: embedding nearest-label classifier.
Downstream/out of current scope: NLL/logprob/effective context window.
```

## Proposed future documentation structure

Suggested target structure:

```text
external_samples/stage2_chunking_sample_clean/
  README.md
  docs/
    README.md
    quickstart_ru.md
    pipeline_map.md
    current_status.md
    schema/
      chunk_schema.md
      taxonomy_notes.md
    classifiers/
      rule_based_and_lexical_baselines.md
      embedding_baseline_plan.md
    sources/
      source_status.md
      dataset_sources_schema.md
      real_samples_output_policy.md
    reports/
      fineweb_edu_tiny_sample_report.md
      finemath_tiny_sample_report.md
      classifier_benchmark_report.md
      real_sample_disagreement_summary.md
    runbooks/
      hf_tiny_sample_runbook.md
    appendix/
      real_sample_disagreement_details.md
      chunking_parameter_sweep.md
      local_real_like_sample_report.md
    archive/
      work_logs/
        night_work_log.md
      old_plans/
        finemath_tiny_sample_plan.md
        real_sample_next_plan.md
        hf_dataset_verification_plan.md
    future_nll/
      probability_profile_schema.md
      nll_pilot_candidate_notes.md
      nll_scoring_next_steps.md
```

## What to keep, merge, translate, archive

| File | Current status | Later action |
| --- | --- | --- |
| `README.md` | important but misleading/outdated | rewrite in Russian as main stage2 overview |
| `CHECK_HOW_TO.md` | simple but early | merge into quickstart |
| `MESSAGE_TO_TEAM.txt` | useful historical team note | archive after extracting commands/status |
| `chunk_schema.md` | current | translate/summarize in Russian |
| `taxonomy_notes.md` | current | merge into schema/taxonomy section |
| `taxonomy_coverage_notes.md` | current benchmark evidence | keep as report/appendix |
| `classifier_comparison_report.md` | current and important | translate/summarize; link from current status |
| `embedding_baseline_readiness.md` | current and important | merge with environment check into embedding plan |
| `local_embedding_environment_check.md` | current environment note | merge into embedding plan |
| `fineweb_edu_tiny_sample_report.md` | current report | translate summary; keep detailed report |
| `finemath_tiny_sample_report.md` | current report | translate summary; adjust future recommendation later |
| `real_sample_disagreement_review.md` | current but too detailed | split summary vs appendix |
| `dataset_source_registry.md` | important but mixed state | split source schema vs source status |
| `next_steps_checklist.md` | useful but mixed | replace with current status board |
| `real_source_labeling_plan.md` | useful | keep/translate as workflow |
| `real_samples_output_structure.md` | useful | merge into data/output policy |
| `future_hf_streaming_runbook.md` | useful | rename/reframe as tiny HF sample runbook |
| `classifier_benchmark_report.md` | useful | merge with benchmark audit/dataset card |
| `classifier_benchmark_audit.md` | useful but overlapping | merge or appendix |
| `local_classifier_benchmark_dataset_card.md` | useful fixture docs | merge into benchmark doc |
| `chunking_parameter_sweep.md` | stale/historical | appendix/archive |
| `chunking_first_findings.md` | early historical note | archive or appendix |
| `local_real_like_sample_report.md` | superseded by real samples | appendix |
| `finemath_tiny_sample_plan.md` | completed plan | archive |
| `real_sample_next_plan.md` | mixed future/completed | rewrite or archive |
| `hf_dataset_verification_plan.md` | partially stale | archive or replace with source status |
| `hf_row_adapter_design.md` | future design | keep in appendix/future |
| `cleaning_and_boilerplate_notes.md` | useful but partly future/NLL | keep as future cleaning note |
| `probability_profile_schema.md` | future NLL | move to future_nll |
| `nll_pilot_candidate_notes.md` | future NLL | move to future_nll |
| `nll_scoring_next_steps.md` | future NLL | move to future_nll |
| `night_work_log.md` | historical work log | archive |

## Recommended cleanup order

1. Create a Russian `docs/README.md` or `quickstart_ru.md` as the entry point.
2. Rewrite top-level stage2 `README.md` to state current scope and next milestone.
3. Add `source_status.md` with FineWeb-Edu, FineMath, OpenWebMath statuses.
4. Merge embedding readiness + environment check into one Russian embedding plan.
5. Split real sample disagreement review into summary and appendix.
6. Merge benchmark docs into one benchmark report plus appendix.
7. Move NLL/effective context window docs into `future_nll/`.
8. Archive old plans/logs after extracting current facts.

## Key messages documentation should make obvious

- This stage2 subproject is about chunking and domain labeling.
- Rule-based labels are transparent baseline labels, not final truth.
- Lexical nearest-label is a cheap baseline and comparison point.
- FineWeb-Edu and FineMath tiny samples have been produced and labeled by rule-based/lexical baselines.
- FineMath is the only active math dataset in the current MVP.
- OpenWebMath is optional_later comparison/backup only.
- The next major stage is MiniLM/Sentence-Transformers embedding nearest-label classification.
- No model download, HF/network run, or real embedding run should happen without explicit approval.
- NLL/logprob/effective context window is downstream and not the current responsibility.
