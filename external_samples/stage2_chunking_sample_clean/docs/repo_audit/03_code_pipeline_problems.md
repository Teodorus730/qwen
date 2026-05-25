# Аудит 03: проблемы кода и pipeline stage2

Дата: 2026-05-26

Область аудита: статический инженерный аудит `external_samples/stage2_chunking_sample_clean/scripts` и связанного pipeline. Код не исправлялся. HF streaming, network, model download, model inference, commit и push не запускались.

## Краткий вывод

Stage2 scripts достаточно понятны для MVP: каждый файл маленький, почти все scripts имеют простую CLI-форму `--input/--output`, а pipeline уже покрывает chunking, validation, rule-based labels, lexical labels, comparison, review и source planning.

Но текущая структура больше похожа на набор рабочих CLI-скриптов, чем на устойчивый pipeline library. Перед MiniLM/Sentence-Transformers embedding classifier важно не начинать с "просто запустить модель", а закрепить общие контракты:

- shared JSONL/schema utilities;
- единая логика label tuple/source_type/domain;
- явные режимы для synthetic benchmark vs real samples;
- безопасная no-download модельная политика;
- один canonical command для embedding baseline;
- сравнение rule-based vs lexical vs embedding на одинаковых inputs.

## Карта scripts

### Chunking and loading

- `sample_fineweb_chunks.py`: local JSONL/HF streaming loader, text extraction, normalization, paragraph chunking, optional tokenizer, metadata propagation, output JSONL, stats JSON.

### Classifiers

- `classify_chunks_rule_based.py`: transparent standard-library keyword/rule classifier.
- `classify_chunks_lexical_baseline.py`: no-dependency bag-of-words nearest-label baseline over taxonomy labels.
- `classify_chunks_embedding_baseline.py`: optional Sentence-Transformers nearest-label scaffold, currently safe by `local_files_only=True` and `--dry-run`.

### Validation, inspection, comparison

- `validate_chunks.py`: schema validation, optional strict label requirement.
- `inspect_chunks.py`: compact counts, token stats, previews.
- `evaluate_chunk_labels.py`: benchmark evaluation against `expected_*` fields.
- `compare_label_runs.py`: compare two labeled runs by `chunk_id`.
- `review_label_mismatches.py`: print expected-vs-predicted review blocks.
- `check_label_consistency.py`: compare predicted domain/field/subfield tuples with taxonomy.

### Source planning

- `inspect_dataset_sources.py`: static registry inspection.
- `plan_real_sample_run.py`: print future sample commands.
- `plan_real_source_pipeline.py`: print full source pipeline commands, including rule/lexical/embedding.

### Local regression / future scoring

- `run_local_benchmark_pipeline.py`: end-to-end local synthetic benchmark runner.
- `smoke_test_rule_based_classifier.py`: direct rule-based smoke tests.
- `prepare_probability_profile_manifest.py`: future NLL manifest preparation; not current core responsibility.

## What works well

- Scripts are short enough to audit manually.
- Most tools avoid hidden side effects: they read JSONL, write JSONL, print text summaries.
- The local benchmark path is reproducible without HF/network/model access.
- The rule-based classifier is intentionally transparent and standard-library only.
- The lexical baseline is a useful cheap comparison point before embeddings.
- `classify_chunks_embedding_baseline.py` already has a no-download intent and supports `--dry-run`.
- Planner scripts reduce the chance of accidentally running a full HF job.
- Validation and inspection scripts make generated outputs reviewable.

## Problems and recommendations

### 1. No shared pipeline library; common logic is duplicated

Severity: High before MiniLM

Observed duplication:

- `iter_jsonl` appears in many scripts.
- `write_jsonl` appears in classifiers and manifest generation.
- `label_tuple`, `preview_text`, and null normalization are repeated.
- Registry loading and `source_kind/kind` are duplicated across source scripts.
- Label text construction is duplicated between lexical and embedding classifiers.
- JSON error handling is inconsistent across scripts.

Why it matters:

MiniLM will add another classifier path with the same inputs, labels, outputs, confidence thresholds, comparison needs, and failure modes. If each script owns its own JSONL/taxonomy/label logic, subtle schema drift becomes likely.

Recommendation:

Before the real MiniLM run, extract a small internal helper layer, for example:

- `stage2_io.py`: `iter_jsonl`, `write_jsonl`, JSONL errors with line numbers.
- `stage2_labels.py`: label fields, expected fields, null normalization, label tuple, label text.
- `stage2_taxonomy.py`: load taxonomy, validate label schema, label-to-source_type mapping.
- `stage2_registry.py`: source registry loading and source kind.

Keep it simple; this does not need to become a package yet.

### 2. `source_type` and `domain/field/subfield` responsibilities are not fully stable

Severity: High before MiniLM

Examples:

- Chunker writes `source_type` from input metadata, guessed dataset label, or CLI.
- Rule-based classifier may overwrite `source_type` based on content.
- Lexical classifier preserves existing `source_type` if it is present and not unknown; otherwise it maps a domain tuple back to source_type through `LABEL_TO_SOURCE_TYPE`.
- Embedding classifier currently preserves `source_type` as existing value or unknown, but does not apply the same label-to-source_type mapping as lexical.

Why it matters:

Real samples show the exact problem: FineMath as a dataset/source can contain educational lessons, math exercises, product/resource listings, or web noise. `source_type=math` from the dataset and `domain=education` from content are both meaningful but not the same layer.

Before MiniLM, the team needs one explicit policy:

- Should classifiers overwrite `source_type`, or only fill `domain/field/subfield`?
- Should dataset/source metadata live in a separate field from classifier source_type?
- Should embedding output include both `input_source_type` and `predicted_source_type`?

Recommendation:

Mandatory before MiniLM: define this contract in schema and implement consistent classifier behavior. The safest direction is:

- keep dataset/source metadata stable;
- treat domain/field/subfield as classifier output;
- if predicted source_type is still useful, store it separately or make overwrite behavior explicit.

### 3. Synthetic benchmark and real samples need different validation/evaluation modes

Severity: High before MiniLM

Current behavior:

- Synthetic benchmark has `expected_*` labels and can use strict `--require-labels`.
- Real samples often have low-confidence labels with `domain/field/subfield = null`.
- `future_hf` planner prints `validate_chunks.py --require-labels` for rule and lexical outputs, but real sample reports already note that strict validation fails for low-confidence real labels.
- `evaluate_chunk_labels.py` is benchmark-only because it requires `expected_*`.

Why it matters:

Embedding outputs will likely also include low-confidence/null labels, especially on real noisy chunks. If synthetic and real modes are not separated, a valid real result can look like a failed pipeline.

Recommendation:

Mandatory before MiniLM:

- Define two validation modes:
  - benchmark/gold mode: require non-null labels and expected-label evaluation;
  - real-sample mode: require valid schema, allow null classifier labels, report low-confidence counts.
- Adjust future pipeline docs/commands later so real sample validation does not imply `--require-labels`.
- Add a comparison mode that accepts null labels as a normal outcome.

### 4. Hardcoded paths and names make the pipeline easy to run only from one working directory

Severity: Medium

Examples:

- Default chunker input/output:
  - `examples/local_docs.jsonl`
  - `data_samples/fineweb_chunks_sample.jsonl`
  - `data_samples/run_stats.json`
- Benchmark runner defaults:
  - `examples/local_docs_classifier_benchmark.jsonl`
  - `data_samples/classifier_benchmark_chunks.jsonl`
  - `data_samples/classifier_benchmark_labeled.jsonl`
- Planner scripts hardcode `data_samples\\real_samples`, `taxonomy\\simple_domain_labels.json`, and script paths.
- Embedding default model is `sentence-transformers/all-MiniLM-L6-v2`.
- `sample_fineweb_chunks.py` defaults to `HuggingFaceFW/fineweb` and `sample-10BT`.

Why it matters:

These defaults are okay for local MVP, but they make current/old outputs easy to overwrite and hide which sample family is being processed. They also assume running from the stage2 root.

Recommendation:

Before MiniLM, make the embedding command explicit and output to a new non-overwriting path. Later, consider a config-driven command or small `pipeline_config.json` for canonical paths.

Can defer:

- full packaging;
- full project-root path resolver;
- cross-platform command rendering.

### 5. CLI usability is consistent but still rough

Severity: Medium

Good:

- Most scripts use `argparse`.
- Required `--input`/`--output` is clear.
- Dry-run exists for lexical and embedding.
- Planner scripts print commands instead of executing them.

Problems:

- No shared `--labels` default for taxonomy, so commands are verbose and can drift.
- No `--output-format json` for summaries; reports are text-only and harder to consume.
- `--top-k` in lexical dry-run reports only the number, not actual top-k label matches.
- Embedding classifier has no `--top-k`, `--batch-size`, `--device`, `--normalize-embeddings`, or `--model-cache-dir` options.
- Planner commands use literal `python` instead of the current interpreter; this has already been a Windows issue in earlier docs.
- Planner scripts print command strings with simple joining, not robust shell quoting.
- There is no single CLI entry point like `stage2_pipeline.py local-benchmark` or `stage2_pipeline.py classify`.

Recommendation:

Mandatory before MiniLM:

- add a safe, documented embedding command;
- add `--model` guidance that prefers a local path or guaranteed local cache;
- add batch-related options if real embedding inference is expected.

Can defer:

- single command wrapper;
- structured JSON summaries;
- shell quoting polish.

### 6. Error handling is uneven

Severity: Medium

Good:

- Validators and some comparison scripts report missing files and JSON line numbers.
- Embedding classifier catches missing `sentence-transformers` and model-load failures.
- Chunker warns on missing text fields.

Weak spots:

- Some classifiers read JSONL with plain `json.loads` and no file/line error context.
- Chunker local JSONL parsing has no explicit JSON error handling.
- Chunker HF path has no explicit handling for `load_dataset` errors, auth issues, missing configs, or network failures.
- Chunker parameters are not validated for relationships such as `min <= target <= max`.
- Output writes are not atomic; a failed run can leave partial JSONL.
- Empty outputs can be valid but may not be clearly marked as failure.
- Tokenizer loading with `AutoTokenizer.from_pretrained(tokenizer_name)` can attempt a download if a remote model name is passed.

Why it matters:

MiniLM introduces a heavier failure surface: missing dependency, absent local model, unexpected model version, slow CPU, memory, accidental download risk.

Recommendation:

Mandatory before MiniLM:

- make embedding model loading fail closed with no-download behavior;
- prefer local model paths or explicit local cache checks;
- include clear error messages and exit codes;
- add line-number JSONL errors to shared IO helpers.

Can defer:

- atomic writes;
- complete HF exception taxonomy;
- retry logic.

### 7. Rule-based classifier is readable but tuned to synthetic benchmark patterns

Severity: Medium

Observations:

- Rules are explicit functions and order is easy to inspect.
- Some rule order choices are important, for example forum before boilerplate, commercial before science/education, math before code.
- Rules use keyword counts and broad substring checks.
- Synthetic benchmark achieves perfect accuracy, but real FineWeb-Edu/FineMath agreement with lexical is much lower.

Why it matters:

Synthetic benchmark is a regression check, not quality evidence. MiniLM should be compared against rule-based/lexical on real tiny samples, not optimized against the synthetic benchmark only.

Recommendation:

Before MiniLM:

- keep rule-based as baseline, not target truth;
- compare MiniLM on FineWeb-Edu and FineMath tiny samples;
- include disagreement review, especially math-vs-education and boilerplate/noise cases.

Can defer:

- large rule refactor;
- span-level boilerplate detection;
- new rule families unless manual review shows a blocking issue.

### 8. Lexical and embedding nearest-label classifiers are conceptually aligned but not behaviorally identical

Severity: High before MiniLM

Shared idea:

- both classify chunk text against `taxonomy/simple_domain_labels.json`;
- both use label text from domain/field/subfield/description/keywords;
- both have low-confidence methods.

Differences:

- lexical maps label tuples back to source_type when source_type is missing/unknown;
- embedding does not use the same mapping;
- lexical supports `--top-k` only as dry-run metadata, not output;
- embedding encodes labels once but encodes chunks one by one;
- embedding stores only best label, no alternatives;
- confidence thresholds are different and uncalibrated.

Why it matters:

For MiniLM evaluation, we need to know whether differences come from embeddings or from classifier-output policy differences.

Recommendation:

Mandatory before MiniLM:

- unify nearest-label output schema for lexical and embedding;
- decide whether to store top-k alternatives;
- decide source_type behavior;
- document confidence as similarity, not probability;
- run comparison using the same input chunks and taxonomy.

### 9. Current embedding baseline is a good scaffold but not production-ready for the next milestone

Severity: High before MiniLM

Good:

- has `--dry-run`;
- does not install dependencies;
- attempts `SentenceTransformer(..., local_files_only=True)`;
- exits if local-only behavior is not supported;
- uses the same taxonomy as lexical baseline.

Gaps:

- no batching, which will be slow even for modest samples;
- no `--device`;
- no explicit local-model-path-first workflow;
- no local cache inspection;
- no top-k output;
- no structured summary of low-confidence/null labels;
- no tests that mock a model encoder;
- no normalization/cosine behavior alignment with Sentence-Transformers recommended usage;
- no protection if a future dependency version changes `local_files_only` behavior beyond the caught TypeError.

Mandatory before MiniLM:

- require explicit approval for dependency install/model preparation;
- use local model path or verified local cache;
- add/test no-download behavior;
- add batch encoding;
- add output schema and comparison plan.

Can defer:

- GPU support beyond optional `--device`;
- large-scale inference optimizations;
- approximate nearest-neighbor index;
- calibration of confidence.

### 10. Registry and planner scripts are useful, but source statuses are not executable contracts

Severity: Medium

Observations:

- `inspect_dataset_sources.py` checks basic registry shape.
- `plan_real_sample_run.py` and `plan_real_source_pipeline.py` generate suggested commands.
- Warnings are printed for `needs_verification` and missing `text_field`.

Weaknesses:

- Planners do not enforce source status; they print commands even for sources that should remain optional/later.
- `openwebmath` can be planned like any other source if selected.
- Generated command text is not robustly quoted.
- Planner full pipeline includes embedding command as commented text, but not a safe MiniLM readiness check.
- Planner full pipeline currently suggests strict label validation for real outputs.

Recommendation:

Before MiniLM:

- add a source status concept to docs/config before using planners as authoritative;
- explicitly mark FineMath active current MVP math source and OpenWebMath optional_later;
- do not include OpenWebMath in active current commands.

Can defer:

- turning planners into an execution engine;
- advanced registry validation.

### 11. Pipeline outputs are easy to overwrite

Severity: Medium

Examples:

- Local benchmark runner writes stable paths by default.
- Chunker default writes `fineweb_chunks_sample.jsonl` and `run_stats.json`.
- Classifier scripts overwrite `--output`.

Why it matters:

When MiniLM outputs are added, it will be easy to overwrite rule/lexical/embedding outputs from different dates, thresholds, or model versions.

Recommendation:

Before MiniLM:

- choose explicit embedding output naming with model/threshold/date or run id in stats/report;
- store model name, threshold, text truncation length, dependency version if available;
- avoid overwriting canonical rule/lexical outputs during experiments.

Can defer:

- full run directory system;
- metadata database.

### 12. NLL/probability scripts are outside current scope but still live in active scripts

Severity: Low for code quality, Medium for project scope

`prepare_probability_profile_manifest.py` is safe and does not run models, but it belongs to downstream NLL/logprob work, not the current chunking/domain-labeling responsibility.

Why it matters:

It competes with MiniLM as the next technical milestone and can confuse future work ordering.

Recommendation:

No code action is needed before MiniLM. Later, move or clearly label NLL/probability scripts as downstream/future.

## Synthetic benchmark vs real samples

### Synthetic benchmark assumptions

- Has `expected_*` labels.
- Uses stable local fixtures.
- Allows strict validation with `--require-labels`.
- Supports accuracy metrics.
- Useful for regression and rule breakage.

### Real sample assumptions

- No human gold labels.
- `domain/field/subfield` may be null by design.
- Rule-based and lexical disagreement is expected and useful.
- Source type from dataset is not the same as predicted domain.
- Manual review is part of the pipeline.

### Required split

Before MiniLM, treat these as separate modes:

| Mode | Inputs | Valid output | Main metric |
| --- | --- | --- | --- |
| Synthetic benchmark | local fixtures with `expected_*` | non-null labels expected | accuracy vs expected |
| Real sample review | FineWeb-Edu/FineMath chunks | null labels allowed | agreement/disagreement and manual review |
| Embedding baseline | same chunks + taxonomy + local MiniLM | similarity labels and low-confidence nulls | comparison vs rule/lexical/manual review |

## MiniLM readiness assessment

Current readiness: partial.

Ready:

- chunk JSONL format exists;
- taxonomy file exists;
- lexical nearest-label baseline gives a conceptual template;
- embedding script scaffold exists;
- comparison scripts exist;
- real tiny FineWeb-Edu and FineMath samples exist;
- no-download concern is already recognized.

Not ready enough:

- source_type/domain output contract is ambiguous;
- real validation mode is not cleanly separated from benchmark mode;
- embedding output schema is minimal;
- no batching/device/local path workflow;
- no top-k alternatives;
- no mock tests for embedding classifier;
- no canonical command/report for MiniLM run;
- no clear source-status gating that keeps OpenWebMath optional_later.

## Mandatory changes before MiniLM

1. Define classifier output contract:
   - stable source metadata vs predicted source_type;
   - domain/field/subfield null behavior;
   - confidence meaning.

2. Unify nearest-label behavior:
   - shared taxonomy label text;
   - same low-confidence policy;
   - optional top-k alternatives;
   - consistent source_type handling.

3. Add safe model-loading plan:
   - no network/model download without explicit approval;
   - prefer local model path or verified local cache;
   - fail closed if local files are missing.

4. Add batch/device/text truncation controls:
   - `--batch-size`;
   - optional `--device`;
   - documented `--text-chars`;
   - record these settings in output summary.

5. Separate validation/evaluation modes:
   - synthetic benchmark can require labels;
   - real samples can have null low-confidence labels;
   - MiniLM comparison should not fail just because a label is null.

6. Add a canonical MiniLM comparison workflow:
   - input: same chunks used by rule/lexical;
   - output: embedding labeled JSONL;
   - comparison: rule vs lexical vs embedding;
   - manual review sample of disagreements.

7. Add at least lightweight tests:
   - mock encoder returns deterministic vectors;
   - low-confidence path;
   - overwrite-existing-labels behavior;
   - source_type/domain contract.

## Changes that can be postponed

- Full package/module restructure.
- Single CLI entry point for all stage2 commands.
- Atomic output writes.
- JSON summary output for every script.
- Robust shell quoting in planners.
- Full HF exception handling.
- Span-level boilerplate detection.
- OpenAlex-like taxonomy migration.
- Large-scale embedding optimization.
- NLL/logprob/effective context window scripts.
- OpenWebMath comparison.

## Suggested future pipeline after MiniLM is prepared

```text
examples/local_docs_classifier_benchmark.jsonl
  -> sample_fineweb_chunks.py
  -> classifier_benchmark_chunks.jsonl
  -> rule-based labels
  -> lexical nearest-label labels
  -> embedding nearest-label labels
  -> compare/evaluate/report

data_samples/real_samples/fineweb_edu_sample10bt_chunks.jsonl
data_samples/real_samples/finemath_chunks.jsonl
  -> rule-based labels
  -> lexical nearest-label labels
  -> embedding nearest-label labels
  -> compare rule/lexical/embedding
  -> manual disagreement review
```

OpenWebMath should stay out of this current path unless explicitly approved later as optional comparison.

## Severity summary

High before MiniLM:

- no shared JSONL/taxonomy/label utilities;
- ambiguous `source_type` vs domain classifier contract;
- synthetic vs real validation modes mixed;
- lexical and embedding output behavior not aligned;
- embedding script scaffold needs batching/local-model workflow/tests.

Medium:

- hardcoded paths and names;
- rough CLI ergonomics;
- uneven error handling;
- planners print commands that are not always current-safe;
- outputs can be overwritten.

Low:

- scripts are not packaged;
- summaries are text-only;
- future NLL helper exists in active scripts.

## Final recommendation

Do not start the MiniLM milestone by running a model. First stabilize the classifier contract and comparison path. The actual embedding implementation can stay small, but the output schema, no-download behavior, source_type/domain policy, and synthetic-vs-real evaluation modes should be clear before any real MiniLM/Sentence-Transformers run.
