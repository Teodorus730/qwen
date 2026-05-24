# Night work log

## Context

Stage 2 prepares small, structured chunk samples for later data selection work. The current local pipeline reads raw/local JSONL documents, creates logical chunks with dataset/source_type metadata, applies transparent rule-based baseline labels, and prepares the path toward embedding/nearest-label labels and observed-token logprob/NLL profiling.

## Initial inventory

- `git status --short`: no modified or untracked files were reported before edits.
- Existing scripts:
  - `scripts/sample_fineweb_chunks.py`: local/HF chunk sampler with paragraph splitting, simple token fallback, local metadata overrides, and chunk stats.
  - `scripts/classify_chunks_rule_based.py`: transparent standard-library rule-based baseline classifier.
- Existing data samples:
  - `examples/local_docs.jsonl`
  - `examples/local_docs_edge_cases.jsonl`
  - `data_samples/fineweb_chunks_sample.jsonl`
  - `data_samples/run_stats.json`
  - `data_samples/edge_case_chunks_sample.jsonl`
  - `data_samples/edge_case_chunks_labeled.jsonl`
  - `data_samples/edge_case_run_stats.json`
- Current known schema:
  - chunks have `chunk_id`, `dataset`, `source_type`, `domain`, `field`, `subfield`, `confidence`, `token_count`, and `text`;
  - labeled chunks add `label_method`;
  - local raw rows have at least `text`, optionally `dataset` and `source_type`;
  - benchmark `expected_*` fields were requested but not yet propagated by the chunker.

Current pipeline in plain words:

- Input files: local JSONL examples or optional HF streaming rows.
- Scripts: chunker creates JSONL chunks; rule-based classifier fills source_type/domain/field/subfield/confidence/label_method.
- Output files: chunk JSONL plus stats JSON; labeled chunk JSONL after classification.
- What works: local chunking, basic metadata, short chunk filtering, rule-based baseline, existing edge-case sample.
- Missing before night work: schema docs, validation/inspection, expected/golden labels, evaluator, benchmark runner, optional embedding baseline, smoke tests, and probability-profiling preparation docs/stub.

## Steps completed

### Step 0: inventory

- Files created/modified: none.
- Commands run:
  - `git status --short`
  - recursive listing of `external_samples/stage2_chunking_sample_clean`
  - read README, CHECK_HOW_TO, MESSAGE_TO_TEAM, chunking findings, chunker, and classifier.
- Result:
  - repository status output was clean;
  - stage2 directory already had base scripts, examples, docs, and generated samples;
  - README/CHECK/MESSAGE terminal display showed mojibake in PowerShell output, but UTF-8 reads of docs/scripts were usable.
- Problems/blockers: none.
- Next action: add docs and local utility scripts.

### Steps 1-8: docs, utilities, taxonomy, optional embedding baseline

- Files created:
  - `docs/night_work_log.md`
  - `docs/chunk_schema.md`
  - `docs/taxonomy_notes.md`
  - `docs/next_steps_checklist.md`
  - `scripts/inspect_chunks.py`
  - `scripts/validate_chunks.py`
  - `scripts/classify_chunks_embedding_baseline.py`
  - `taxonomy/simple_domain_labels.json`
- Files modified:
  - `README.md`
- Commands run:
  - `python scripts\inspect_chunks.py --input data_samples\edge_case_chunks_labeled.jsonl --limit 13`
  - `python scripts\validate_chunks.py --input data_samples\edge_case_chunks_sample.jsonl`
  - `python scripts\validate_chunks.py --input data_samples\edge_case_chunks_labeled.jsonl --require-labels`
  - `python scripts\classify_chunks_embedding_baseline.py --help`
  - `python scripts\classify_chunks_embedding_baseline.py --input data_samples\edge_case_chunks_sample.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\edge_case_chunks_embedding_labeled.jsonl --dry-run`
- Result:
  - inspection found 13 labeled edge-case chunks;
  - source_type counts: `boilerplate_or_noise=4`, `educational=4`, `code=1`, `commercial_product=1`, `forum_qa=1`, `math=1`, `unknown=1`;
  - token_count stats: min 85, mean 128.23, median 114, max 180;
  - validation passed for both sample and labeled edge-case JSONL;
  - embedding baseline help works;
  - embedding baseline dry-run works with 13 records and 16 labels and does not load a model.
- Problems/blockers:
  - plain `python` resolves first to `C:\Users\pervo\AppData\Local\Microsoft\WindowsApps\python.exe` and fails with `Системе не удается найти указанный путь`;
  - used working interpreter `C:\Users\pervo\AppData\Local\Python\bin\python.exe` for local runs.
- Next action: build synthetic classifier benchmark and expected-label plumbing.

### Steps 9-12: synthetic benchmark, expected-label plumbing, evaluator, classifier improvement loop

- Files created:
  - `examples/local_docs_classifier_benchmark.jsonl`
  - `scripts/evaluate_chunk_labels.py`
  - `data_samples/classifier_benchmark_chunks.jsonl`
  - `data_samples/classifier_benchmark_labeled.jsonl`
  - `data_samples/classifier_benchmark_run_stats.json`
- Files modified:
  - `scripts/sample_fineweb_chunks.py`
  - `scripts/classify_chunks_rule_based.py`
- Commands run:
  - `python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_classifier_benchmark.jsonl --max-docs 100 --out data_samples\classifier_benchmark_chunks.jsonl --stats-out data_samples\classifier_benchmark_run_stats.json`
  - `python scripts\validate_chunks.py --input data_samples\edge_case_chunks_sample.jsonl`
  - `python scripts\validate_chunks.py --input data_samples\classifier_benchmark_chunks.jsonl`
  - `python scripts\classify_chunks_rule_based.py --input data_samples\classifier_benchmark_chunks.jsonl --output data_samples\classifier_benchmark_labeled.jsonl`
  - `python scripts\evaluate_chunk_labels.py --input data_samples\classifier_benchmark_labeled.jsonl`
- Result:
  - benchmark has 30 synthetic documents and 30 default chunks;
  - controlled `expected_*` metadata is preserved by the chunker;
  - old `edge_case_chunks_sample.jsonl` still validates;
  - evaluator reports expected vs predicted metrics and source_type confusion.
- Problems/blockers:
  - first benchmark draft produced only 8 chunks because many docs were below default `min_chunk_tokens=80`; expanded benchmark text and reran;
  - PowerShell rewrite added a UTF-8 BOM, so `sample_fineweb_chunks.py` now reads local JSONL with `utf-8-sig`;
  - classifier summary crashed on Python 3.14 when sorting Counter keys containing both `None` and strings; fixed summary serialization.
- Next action: add runner, reports, smoke tests, sweep, and probability-profile prep.

## Iteration notes

Initial benchmark metrics before classifier rule improvements:

- records_evaluated: 30
- source_type_accuracy: 0.5333
- domain_accuracy: 0.3667
- field_accuracy: 0.3667
- subfield_accuracy: 0.5667
- full_label_accuracy: 0.3667

Main mismatch summary:

- algebra hit boilerplate because `terms` was too loose;
- product page hit code because `returns` matched `return`;
- biology/physics/environment/infrastructure collapsed to generic education;
- legal/government, wiki/reference, and news had no dedicated rules;
- math with code hit code because code ran before math;
- useful article plus footer noise sometimes became boilerplate.

Rule changes:

- added robust Counter summary printing;
- moved mixed-language and forum Q&A early;
- added legal/government, news, wiki/reference, science subfield, infrastructure, algebra, and calculus rules;
- made boilerplate require stronger evidence;
- made commercial and code rules less substring-aggressive;
- added markdown-heavy developer-doc signals.

Final benchmark metrics:

- records_evaluated: 30
- source_type_accuracy: 1.0000
- domain_accuracy: 1.0000
- field_accuracy: 1.0000
- subfield_accuracy: 1.0000
- full_label_accuracy: 1.0000

Remaining mismatches:

- none on the default synthetic benchmark.

Benchmark fairness note:

- the benchmark is useful for smoke/regression checks, but it is synthetic and probably easier than real web data. The target-120 chunking stress run showed that document-level expected labels become unfair when generic benchmark context paragraphs split into separate chunks.

### Steps 13-20: runner, reports, smoke tests, sweep, cleaning/probability preparation

- Files created:
  - `scripts/run_local_benchmark_pipeline.py`
  - `scripts/smoke_test_rule_based_classifier.py`
  - `scripts/prepare_probability_profile_manifest.py`
  - `docs/classifier_benchmark_report.md`
  - `docs/cleaning_and_boilerplate_notes.md`
  - `docs/local_classifier_benchmark_dataset_card.md`
  - `docs/probability_profile_schema.md`
  - `docs/chunking_parameter_sweep.md`
  - `data_samples/probability_profile_manifest.jsonl`
  - sweep outputs under `data_samples/classifier_benchmark_*maxdocs*` and `*target120*`
- Files modified:
  - `.gitignore` to ignore `__pycache__/` and `*.pyc`;
  - `MESSAGE_TO_TEAM.txt`
- Commands run:
  - `python scripts\run_local_benchmark_pipeline.py`
  - `python scripts\smoke_test_rule_based_classifier.py`
  - `python scripts\prepare_probability_profile_manifest.py --input data_samples\classifier_benchmark_labeled.jsonl --output data_samples\probability_profile_manifest.jsonl --model-name Qwen/Qwen2.5-0.5B --window-sizes 256,512,1024,2048`
  - chunking sweep for max-docs 20, max-docs 40, and target/min/max token variant.
- Result:
  - local benchmark runner completed successfully;
  - benchmark labeled JSONL validated with `--require-labels`;
  - smoke test passed 9 cases;
  - probability manifest wrote 30 pending records;
  - sweep stats:
    - default/max-docs 100: 30 docs, 30 chunks, min/mean/max 129/144.87/180;
    - max-docs 20: 20 docs, 20 chunks, min/mean/max 129/144.60/180;
    - max-docs 40: 30 docs, 30 chunks, min/mean/max 129/144.87/180;
    - target 120: 30 docs, 54 chunks, min/mean/max 60/75.72/118; target-120 full_label_accuracy 0.4630 due document-level expected labels on generic split chunks.
- Problems/blockers:
  - sandbox policy rejected recursive deletion of the generated `scripts/__pycache__`; added Python ignore patterns to `.gitignore` so it no longer appears in git status.
- Next action: final status and chat report.

## Final status

- Files created: schema/taxonomy/checklist/benchmark/report/cleaning/probability docs; inspect/validate/evaluate/runner/smoke/embedding/manifest scripts; synthetic benchmark JSONL; generated local benchmark outputs; probability manifest.
- Files modified: `.gitignore`, `README.md`, `MESSAGE_TO_TEAM.txt`, `scripts/sample_fineweb_chunks.py`, `scripts/classify_chunks_rule_based.py`.
- Validation results:
  - old edge-case sample validates;
  - new benchmark chunks validate;
  - new benchmark labeled output validates with `--require-labels`;
  - pipeline runner completed end to end.
- Benchmark results:
  - default synthetic benchmark full_label_accuracy: 1.0000;
  - target-120 stress benchmark full_label_accuracy: 0.4630, documented as benchmark-label limitation.
- Smoke test results:
  - `SMOKE TEST PASSED: 9 cases`.
- Blockers:
  - plain `python` WindowsApps shim fails; working interpreter is `C:\Users\pervo\AppData\Local\Python\bin\python.exe`;
  - recursive deletion of generated `__pycache__` was blocked by sandbox policy, handled by `.gitignore`.
- Remaining limitations:
  - benchmark is synthetic and may be too easy;
  - rule-based labels are not final taxonomy;
  - boilerplate is not cleaned at span level;
  - embedding baseline was only dry-run tested;
  - no HF streaming, model download, logits, training, or heavy processing was run.
- Suggested commit grouping:
  - docs/schema/utilities;
  - taxonomy and optional embedding baseline;
  - synthetic benchmark and evaluator;
  - rule-based classifier improvements and smoke test;
  - pipeline runner and benchmark outputs/report;
  - cleaning and probability profiling preparation.
