# Аудит 04: data artifacts и git hygiene stage2

Дата: 2026-05-26

Область аудита: `examples`, `data_samples`, `data_samples/real_samples`, generated outputs, cache/temp files, old sweep outputs, `config`, `taxonomy` внутри `external_samples/stage2_chunking_sample_clean`. Файлы не удалялись, `.gitignore` не менялся, HF/network не запускались, commit/push не выполнялись.

## Краткий вывод

Stage2 пока не раздут по размеру: текущие tracked data artifacts занимают около 1 MB, ignored HF cache около 39 KB, `__pycache__` около 15 KB. Проблема не в объеме, а в policy и читаемости:

- в `data_samples/` смешаны canonical fixtures, generated benchmark outputs, old sweep outputs, tiny real samples, NLL future artifacts и run stats;
- старые sweep outputs сейчас untracked, но не ignored, поэтому их легко случайно добавить в commit;
- HF cache не попал в tracked files и сейчас ignored, это хорошо;
- tiny real samples FineWeb-Edu/FineMath полезны для воспроизводимости и будущего MiniLM comparison, но требуют явной privacy/license policy;
- naming convention частично понятна, но не отделяет `canonical`, `generated`, `sweep`, `real`, `future_nll`.

## Inventory

### `examples/`

Tracked files:

| File | Records | Size | Role |
| --- | ---: | ---: | --- |
| `local_docs.jsonl` | 5 | 20 KB | ранний локальный smoke input |
| `local_docs_edge_cases.jsonl` | 12 | 10 KB | synthetic edge cases |
| `local_docs_classifier_benchmark.jsonl` | 42 | 46 KB | canonical synthetic benchmark input |
| `local_real_like_mini_sample.jsonl` | 10 | 6 KB | local real-like input fixture |

Assessment:

- These are small, deterministic, and useful as fixtures.
- They should stay tracked.
- The naming is mostly understandable, but `local_docs.jsonl` is vague and looks less current than the benchmark fixture.

Recommendation:

- Keep all current `examples/*.jsonl` tracked.
- Later, add a README/table that marks each as `smoke`, `edge_cases`, `benchmark`, or `local_real_like`.

### `config/`

Tracked files:

- `config/dataset_sources.json`, 5 KB.

Assessment:

- Should be tracked.
- It is a core planning/config artifact.
- Current issue is not git hygiene but status clarity: sources mix active, planned, sampled, and optional_later.

Recommendation:

- Keep tracked.
- Later, add explicit status fields or a derived source-status table.
- Preserve `OpenWebMath` as `optional_later`, not active current MVP.

### `taxonomy/`

Tracked files:

- `taxonomy/simple_domain_labels.json`, 5 KB, 16 labels.

Assessment:

- Should be tracked.
- It is central to lexical and future embedding nearest-label classifiers.
- Small and reproducible.

Recommendation:

- Keep tracked.
- Before MiniLM, treat taxonomy as an explicit versioned input; embedding outputs should record which taxonomy file/version was used.

### `data_samples/` root

Tracked canonical-ish files:

| File | Records/lines | Size | Assessment |
| --- | ---: | ---: | --- |
| `classifier_benchmark_chunks.jsonl` | 42 | 51 KB | useful generated benchmark output; can be regenerated |
| `classifier_benchmark_labeled.jsonl` | 42 | 53 KB | useful rule-based benchmark output; can be regenerated |
| `classifier_benchmark_lexical_labeled.jsonl` | 42 | 54 KB | useful lexical benchmark output; useful for comparison |
| `classifier_benchmark_run_stats.json` | 20 lines | 510 B | useful run stats |
| `edge_case_chunks_sample.jsonl` | 13 | 11 KB | small historical/generated edge output |
| `edge_case_chunks_labeled.jsonl` | 13 | 12 KB | small historical/generated edge output |
| `edge_case_run_stats.json` | 17 lines | 413 B | small stats |
| `fineweb_chunks_sample.jsonl` | 36 | 28 KB | early chunk sample; name may be misleading if local/smoke |
| `run_stats.json` | 17 lines | 424 B | early smoke stats |
| `probability_profile_manifest.jsonl` | 30 | 12 KB | future NLL/probability artifact, not current owner scope |
| `README.md` | 5 lines | 383 B | too short for current policy |

Assessment:

- Size is fine.
- The root `data_samples/` is overloaded.
- Some files are canonical enough to keep for reproducibility, but others are historical or downstream.
- `probability_profile_manifest.jsonl` is not dangerous, but it belongs to future NLL scope and should not be prominent in current corpus-labeling work.

Recommendation:

- Keep a small canonical generated benchmark set tracked if the team wants reproducible expected outputs.
- Later, move old smoke outputs and future NLL manifests to archive/future areas or regenerate-on-demand.
- Expand `data_samples/README.md` into a real data policy.

### `data_samples/real_samples/`

Tracked files:

| Group | Files | Records | Approx size | Role |
| --- | --- | ---: | ---: | --- |
| FineWeb-Edu | chunks, rule labels, lexical labels, stats | 44 each output | 249 KB total | first real educational tiny sample |
| FineMath | chunks, rule labels, lexical labels, stats | 101 each output | 300 KB total | current MVP math tiny sample |
| local_real_like | chunks, rule labels, lexical labels, stats | 10 each output | 24 KB total | generated from local fixture |
| NLL pilot | `nll_pilot_candidates.jsonl` | 20 | 33 KB | future NLL candidate manifest |

Assessment:

- FineWeb-Edu and FineMath tiny samples are the most valuable real artifacts for the next MiniLM classifier comparison.
- FineMath is the only current MVP math source.
- OpenWebMath has no sample here, which is good for current scope.
- These files include raw text chunks from real web/HF sources, so tracked status should depend on privacy/license review.
- Local real-like generated outputs are less valuable than the input fixture because they can be regenerated.
- `nll_pilot_candidates.jsonl` contains text and belongs to downstream NLL work, not current MiniLM milestone.

Recommendation:

- Keep FineWeb-Edu and FineMath tiny sample outputs tracked only if privacy/license review says tiny excerpts are acceptable for the project repo.
- If review is uncertain, keep only stats/reports tracked and keep raw chunk JSONL local or ignored later.
- Keep FineMath tiny sample as current MVP math evidence.
- Do not add OpenWebMath samples unless optional_later comparison is explicitly approved.
- Move or mark NLL pilot candidates as future/downstream.

### Untracked old sweep outputs

Currently untracked:

| File | Records/lines | Size | Assessment |
| --- | ---: | ---: | --- |
| `classifier_benchmark_chunks_maxdocs20.jsonl` | 20 | 25 KB | temporary sweep output |
| `classifier_benchmark_chunks_maxdocs40.jsonl` | 30 | 38 KB | temporary sweep output |
| `classifier_benchmark_chunks_target120.jsonl` | 54 | 47 KB | temporary sweep output |
| `classifier_benchmark_labeled_target120.jsonl` | 54 | 50 KB | temporary sweep output |
| `classifier_benchmark_run_stats_maxdocs20.json` | 17 lines | 427 B | temporary sweep stats |
| `classifier_benchmark_run_stats_maxdocs40.json` | 17 lines | 428 B | temporary sweep stats |
| `classifier_benchmark_run_stats_target120.json` | 17 lines | 427 B | temporary sweep stats |

Assessment:

- These are old sweep/generated outputs.
- They are useful as temporary evidence for `chunking_parameter_sweep.md`, but should not be committed as root-level data artifacts.
- They currently clutter `git status`.

Recommendation:

- Do not commit these raw sweep outputs unless the team decides to preserve a reproducibility appendix.
- Later options:
  - delete them after extracting summary to docs;
  - move to `data_samples/sweeps/` and ignore by default;
  - keep only `chunking_parameter_sweep.md` plus run stats, not full JSONL.

### Ignored cache/temp files

Ignored by current `.gitignore`:

| Path | Files | Size | Pattern |
| --- | ---: | ---: | --- |
| `data_samples/hf_cache_test/` | 11 | 39 KB | explicit stage2 path |
| `scripts/__pycache__/` | 1 | 15 KB | `__pycache__/` |

Assessment:

- Cache did not enter tracked files, good.
- Current cache is small, but Hugging Face cache can grow very quickly if real datasets/models are used.
- `__pycache__` is correctly ignored.

Recommendation:

- Keep all HF/model/cache directories untracked and ignored.
- Later, generalize ignore patterns beyond only `hf_cache_test/`.

## What should be tracked

Recommended tracked:

- `examples/*.jsonl` small local fixtures.
- `config/dataset_sources.json`.
- `taxonomy/simple_domain_labels.json`.
- `data_samples/README.md`, after expanding it into a real policy.
- Canonical small benchmark outputs only if used for reproducible checks:
  - `classifier_benchmark_chunks.jsonl`;
  - `classifier_benchmark_labeled.jsonl`;
  - `classifier_benchmark_lexical_labeled.jsonl`;
  - `classifier_benchmark_run_stats.json`.
- Tiny real sample reports and stats.
- Tiny real sample raw chunks/labels only after privacy/license review:
  - FineWeb-Edu sample;
  - FineMath sample.

Recommended maybe tracked:

- `edge_case_*` outputs if they are used as fixtures in docs/tests.
- `local_real_like_*` generated outputs if the team wants fixed comparison evidence; otherwise regenerate from `examples/local_real_like_mini_sample.jsonl`.

Recommended not tracked in current scope:

- temporary sweep JSONL outputs;
- regenerated experiment outputs with different thresholds/settings;
- HF caches;
- model caches;
- `__pycache__`;
- future NLL scoring outputs;
- large raw web samples;
- OpenWebMath samples before optional_later approval.

## What can be deleted later

Do not delete during audit. Later cleanup candidates:

- untracked sweep files in `data_samples/classifier_benchmark_*maxdocs*` and `*target120*`;
- generated `local_real_like_*` outputs if reproducibility can come from the input fixture and report;
- early `fineweb_chunks_sample.jsonl` / `run_stats.json` if replaced by a clearer canonical smoke sample;
- `probability_profile_manifest.jsonl` if NLL artifacts are moved to future/archive;
- `nll_pilot_candidates.jsonl` if NLL scope is moved out of current stage2 owner area;
- local `hf_cache_test/` cache after confirming no needed metadata is only there.

## What should be ignored later

Do not edit `.gitignore` now. Suggested later additions or refinements:

```gitignore
# Stage2 local caches
external_samples/stage2_chunking_sample_clean/data_samples/hf_cache*/
external_samples/stage2_chunking_sample_clean/.cache/
external_samples/stage2_chunking_sample_clean/**/.cache/
external_samples/stage2_chunking_sample_clean/**/huggingface_cache/
external_samples/stage2_chunking_sample_clean/**/model_cache/

# Stage2 temporary/generated experiment outputs
external_samples/stage2_chunking_sample_clean/data_samples/tmp/
external_samples/stage2_chunking_sample_clean/data_samples/sweeps/
external_samples/stage2_chunking_sample_clean/data_samples/generated/
external_samples/stage2_chunking_sample_clean/data_samples/**/*_maxdocs*.jsonl
external_samples/stage2_chunking_sample_clean/data_samples/**/*_maxdocs*.json
external_samples/stage2_chunking_sample_clean/data_samples/**/*_target*.jsonl
external_samples/stage2_chunking_sample_clean/data_samples/**/*_target*.json

# Stage2 model/embedding outputs unless promoted intentionally
external_samples/stage2_chunking_sample_clean/data_samples/**/*_labeled_embedding*.jsonl
external_samples/stage2_chunking_sample_clean/data_samples/**/*_embedding_run_stats*.json

# Large or local-only real samples
external_samples/stage2_chunking_sample_clean/data_samples/real_samples/local_only/
external_samples/stage2_chunking_sample_clean/data_samples/real_samples/full/
```

Caution:

- Broad patterns like `*_target*.jsonl` can accidentally hide intentionally promoted fixtures. If used, add exceptions or put temporary outputs under a dedicated ignored directory instead.
- Prefer directory-based ignores (`sweeps/`, `tmp/`, `generated/`) over broad filename globs.

## Proposed data artifact policy

### Commit

Commit:

- small hand-authored fixtures in `examples/`;
- configs and taxonomy;
- docs and reports;
- small canonical benchmark fixture outputs if they are used for regression review;
- tiny real sample stats/reports;
- tiny real sample JSONL only after privacy/license review and only when needed for reproducibility.

### Do not commit

Do not commit:

- HF caches and model caches;
- full or large dataset dumps;
- local temporary sweeps;
- one-off generated outputs;
- model-dependent embedding outputs unless promoted as a named reproducibility artifact;
- NLL/logprob scoring outputs in the current stage2 scope;
- OpenWebMath samples before optional_later approval.

### Where to put real tiny samples

Recommended:

```text
data_samples/real_samples/
  fineweb_edu_sample10bt/
    chunks.jsonl
    labeled_rule_based.jsonl
    labeled_lexical.jsonl
    run_stats.json
    sample_card.md
  finemath/
    chunks.jsonl
    labeled_rule_based.jsonl
    labeled_lexical.jsonl
    run_stats.json
    sample_card.md
```

Current flat naming is acceptable for MVP, but per-source directories would make future MiniLM outputs easier:

```text
finemath/labeled_embedding_minilm_l6_v2.jsonl
finemath/embedding_run_stats_minilm_l6_v2.json
```

### Where to put temporary sweeps

Recommended:

```text
data_samples/sweeps/
  2026-05-14_chunking_parameter_sweep/
    classifier_benchmark_chunks_maxdocs20.jsonl
    classifier_benchmark_run_stats_maxdocs20.json
```

Default policy:

- ignored by git;
- summarized in docs;
- raw files deleted later unless explicitly promoted.

### Where to put generated benchmark outputs

Recommended:

```text
data_samples/generated_benchmark/
  canonical/
  tmp/
```

Alternative simpler MVP:

- keep current canonical files in `data_samples/`;
- move all non-canonical variants to `data_samples/sweeps/` or delete later.

## Repo size assessment

Current stage2 data size is acceptable:

- `examples`: about 83 KB.
- `data_samples` total including real samples and ignored cache: about 1.0 MB.
- ignored `hf_cache_test`: about 39 KB.
- ignored `scripts/__pycache__`: about 15 KB.

Risk:

- HF caches and model caches can grow from KB to GB.
- Real web samples can quickly become large and legally/privacy-sensitive.
- Embedding outputs can multiply files by model, threshold, source, and date.

Recommendation:

- Keep only tiny, named, reviewed samples in git.
- Store larger or experimental runs outside git or under ignored directories.
- Always record stats/report even when raw output is not committed.

## Naming convention assessment

Good:

- Real sample file names mostly follow `<source>_chunks`, `<source>_labeled_rule_based`, `<source>_labeled_lexical`, `<source>_run_stats`.
- FineWeb-Edu uses a config-aware prefix: `fineweb_edu_sample10bt`.
- FineMath uses a clean prefix: `finemath`.

Weak:

- `fineweb_chunks_sample.jsonl` sounds like real FineWeb but appears to be an early smoke sample.
- Root `data_samples/run_stats.json` is generic and can be overwritten.
- Sweep names are clear individually but live beside canonical outputs.
- NLL candidate file lives beside current real samples, making future NLL look active.
- No date/run-id/model-id naming for future embedding outputs.

Recommended later naming:

- canonical benchmark:
  - `classifier_benchmark_chunks.jsonl`;
  - `classifier_benchmark_labeled_rule_based.jsonl`;
  - `classifier_benchmark_labeled_lexical.jsonl`;
  - `classifier_benchmark_run_stats.json`.
- real samples:
  - `<source_id>_chunks.jsonl`;
  - `<source_id>_labeled_rule_based.jsonl`;
  - `<source_id>_labeled_lexical.jsonl`;
  - `<source_id>_labeled_embedding_<model_short>.jsonl`;
  - `<source_id>_run_stats.json`;
  - `<source_id>_embedding_run_stats_<model_short>.json`.
- sweeps:
  - `sweeps/<date>_<purpose>/<source>_<params>.jsonl`.

## Tiny samples worth keeping for reproducibility

High value:

- `examples/local_docs_classifier_benchmark.jsonl`: canonical synthetic benchmark input.
- `data_samples/classifier_benchmark_chunks.jsonl`: canonical generated chunks for benchmark.
- `data_samples/classifier_benchmark_labeled.jsonl`: rule-based output.
- `data_samples/classifier_benchmark_lexical_labeled.jsonl`: lexical output.
- `data_samples/real_samples/fineweb_edu_sample10bt_*`: current real educational tiny sample, if privacy/license acceptable.
- `data_samples/real_samples/finemath_*`: current MVP math tiny sample, if privacy/license acceptable.
- `taxonomy/simple_domain_labels.json`: required for lexical/embedding comparison.

Medium value:

- `examples/local_docs_edge_cases.jsonl` and `edge_case_*` generated outputs.
- `examples/local_real_like_mini_sample.jsonl`.
- `local_real_like_*` outputs.

Low/current-scope questionable:

- `probability_profile_manifest.jsonl`.
- `nll_pilot_candidates.jsonl`.
- old sweep outputs.
- early `fineweb_chunks_sample.jsonl` and generic `run_stats.json` once replaced by clearer fixtures.

## Specific git hygiene issues

### 1. Old sweep outputs are untracked and visible

Severity: Medium

They clutter status and are easy to accidentally commit. They should eventually be ignored, moved under ignored `sweeps/`, or deleted after summary extraction.

### 2. Cache is ignored, but ignore policy is too specific

Severity: Medium

Only `data_samples/hf_cache_test/` is explicitly ignored. Future HF/model cache paths may differ.

### 3. `data_samples/README.md` is outdated/too short

Severity: Medium

It says only small files for pipeline checks live there, but now the folder contains real HF tiny samples, labels, NLL candidate manifest, benchmark outputs, and old sweeps.

### 4. Raw real text is tracked without a visible review marker

Severity: Medium

FineWeb-Edu and FineMath tiny samples are useful, but raw web/HF text should have a clear privacy/license decision documented near the files.

### 5. NLL future artifacts are tracked in current data area

Severity: Low for git size, Medium for scope clarity

`probability_profile_manifest.jsonl` and `nll_pilot_candidates.jsonl` are small, but they blur current chunking/domain-labeling scope.

## Recommended cleanup order later

1. Expand `data_samples/README.md` into a real data policy.
2. Decide whether raw FineWeb-Edu/FineMath tiny samples are allowed to stay tracked.
3. Move/ignore/delete old sweep outputs.
4. Split `data_samples/` into `canonical/`, `real_samples/`, `sweeps/`, `tmp/`, or at least document those roles.
5. Generalize cache/model/temp ignore patterns.
6. Move future NLL artifacts out of current data path or mark them as downstream.
7. Add naming policy for MiniLM embedding outputs before running them.

## Current policy recommendation for next MiniLM milestone

Before adding MiniLM/Sentence-Transformers outputs:

- use FineWeb-Edu and FineMath tiny samples as the first real comparison inputs;
- keep OpenWebMath out of the current run;
- write embedding outputs to explicit new names, not generic paths;
- record model name, local path/cache status, threshold, text truncation, date, and source;
- do not commit embedding outputs automatically;
- promote only small, reviewed, reproducible embedding outputs after comparing with rule-based and lexical baselines.
