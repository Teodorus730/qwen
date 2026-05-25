# Stage2 data policy

Дата: 2026-05-26

This policy covers files under `external_samples/stage2_chunking_sample_clean/`.

## Commit by default

These are useful for reproducibility and team handoff:

- small local fixtures in `examples/`;
- `config/dataset_sources.json`;
- `taxonomy/simple_domain_labels.json`;
- canonical small benchmark outputs in `data_samples/`;
- tiny real sample reports and run stats;
- docs and audit reports.

## Commit only after review

These may be useful, but should stay small and inspected:

- FineWeb-Edu tiny chunks and labeled outputs;
- FineMath tiny chunks and labeled outputs;
- future tiny real samples;
- model-dependent embedding outputs promoted as canonical examples.

Review before committing:

- source/license notes;
- whether raw text contains sensitive or unwanted content;
- file size;
- whether the sample is reproducible enough to help the team.

## Do not commit

- HF caches;
- model caches;
- dependency folders;
- `__pycache__/` and `*.pyc`;
- large raw dataset dumps;
- large generated sweeps;
- temporary benchmark variants;
- accidental full-dataset outputs;
- OpenWebMath samples before optional_later approval;
- NLL/logprob outputs for the current stage2 scope;
- MiniLM embedding outputs unless explicitly promoted.

## Old sweep outputs

Old untracked sweep outputs currently create `git status` noise. Do not delete them in this cleanup step.

Later options:

- summarize useful results in docs and delete local scratch files;
- move temporary sweeps to an ignored `data_samples/sweeps/` folder;
- add explicit `.gitignore` patterns after team approval;
- promote only one canonical benchmark output if needed.

## Tiny real samples

Preferred location:

```text
data_samples/real_samples/
```

Naming pattern:

```text
<source_prefix>_chunks.jsonl
<source_prefix>_labeled_rule_based.jsonl
<source_prefix>_labeled_lexical.jsonl
<source_prefix>_labeled_embedding_<model_short>.jsonl
<source_prefix>_run_stats.json
```

For the current MVP:

- `fineweb_edu_sample10bt_*`
- `finemath_*`

## Future embedding outputs

Embedding outputs should include enough naming or sidecar metadata to recover:

- source prefix;
- classifier method;
- model short name;
- threshold or low-confidence policy;
- run date or stats file;
- taxonomy version/path.

Do not overwrite rule-based or lexical outputs.
