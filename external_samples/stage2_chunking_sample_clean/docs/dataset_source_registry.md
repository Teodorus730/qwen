# Dataset source registry

## Purpose

`config/dataset_sources.json` lists planned local and future HF sources for tiny stage2 samples. The registry is planning metadata only. It does not download data or start HF streaming.

## Source layers

- HF streaming is a loading mechanism, not dataset identity.
- `dataset_label` is output metadata written into chunk records.
- `source_type` is the broad source or format, such as `web_general`, `educational`, `math`, or `unknown`.
- `domain/field/subfield` are classifier labels produced later by rule-based, lexical, or embedding methods.
- `text_field` and `id_field` are planned row-adapter metadata. They must be verified for HF sources before streaming.
- `planned_output_prefix` controls the intended file prefix under `data_samples/real_samples`.

## Current planned sources

- `fineweb_general`: planned HF web-general sample; dataset/config need verification.
- `fineweb_edu`: planned HF educational web sample; dataset id/config need verification.
- `finemath`: planned HF math-oriented sample; dataset id/config need verification.
- `openwebmath`: planned HF math-oriented sample; dataset id/config need verification.
- `cosmopedia_or_smollm_corpus`: planned synthetic educational corpus candidate; dataset id/config need verification.
- `local_edge_cases`: existing local smoke sample.
- `local_classifier_benchmark`: hardened local synthetic classifier benchmark.

## Safety rules

- No HF/network without explicit approval.
- Start with 20-100 docs per dataset.
- Never process full FineWeb locally.
- Commit only small reproducible samples if needed.
- Record dataset id, config, split, and max_docs for every real sample run.

## Verification needed before HF run

The following registry entries have `needs_verification: true`:

- `fineweb_general`
- `fineweb_edu`
- `finemath`
- `openwebmath`
- `cosmopedia_or_smollm_corpus`

For all of them, verify dataset id, config, split, licensing notes, and expected text field before running.

## Suggested first real sample

Recommended order, still only after explicit HF/network approval:

1. FineWeb-Edu or local real-like docs.
2. FineMath/OpenWebMath tiny sample.
3. FineWeb general tiny sample.
4. Cosmopedia/SmolLM synthetic educational sample.

This is a plan, not an executed run.

## Current HF run blocker

The first controlled FineWeb-Edu command was attempted with `max_docs=20`, but the active Python environment is missing the `datasets` package. The run failed before streaming any rows. See `docs/fineweb_edu_tiny_sample_report.md`.

## Related planning docs

- `docs/hf_dataset_verification_plan.md`
- `docs/hf_row_adapter_design.md`
- `docs/future_hf_streaming_runbook.md`
- `docs/real_source_labeling_plan.md`
