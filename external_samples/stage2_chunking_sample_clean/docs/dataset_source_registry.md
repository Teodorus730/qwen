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
- `finemath`: current MVP HF math-oriented sample; dataset id/config need verification before a first run.
- `openwebmath`: optional later comparison/backup only, not an active current planned source.
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
2. FineMath tiny sample as the only current MVP math source.
3. FineWeb general tiny sample.
4. Cosmopedia/SmolLM synthetic educational sample.

OpenWebMath may be reconsidered later as a comparison or backup source, but it is not part of the current MVP source sequence.

This is a plan, not an executed run.

## Current HF run note

The first controlled FineWeb-Edu tiny sample completed after using a writable local Hugging Face cache. It streamed 20 documents and wrote 44 chunks under `data_samples\real_samples`. See `docs/fineweb_edu_tiny_sample_report.md`.

## Related planning docs

- `docs/hf_dataset_verification_plan.md`
- `docs/hf_row_adapter_design.md`
- `docs/future_hf_streaming_runbook.md`
- `docs/real_source_labeling_plan.md`
