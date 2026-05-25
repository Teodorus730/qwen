# HF dataset verification plan

## Purpose

We need to verify dataset IDs, configs, splits, and row schemas before any HF streaming run. This document is planning-only and does not imply that network or HF access is approved.

## Current candidate sources

- FineWeb general
  - registry source_id: `fineweb_general`
  - current hf_dataset: `HuggingFaceFW/fineweb`
  - current hf_config: `sample-10BT`
  - current split: `train`
  - expected source_type: `web_general`
  - verification status: needs verification
  - check later: dataset id, config, split, row schema, text field, license, and whether the sample config is still appropriate.

- FineWeb-Edu
  - registry source_id: `fineweb_edu`
  - current hf_dataset: null
  - current hf_config: null
  - current split: `train`
  - expected source_type: `educational`
  - verification status: needs verification
  - check later: exact dataset id/config, split, row schema, text field, license, and whether data is raw web text or cleaned educational text.

- FineMath
  - registry source_id: `finemath`
  - current hf_dataset: null
  - current hf_config: null
  - current split: `train`
  - expected source_type: `math`
  - verification status: needs verification
  - check later: exact dataset id/config, split, row schema, text field, formula/LaTeX representation, and license.

- OpenWebMath
  - registry source_id: `openwebmath`
  - current hf_dataset: null
  - current hf_config: null
  - current split: `train`
  - expected source_type: `math`
  - verification status: needs verification
  - check later: exact dataset id/config, split, row schema, text field, formula/HTML/markdown format, and license.

- Cosmopedia / SmolLM-Corpus
  - registry source_id: `cosmopedia_or_smollm_corpus`
  - current hf_dataset: null
  - current hf_config: null
  - current split: `train`
  - expected source_type: `educational_synthetic`
  - verification status: needs verification
  - check later: choose one concrete dataset, exact config/split, row schema, text field, generated-text style, and license.

## What must be verified online later

- Exact HF dataset id.
- Config/subset name.
- Split name.
- Row schema.
- Text field name.
- Whether `streaming=True` works.
- Whether license/terms allow committing tiny samples.
- Approximate row/document format.
- Whether the sample contains text, HTML, markdown, math, code, or nested fields.

## Do not run yet

No network, HF streaming, dataset download, or schema probing should be run until the user explicitly approves it.

## First safe run recommendation

For the first future HF run, use `max_docs=20`, one source only, output under `data_samples/real_samples`, then validate and inspect manually before running any other source.
