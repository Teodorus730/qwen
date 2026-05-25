# Next steps checklist

> Current scope note, 2026-05-26: this checklist is historical/useful context, not the primary handoff document. For current stage2 status use `current_status_ru.md`; for source decisions use `source_status_ru.md`; for MiniLM preparation use `minilm_readiness_plan_ru.md`. Current work is chunking + domain labeling. NLL/logprob/effective context is downstream/out of current scope.

## Done / current

- [x] chunker works on local JSONL
- [x] dataset/source_type are configurable
- [x] local rows can override metadata
- [x] rule-based baseline exists
- [x] validation/inspection tools exist
- [x] stage2 MVP commits completed locally
- [x] push intentionally postponed
- [x] synthetic benchmark hardened to 42 documents
- [x] embedding environment check done
- [x] lexical nearest-label baseline added
- [x] real sample plan added
- [x] dataset source registry added
- [x] dry-run real sample planners added
- [x] local real-like mini sample dry run added
- [x] HF dataset verification plan added
- [x] future HF streaming runbook added
- [x] HF row adapter design documented
- [x] first controlled FineWeb-Edu command attempted
- [x] first FineWeb-Edu tiny sample completed
- [x] first FineMath tiny sample completed
- [x] NLL scoring plan documented, not implemented

## Immediate next steps

- [x] confirm `datasets` imports in the working Python
- [x] add Russian stage2 handoff entry docs
- [x] document source status and data policy
- [x] define classifier contract for rule-based, lexical, and future embedding outputs
- [x] separate synthetic benchmark and real-sample validation modes
- [x] document MiniLM no-download readiness plan
- [ ] manually review FineWeb-Edu tiny sample chunks and label disagreements
- [x] use FineMath as the only current MVP math source
- [x] keep OpenWebMath as optional later comparison/backup only
- [ ] manually review FineMath tiny sample chunks and label disagreements
- [ ] verify HF dataset ids/configs/splits online later, only after approval
- [ ] record verified `text_field` values in `config/dataset_sources.json`
- [ ] review labeled edge-case output manually
- [ ] decide whether to commit generated samples
- [ ] add tests for classifier rules
- [ ] add `contains_boilerplate` flag later, maybe
- [ ] add small real HF streaming sample later, not now
- [ ] if a local embedding model exists, compare embedding vs rule-based
- [ ] if no local embedding model exists, decide whether to allow download later
- [ ] choose first real small dataset sample
- [ ] review `config/dataset_sources.json` before any HF run
- [ ] run `scripts\inspect_dataset_sources.py` before any HF run
- [ ] run `scripts\plan_real_source_pipeline.py` before any HF run
- [ ] follow `docs\real_sample_readiness_checklist.md`
- [ ] follow `docs\future_hf_streaming_runbook.md` before first HF run
- [ ] use `docs\real_source_labeling_plan.md` for manual review
- [ ] only after current stage2 handoff/MiniLM comparison is complete, revisit observed-token NLL scoring as downstream work

## Later

- [ ] sentence-transformers embedding baseline
- [ ] OpenAlex-like taxonomy
- [ ] MiniLM/e5 nearest-label classification
- [ ] any additional real FineWeb/FineWeb-Edu/FineMath small streaming samples after approval
- [ ] observed-token logprob/NLL pipeline
- [ ] effective vs full window comparison

## Do not do yet

- [ ] full FineWeb processing
- [ ] big BERT/zero-shot classifier
- [ ] training Qwen
- [ ] storing full softmax distributions
