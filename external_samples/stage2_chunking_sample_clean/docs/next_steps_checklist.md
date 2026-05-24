# Next steps checklist

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
- [x] NLL scoring plan documented, not implemented

## Immediate next steps

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
- [ ] only after real labels look sane, start observed-token NLL scoring design/implementation

## Later

- [ ] sentence-transformers embedding baseline
- [ ] OpenAlex-like taxonomy
- [ ] MiniLM/e5 nearest-label classification
- [ ] real FineWeb/FineWeb-Edu/FineMath small streaming samples
- [ ] observed-token logprob/NLL pipeline
- [ ] effective vs full window comparison

## Do not do yet

- [ ] full FineWeb processing
- [ ] big BERT/zero-shot classifier
- [ ] training Qwen
- [ ] storing full softmax distributions
