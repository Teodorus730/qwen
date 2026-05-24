# Next steps checklist

## Done / current

- [x] chunker works on local JSONL
- [x] dataset/source_type are configurable
- [x] local rows can override metadata
- [x] rule-based baseline exists
- [x] validation/inspection tools exist

## Immediate next steps

- [ ] review labeled edge-case output manually
- [ ] decide whether to commit generated samples
- [ ] add tests for classifier rules
- [ ] add `contains_boilerplate` flag later, maybe
- [ ] add small real HF streaming sample later, not now

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
