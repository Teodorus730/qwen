# Stage2 handoff

## 1. What this folder is

`external_samples/stage2_chunking_sample_clean/` is the Stage2 corpus preparation and annotation layer for future NLL/logprob/perplexity profiling.

The project direction is a 0.8B Qwen-like model trained from scratch. Qwen pretrained weights are not used. The canonical tokenizer for the current MVP is frozen Qwen-family infrastructure:

```text
Qwen/Qwen3.5-0.8B-Base
revision: dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
```

## 2. Short path we went through

1. The old `source_type + domain/field/subfield` setup was too mixed. It blended topic, genre/function, surface format, quality/noise, and provenance.
2. Stage2 pivoted to `annotation_v2`: provenance, deterministic text stats, tokenization stats, surface features, quality/noise, and weak topic labels.
3. `weak_topic_domain_v2.1` became the current transparent legacy weak baseline.
4. Held-out v2-test showed that the old mixed `topic.domain` baseline degrades and should not be tuned further on v2-test.
5. Methodology review led to a cleaner split:
   - `semantic_topic_domain`
   - `genre_function`
6. A BGE-M3 dev bake-off showed useful semantic signal, but only on v1-dev.

## 3. Current status

Ready for NLL pilot grouping:

- provenance/dataset metadata;
- deterministic text stats;
- Qwen3.5 tokenization stats;
- surface features: math/code/symbol/formula/API/table/url flags;
- quality/noise fields, with caveats.

Experimental:

- legacy `weak_topic_domain_v2.1`;
- cleaned `semantic_topic_domain_v1` pseudo-gold;
- BGE-M3 semantic topic embedding candidate;
- deterministic top-k reranking policy.

Do not claim final topic-classifier quality yet.

## 4. Main files/directories

- `scripts/`: feature, classifier, evaluation, summarization, and bake-off utilities.
- `taxonomy/`: old simple labels and cleaned semantic/genre taxonomy descriptions.
- `data_samples/`: dev/test pseudo-gold, predictions, summaries, and generated sample artifacts.
- `examples/`: small local examples and smoke artifacts.
- `docs/`: reports, methodology notes, decisions, and runbooks.

Important docs:

- `docs/annotation_schema_v2_domain_layer_readiness_report.md`
- `docs/annotation_schema_v2_axis_cleanup_plan.md`
- `docs/semantic_topic_genre_function_pseudo_gold_report.md`
- `docs/semantic_topic_embedding_bge_m3_dev_report.md`
- `docs/semantic_topic_embedding_bge_m3_error_and_v1_1_cards_report.md`

## 5. Main commands

Assume working directory:

```powershell
cd external_samples\stage2_chunking_sample_clean
```

Use the repo-root embedding environment for embedding scripts:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe
```

Run `weak_topic_domain_v2.1` on tokenized features:

```powershell
python scripts\classify_topic_domain_v2.py `
  --input data_samples\real_samples\real_hf_benchmark_v1_fineweb_features_v2_tokenized.jsonl `
  --output data_samples\real_samples\real_hf_benchmark_v1_fineweb_topic_domain_v2_1.jsonl `
  --version v2_1
```

Evaluate topic-domain predictions:

```powershell
python scripts\evaluate_topic_domain_v2.py `
  --predictions data_samples\real_samples\real_hf_benchmark_v1_fineweb_topic_domain_v2_1.jsonl `
  --gold data_samples\real_hf_benchmark_v1_annotation_v2_pseudo_gold_patched.jsonl `
  --output-json data_samples\real_hf_benchmark_v1_topic_domain_v2_1_eval_patched_fineweb.json
```

Run BGE-M3 semantic-topic embedding classifier from the local snapshot:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\classify_semantic_topic_embedding.py `
  --input data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --domain-descriptions taxonomy\semantic_topic_domain_v1_1_descriptions.json `
  --output data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards.jsonl `
  --model "C:\Users\pervo\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181" `
  --threshold 0.0 `
  --margin-threshold 0.0 `
  --top-k 3 `
  --text-field text
```

Evaluate semantic-topic embedding predictions:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\evaluate_semantic_topic_predictions.py `
  --predictions data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards.jsonl `
  --gold data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --output-json data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards_eval.json
```

Run deterministic top-k rerank on saved BGE predictions:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\rerank_semantic_topic_embedding_topk.py `
  --input data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards.jsonl `
  --output data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_reranked.jsonl `
  --margin-threshold 0.0
```

## 6. Current benchmark numbers

Old `source_type/domain/field/subfield`:

- rule-based full-label: about 0.10;
- MiniLM full-label: about 0.00;
- hybrid full-label: about 0.10.

`weak_topic_domain_v2.1`:

| Split | Coverage | Accuracy on answered | Strict |
| --- | ---: | ---: | ---: |
| v1 patched dev | 0.9083 | 0.8165 | 0.7417 |
| v2-test held-out | 0.7889 | 0.6197 | 0.4889 |

BGE-M3 on cleaned `semantic_topic_domain_v1`, v1-dev only:

| Setup | Accuracy | Top-3 contains gold |
| --- | ---: | ---: |
| v1 cards | 0.3814 | 0.7966 |
| v1.1 cards | 0.5678 | 0.8644 |
| v1.1 + rerank | 0.6356 | 0.8644 |

The BGE-M3 numbers are not held-out quality. They are dev-only.

## 7. Known problems

- FineWeb is hard and noisy.
- Old `topic.domain` is mixed and should be treated as legacy weak metadata.
- BGE-M3 semantic-topic results are dev-only and may be overfit to v1-dev error analysis.
- There is no cleaned held-out split yet.
- Local HF caches and virtual environments should not be committed.

## 8. What can be used for NLL pilot now

Use confidently:

- provenance/dataset;
- deterministic text stats;
- Qwen3.5 tokenization stats;
- surface flags;
- quality/noise fields with caveats.

Use cautiously:

- legacy `topic.domain`;
- cleaned semantic labels;
- BGE-M3 predictions.

Topic labels should be exploratory metadata with confidence, abstention, and split caveats.

## 9. Next suggested steps

1. Decide cleaned held-out strategy:
   - relabel v2-test under cleaned axes with audit notes; or
   - create a fresh v3 held-out split.
2. Compare another embedding model only after approval/download.
3. Build a small NLL grouping interface using robust axes first.
4. Keep v2-test out of tuning loops.

## 10. Contact / maintenance notes

- Use `.venv-embedding` for embedding scripts.
- Do not tune on v2-test.
- Do not commit local HF caches:
  - `.hf_dataset_cache/`
  - `.hf_embedding_cache/`
  - `.hf_tokenizer_cache/`
- Keep generated smoke/sweep artifacts separate from intended benchmark outputs.
