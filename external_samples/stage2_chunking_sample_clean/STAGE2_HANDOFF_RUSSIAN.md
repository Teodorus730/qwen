# Stage2 handoff Russian

## 1. Что это за папка

`external_samples/stage2_chunking_sample_clean/` — это Stage2 слой подготовки корпуса и аннотаций для будущего NLL/logprob/perplexity profiling.

Текущее проектное направление: 0.8B Qwen-like модель обучается from scratch. Pretrained weights от Qwen не используются. Canonical tokenizer для текущего MVP — frozen Qwen-family infrastructure:

```text
Qwen/Qwen3.5-0.8B-Base
revision: dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
```

## 2. Короткий путь, который мы прошли

1. Старый setup `source_type + domain/field/subfield` оказался слишком смешанным. Он объединял topic, genre/function, surface format, quality/noise и provenance.
2. Stage2 перешел на `annotation_v2`: provenance, deterministic text stats, tokenization stats, surface features, quality/noise и weak topic labels.
3. `weak_topic_domain_v2.1` стал текущим прозрачным legacy weak baseline.
4. Held-out v2-test показал, что старый mixed `topic.domain` baseline деградирует и его нельзя дальше тюнить на v2-test.
5. Methodology review привел к более чистому разделению:
   - `semantic_topic_domain`
   - `genre_function`
6. BGE-M3 dev bake-off показал полезный semantic signal, но только на v1-dev.

## 3. Текущий статус

Готово для NLL pilot grouping:

- provenance/dataset metadata;
- deterministic text stats;
- Qwen3.5 tokenization stats;
- surface features: math/code/symbol/formula/API/table/url flags;
- quality/noise fields, with caveats.

Экспериментально:

- legacy `weak_topic_domain_v2.1`;
- cleaned `semantic_topic_domain_v1` pseudo-gold;
- BGE-M3 semantic topic embedding candidate;
- deterministic top-k reranking policy.

Пока нельзя заявлять финальное качество topic classifier.

## 4. Основные файлы и директории

- `scripts/`: feature, classifier, evaluation, summarization и bake-off utilities.
- `taxonomy/`: старые simple labels и cleaned semantic/genre taxonomy descriptions.
- `data_samples/`: dev/test pseudo-gold, predictions, summaries и generated sample artifacts.
- `examples/`: маленькие local examples и smoke artifacts.
- `docs/`: reports, methodology notes, decisions и runbooks.

Важные docs:

- `docs/annotation_schema_v2_domain_layer_readiness_report.md`
- `docs/annotation_schema_v2_axis_cleanup_plan.md`
- `docs/semantic_topic_genre_function_pseudo_gold_report.md`
- `docs/semantic_topic_embedding_bge_m3_dev_report.md`
- `docs/semantic_topic_embedding_bge_m3_error_and_v1_1_cards_report.md`

## 5. Основные команды

Предполагаемый working directory:

```powershell
cd external_samples\stage2_chunking_sample_clean
```

Для embedding scripts использовать repo-root embedding environment:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe
```

Запуск `weak_topic_domain_v2.1` на tokenized features:

```powershell
python scripts\classify_topic_domain_v2.py `
  --input data_samples\real_samples\real_hf_benchmark_v1_fineweb_features_v2_tokenized.jsonl `
  --output data_samples\real_samples\real_hf_benchmark_v1_fineweb_topic_domain_v2_1.jsonl `
  --version v2_1
```

Оценка topic-domain predictions:

```powershell
python scripts\evaluate_topic_domain_v2.py `
  --predictions data_samples\real_samples\real_hf_benchmark_v1_fineweb_topic_domain_v2_1.jsonl `
  --gold data_samples\real_hf_benchmark_v1_annotation_v2_pseudo_gold_patched.jsonl `
  --output-json data_samples\real_hf_benchmark_v1_topic_domain_v2_1_eval_patched_fineweb.json
```

Запуск BGE-M3 semantic-topic embedding classifier из local snapshot:

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

Оценка semantic-topic embedding predictions:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\evaluate_semantic_topic_predictions.py `
  --predictions data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards.jsonl `
  --gold data_samples\real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl `
  --output-json data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards_eval.json
```

Запуск deterministic top-k rerank по сохраненным BGE predictions:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\rerank_semantic_topic_embedding_topk.py `
  --input data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards.jsonl `
  --output data_samples\real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_reranked.jsonl `
  --margin-threshold 0.0
```

## 6. Unified handoff outputs

Unified chunk-level handoff files находятся здесь:

- `data_samples/handoff/stage2_v1_dev_annotations.jsonl`
- `data_samples/handoff/stage2_v2_test_annotations.jsonl`
- `data_samples/handoff/stage2_handoff_validation_summary.json`

Эти файлы join-ят существующие component outputs по `chunk_id`. Они включают deterministic fields из annotation_v2, tokenization stats, surface/quality fields, legacy `weak_topic_domain_v2.1` predictions, а также доступные review labels или embedding predictions.

Текущие validation counts:

- v1-dev handoff: 396 chunk-level records;
- v2-test handoff: 416 chunk-level records.

Важный статус labels:

- v1-dev включает cleaned `semantic_topic_domain_v1` и `genre_function_v1` pseudo-gold там, где они доступны, плюс BGE-M3 v1.1 dev predictions/reranked predictions там, где они доступны.
- v2-test включает legacy mixed-topic pseudo-gold и weak topic predictions там, где они доступны.
- v2-test не имеет cleaned semantic/genre gold. Cleaned held-out evaluation еще не сделан.
- Любой cleaned semantic prediction на v2-test нужно считать prediction-only, пока не принято решение по cleaned held-out labeling.

## 7. Текущие benchmark numbers

Краткая сводка benchmark ниже. Canonical final handoff status с packaging counts и quality metrics находится в `docs/stage2_final_handoff_status.md`.

Старый `source_type/domain/field/subfield`:

- rule-based full-label: about 0.10;
- MiniLM full-label: about 0.00;
- hybrid full-label: about 0.10.

`weak_topic_domain_v2.1`:

| Split | Coverage | Accuracy on answered | Strict |
| --- | ---: | ---: | ---: |
| v1 patched dev | 0.9083 | 0.8165 | 0.7417 |
| v2-test held-out | 0.7889 | 0.6197 | 0.4889 |

BGE-M3 на cleaned `semantic_topic_domain_v1`, только v1-dev:

| Setup | Accuracy | Top-3 contains gold |
| --- | ---: | ---: |
| v1 cards | 0.3814 | 0.7966 |
| v1.1 cards | 0.5678 | 0.8644 |
| v1.1 + rerank | 0.6356 | 0.8644 |

BGE-M3 numbers не являются held-out quality. Это dev-only результаты.

## 8. Известные проблемы

- FineWeb сложный и noisy.
- Старый `topic.domain` смешанный и должен считаться legacy weak metadata.
- BGE-M3 semantic-topic результаты dev-only и могут быть overfit к v1-dev error analysis.
- Cleaned held-out split пока нет.
- Local HF caches и virtual environments нельзя коммитить.

## 9. Что можно использовать для NLL pilot сейчас

Можно использовать уверенно:

- provenance/dataset;
- deterministic text stats;
- Qwen3.5 tokenization stats;
- surface flags;
- quality/noise fields with caveats.

Использовать осторожно:

- legacy `topic.domain`;
- cleaned semantic labels;
- BGE-M3 predictions.

Topic labels должны быть exploratory metadata с confidence, abstention и split caveats.

## 10. Следующие рекомендуемые шаги

1. Определить стратегию cleaned held-out:
   - relabel v2-test под cleaned axes с audit notes; или
   - создать свежий v3 held-out split.
2. Сравнить еще одну embedding model только после approval/download.
3. Собрать небольшой NLL grouping interface на robust axes.
4. Не использовать v2-test для tuning loops.

## 11. Contact / maintenance notes

- Использовать `.venv-embedding` для embedding scripts.
- Не тюнить на v2-test.
- Не коммитить local HF caches:
  - `.hf_dataset_cache/`
  - `.hf_embedding_cache/`
  - `.hf_tokenizer_cache/`
- Держать generated smoke/sweep artifacts отдельно от intended benchmark outputs.
