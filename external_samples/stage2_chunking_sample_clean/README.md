# Stage 2: sample chunking

Черновой прототип для нашей части проекта: подготовка текстов к дальнейшему анализу через logprob/NLL.

## Что делает скрипт

`scripts/sample_fineweb_chunks.py` берёт документы, чистит текст, режет его на chunks по абзацам и сохраняет результат в JSONL.

Сейчас в архиве уже есть локальный smoke test:

```bash
python scripts/sample_fineweb_chunks.py --max-docs 5
```

Результат:

```text
data_samples/fineweb_chunks_sample.jsonl
data_samples/run_stats.json
```

## Реальный запуск на FineWeb

Когда нужен настоящий sample FineWeb, запускаем через Hugging Face streaming:

```bash
pip install -r requirements.txt

python scripts/sample_fineweb_chunks.py \
  --use-hf-streaming \
  --dataset HuggingFaceFW/fineweb \
  --config sample-10BT \
  --max-docs 50 \
  --tokenizer-name Qwen/Qwen2.5-0.5B
```

Если `sample-10BT` не загрузится, можно попробовать конкретный dump:

```bash
python scripts/sample_fineweb_chunks.py \
  --use-hf-streaming \
  --dataset HuggingFaceFW/fineweb \
  --config CC-MAIN-2024-10 \
  --max-docs 50 \
  --tokenizer-name Qwen/Qwen2.5-0.5B
```

## Формат выходной записи

```json
{
  "chunk_id": "local_fineweb_like_sample_000000_000",
  "dataset": "local_fineweb_like_sample",
  "source_type": "web_general",
  "domain": null,
  "field": null,
  "subfield": null,
  "confidence": null,
  "token_count": 175,
  "text": "..."
}
```

Поля `domain`, `field`, `subfield`, `confidence` пока пустые: их заполняет следующий шаг — классификация chunks по доменам.

## Что важно

Это не обучение модели и не расчёт logits. Это первый технический слой:

```text
raw dataset -> logical chunks -> structured JSONL
```

Дальше этот JSONL можно передать в классификацию, а потом в расчёт `NLL_eff`, `NLL_full` и `delta`.

## Current local MVP pipeline

Generate chunk sample:

```bash
python scripts\sample_fineweb_chunks.py --local-input examples\local_docs_edge_cases.jsonl --max-docs 12 --out data_samples\edge_case_chunks_sample.jsonl --stats-out data_samples\edge_case_run_stats.json
```

Validate chunk sample:

```bash
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_sample.jsonl
```

Rule-based labeling:

```bash
python scripts\classify_chunks_rule_based.py --input data_samples\edge_case_chunks_sample.jsonl --output data_samples\edge_case_chunks_labeled.jsonl
```

Validate labeled sample:

```bash
python scripts\validate_chunks.py --input data_samples\edge_case_chunks_labeled.jsonl --require-labels
```

Inspect:

```bash
python scripts\inspect_chunks.py --input data_samples\edge_case_chunks_labeled.jsonl --limit 13
```

Optional embedding baseline:

```bash
python scripts\classify_chunks_embedding_baseline.py --help
python scripts\classify_chunks_embedding_baseline.py --input data_samples\edge_case_chunks_sample.jsonl --labels taxonomy\simple_domain_labels.json --output data_samples\edge_case_chunks_embedding_labeled.jsonl --dry-run
```

Notes:

- HF streaming is intentionally not part of this local smoke test.
- Edge cases are synthetic/local and small.
- Rule-based labels are only a transparent baseline.
- The embedding baseline is optional and dependency-dependent.
- Expected labels are only for local benchmark evaluation.
