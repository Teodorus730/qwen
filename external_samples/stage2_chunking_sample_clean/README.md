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
