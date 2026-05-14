# Как проверить, что всё работает

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

Для локального smoke test зависимости почти не нужны, но для FineWeb streaming нужны `datasets` и `transformers`.

2. Запустить локальный тест:

```bash
python scripts/sample_fineweb_chunks.py --max-docs 5
```

3. Проверить результат:

```bash
cat data_samples/run_stats.json
head -n 1 data_samples/fineweb_chunks_sample.jsonl
```

4. Признак, что всё ок:

- `chunks_written` больше 0;
- появился файл `data_samples/fineweb_chunks_sample.jsonl`;
- в каждой строке есть `chunk_id`, `token_count`, `text`;
- поля `domain/field/subfield` пока пустые — так и должно быть.
