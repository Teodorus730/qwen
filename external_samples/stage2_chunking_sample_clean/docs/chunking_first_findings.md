# Chunking first findings

## Что было проверено

Проверили два локальных сценария для первичного chunking-прототипа:

1. Local smoke sample из `examples/local_docs.jsonl`.
2. Edge-case sample из 10 искусственных документов в `examples/local_docs_edge_cases.jsonl`.

Оба сценария проверяют только технический слой:

```text
local JSONL -> paragraph-based chunks -> structured JSONL
```

Это пока не обучение модели, не расчет logits и не NLL/logprob analysis.

## Результаты local smoke sample

Local smoke sample был нужен, чтобы быстро проверить базовую механику без скачивания FineWeb.

Результат:

- документов: 5;
- chunks: 36;
- mean `token_count`: около 103;
- формат JSONL создается корректно;
- поля `chunk_id`, `dataset`, `source_type`, `token_count`, `text` присутствуют.

Главное ограничение этого sample: он слишком чистый и шаблонный. Тексты искусственные, однотипные, почти без web-noise, boilerplate, меню, футеров, рекламных блоков и других проблем, которые будут в настоящем FineWeb/FineWeb-Edu.

## Результаты edge-case sample

Edge-case sample содержит 10 искусственных документов с разными типами входа: обычная образовательная статья, длинный текст с секциями, очень короткий документ, cookie/privacy/footer boilerplate, navigation/menu-like noise, math/LaTeX-like text, code/documentation-like text, mixed English/Russian, forum/Q&A style и commercial/product-like page.

Результат запуска:

- `docs_seen`: 10;
- `docs_with_text`: 10;
- `chunks_written`: 9;
- `min_chunk_tokens`: 85;
- `mean_chunk_tokens`: 116.44;
- `max_chunk_tokens`: 159.

Один very short документ был отфильтрован, потому что он оказался ниже `min_chunk_tokens`. Это подтверждает, что минимальный token filter работает.

## Главный вывод

Chunker технически работает:

- локальный JSONL input читается;
- paragraph-based chunking выполняется;
- JSONL schema для output работает;
- отдельные `--out` и `--stats-out` позволяют не перезаписывать старый sample;
- фильтр коротких chunks работает.

Но cleaning/boilerplate removal пока нет. Cookie/privacy/footer blocks, navigation/menu-like text и commercial/promotional noise проходят как обычный текст и попадают в chunks вместе с полезным содержанием.

## Что нужно улучшить перед реальным FineWeb/FineWeb-Edu

Перед запуском на настоящем FineWeb/FineWeb-Edu стоит сделать следующие улучшения:

1. Добавить boilerplate detection для cookie banners, privacy notices, footers и repeated legal text.
2. Добавить простую фильтрацию navigation/menu-like chunks.
3. Увеличить target chunk size для реальных данных, потому что текущий `target_chunk_tokens=180` слишком мал для полноценного анализа контекста.
4. Проверить поведение на настоящем Hugging Face streaming sample без сохранения больших датасетов в репозиторий.
5. Добавить ручную проверку 20-50 chunks после первого реального запуска: смотреть завершенность, шум, повторы, обрывы и распределение `token_count`.
