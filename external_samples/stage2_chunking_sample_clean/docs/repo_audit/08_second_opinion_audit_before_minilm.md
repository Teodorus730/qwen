# Second-opinion audit before MiniLM

## Executive summary
Проект stage2 значительно улучшился после фазы cleanup: появились русскоязычные документы, определены статусы источников, data policy и спецификации выходов валидаторов. Концептуально этап полностью готов к прогону MiniLM. Однако кодовая база `classify_chunks_embedding_baseline.py` на данный момент расходится с обновлённым `classifier_contract_ru.md` и планом готовности по части логики, производительности (нет батчинга) и формату метаданных. Переход к MiniLM возможен сразу после исправления этого скрипта.

## What improved after cleanup
- Появились русскоязычные entry points (`current_status_ru.md`, `README.md` и др.), описывающие проект без погружения в устаревшие планы.
- Зафиксирован статус источников: FineMath признан текущим MVP, а OpenWebMath отложен (optional_later).
- Чётко определен `classifier contract`, отделяющий атрибуты источника (`source_type`) от предсказанных лейблов (`domain/field/subfield`).
- Сформулированы правила для валидации: synthetic (accuracy) vs real samples (agreement).
- Прописан `no-download workflow` для безопасного запуска моделей без случайных скачиваний.

## Known issues that are intentionally accepted for now
- Stage2 по-прежнему изолирован в `external_samples/stage2_chunking_sample_clean`.
- Корневой репозиторий не реорганизован.
- Разбросанные по папке старые sweep outputs и англоязычные отчеты временно игнорируются, не мешая текущей работе.
- Полноценный рефакторинг в Python-пакет и создание единой CLI-оболочки отложены.

## Known issues still worth fixing before MiniLM
- В `classify_chunks_embedding_baseline.py` полностью отсутствует батчинг (вызов `model.encode([text])` в цикле для каждой записи отдельно), что сделает инференс невыносимо долгим даже на малых real samples.
- Нет управления `batch_size` и `device` через аргументы командной строки.
- Скрипт не выводит/не собирает дополнительную информацию по top-K предсказаниям, что снижает полезность disagreement review.

## New issues found
- Несоответствие контракту: скрипт использует `label_method = "embedding_nearest_label_v1"`, в то время как контракт требует `embedding_nearest_label_minilm`.
- Нарушение контракта метрик: код зачем-то перезаписывает `source_type` (`"source_type": out.get("source_type") or "unknown"`). Контракт прямо запрещает скрытую перезапись metadata источника.
- Отсутствуют полезные метаданные профиля (model_name, threshold) в итоговых JSONL, обещанные в контракте.
- Использование ручного расчета `cosine_similarity` внутри скалярного Python-цикла вместо матричных операций или встроенного `util.cos_sim` из SentenceTransformers.

## MiniLM readiness verdict
- **Documentation readiness**: High.
- **Data readiness**: High.
- **Code readiness**: Low-Medium (требуется переписывание пайплайна эмбеддингов в baseline-скрипте под контракт).
- **Safety/no-download readiness**: High (скрипт корректно использует `local_files_only=True`).
- **Comparison readiness**: High.

## Blockers before MiniLM
1. Обновить `classify_chunks_embedding_baseline.py`: исправить константы `label_method`, прекратить мутацию `source_type` и добавить сохранение метаданных модели в итоговый record.
2. Переписать цикл `classify_records` на батчевую обработку `model.encode(texts, batch_size=...)` для корректной скорости работы.
3. Заменить ручной `cosine_similarity` на векторизованное вычисление (numpy/torch/SentenceTransformers util).

## Non-blockers / postpone
- Реализация top-k labels (можно оставить на потом; хватит nearest-label с low-confidence флагом).
- Перемещение папки stage2 в root репозитория.
- Написание абстрактного JSONL/taxonomy процессора для всех видов классификаторов (дублирование кода на данном этапе дешевле, чем сложный рефакторинг).
- Автоматическое выкачивание моделей. 

## Recommended next actions
1. Исправить код `classify_chunks_embedding_baseline.py` согласно блокерам выше.
2. Скачать (вручную или отдельным явным скриптом) модель `all-MiniLM-L6-v2` локально для тестирования, если её ещё нет в кэше.
3. Провести пробный запуск на `classifier_benchmark_chunks.jsonl` и валидацию выхода.
4. Запустить MiniLM на FineWeb-Edu и FineMath tiny samples, сформировать disagreement отчет согласованности (rule-based vs lexical vs embedding).

## Risk of over-cleanup
Риск увязнуть в излишней чистке отсутствует, если мы сфокусируемся **только на обновлении одного файла** (`classify_chunks_embedding_baseline.py`) и сразу перейдём к запуску эмбеддингов. Главное — не пытаться сейчас рефакторить старые бейзлайны, улучшать утилиты или трогать общую структуру.
