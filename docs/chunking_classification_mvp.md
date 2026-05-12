# MVP: разбиение датасетов на chunks и доменная классификация

## Цель

Наша задача — подготовить данные для дальнейшего вероятностного анализа эталонной модели  то естьпревратить сырые текстовые датасеты в воспроизводимый набор логически завершённых фрагментов текста с базовой доменной разметкой.

Общий pipeline:

```
raw datasets → logical chunks → source/domain labels → JSONL/Parquet
```
Полученные chunks дальше можно будет использовать для анализа:

вероятностей следующих токенов;
perplexity / NLL;
сравнения effective window и adaptive/full window;
оценки того, какие домены требуют более длинного контекста.
Что такое chunk

Chunk — это логически завершённый кусок текста.

Возможные варианты chunk:

целый документ;
секция документа;
глава;
абзац или группа абзацев;
задача с решением;
учебный фрагмент;
web-страница после очистки.

Для MVP важно не идеально восстановить структуру документа, а получить достаточно осмысленные и воспроизводимые куски текста.

Параметры MVP

Предлагаемые параметры для первого эксперимента:

min_chunk_tokens = 256
target_chunk_tokens = 1024–2048
max_chunk_tokens = 4096

Логика:

слишком короткие chunks могут быть малоинформативны;
слишком длинные chunks дороже анализировать;
диапазон 1024–2048 токенов выглядит как разумный старт для смысловых фрагментов;
max 4096 токенов ограничивает стоимость дальнейшего inference.
Стратегия chunking

Для MVP используем простой rule-based подход.

1. Document-as-chunk baseline

Если документ уже имеет нормальную длину, считаем его одним chunk.

Это самый простой и воспроизводимый baseline.

2. Разбиение длинных документов

Если документ слишком длинный, режем его по естественным границам:

markdown-заголовки;
HTML-derived headings, если они есть;
пустые строки;
абзацы;
секции;
блоки задач/решений;
code/math blocks, если они явно выделены.
3. Обработка коротких chunks

Если chunk слишком короткий:

либо склеиваем его с соседним фрагментом;
либо отбрасываем, если он выглядит как мусор, меню, footer, boilerplate.
4. Fixed-size chunks

Fixed-size нарезку по токенам можно использовать только как baseline для сравнения.

Основной метод должен сохранять логическую завершённость, потому что это явно требуется в постановке задачи.

MVP datasets

Для первого этапа не обрабатываем все датасеты целиком.

Берём маленькие streaming samples:

FineWeb;
FineWeb-Edu;
FineMath или OpenWebMath;
Cosmopedia / SmolLM-Corpus — optional.

Ориентир для первого теста:

1000 документов на датасет

Для локальной проверки скриптов можно начинать ещё меньше:

50–100 документов
Базовая схема разметки

Для каждого chunk сохраняем два уровня информации.

Layer A: source_type

Грубый тип источника:

web_general
educational
math
code
science
synthetic
wiki_reference
forum_qa
commercial_product
legal_government
unknown
Layer B: domain / field / subfield

Более детальная доменная разметка.

Кандидат для английского MVP — OpenAlex-like taxonomy:

domain → field → subfield

На первом этапе допускается грубая разметка по датасету:

FineWeb → web_general
FineWeb-Edu → educational
FineMath / OpenWebMath → math
Cosmopedia → educational + synthetic

Позже эту разметку можно улучшить через embeddings, zero-shot classification или supervised classifier.

Итоговая JSONL schema

Каждая запись должна иметь примерно такой формат:

{
  "chunk_id": "fineweb_000001_000",
  "dataset": "FineWeb",
  "source_type": "web_general",
  "domain": null,
  "field": null,
  "subfield": null,
  "confidence": null,
  "token_count": 1240,
  "text": "..."
}

Описание полей:

chunk_id — уникальный идентификатор chunk;
dataset — источник данных;
source_type — грубый тип источника;
domain — домен верхнего уровня;
field — область внутри домена;
subfield — под область;
confidence — уверенность классификатора, если используется;
token_count — длина chunk в токенах;
text — текст chunk.
Что считаем успехом MVP

MVP считается успешным, если:

Есть воспроизводимый способ взять маленький sample из датасета.
Есть rule-based chunking.
Есть JSONL-файл с chunks.
Для каждого chunk сохранены:
dataset;
source_type;
token_count;
text.
Есть базовая статистика:
сколько документов обработано;
сколько chunks получилось;
min / mean / max token length;
распределение chunks по датасетам;
распределение chunks по source_type.
Есть ручная проверка небольшой выборки chunks.
Pipeline можно расширить на другие датасеты.
Что не входит в MVP

На этом этапе не делаем:

обучение модели;
подсчёт logits;
подсчёт perplexity;
full softmax по словарю;
обработку всего FineWeb;
сложную semantic segmentation;
LLM-based segmentation;
supervised classifier.

Эти части относятся к следующим этапам проекта.

Ближайшие шаги
Подготовить пробный скрипт для FineWeb streaming sample.
Нарезать 50–100 документов на chunks.
Сохранить результат в JSONL.
Посмотреть руками 20–50 chunks.
После проверки повторить на FineWeb-Edu и FineMath/OpenWebMath.
Затем перейти к файлу с таксономиями и классификаторами.
