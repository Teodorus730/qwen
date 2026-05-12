# Таксономии и классификаторы для доменной разметки chunks

## Зачем это нужно

После разбиения датасетов на логически завершённые chunks каждому chunk нужно присвоить тематическую метку.

Это нужно для дальнейшего анализа вероятностных распределений эталонной модели по доменам:

```text
chunk → source_type / domain / field / subfield → logprob / NLL / perplexity analysis
````

Артем формулирует это как “присвоить каждому кусочку код до 3 уровней”.

Для MVP нам нужен не идеальный научный классификатор, а воспроизводимая схема доменной разметки, которую можно постепенно улучшать.

## Двухслойная схема разметки

Для MVP предлагаем разделить разметку на два слоя.

### Layer A: source_type

Грубый тип источника или текста:

```text
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
```

Этот слой можно присваивать простыми правилами или по источнику датасета.

Примеры:

```text
FineWeb → web_general
FineWeb-Edu → educational
FineMath / OpenWebMath → math
Cosmopedia / SmolLM-Corpus → educational + synthetic
```

### Layer B: domain / field / subfield

Более детальная тематическая классификация.

Пример структуры:

```text
domain → field → subfield
```

Этот слой нужен для анализа по областям знаний.

Например:

```text
Physical Sciences → Mathematics → Applied Mathematics
Life Sciences → Medicine → Epidemiology
Social Sciences → Education → Curriculum and Pedagogy
Computer Science → Artificial Intelligence → Natural Language Processing
```

## Основной кандидат для английского MVP: OpenAlex Topics

OpenAlex Topics подходит как основной кандидат для английской доменной разметки, потому что имеет иерархическую структуру:

```text
domain → field → subfield → topic
```

Для нашего MVP можно использовать первые три уровня:

```text
domain / field / subfield
```

Плюсы:

* уже есть готовая иерархия;
* хорошо подходит для научных и образовательных текстов;
* близко к требованию “код до 3 уровней”;
* можно использовать как основу для будущего supervised classifier.

Минусы:

* OpenAlex ориентирован на scholarly works, а не на весь web;
* обычные web-страницы могут плохо ложиться в академическую классификацию;
* нужен confidence threshold;
* нужен класс `unknown`, чтобы не заставлять классификатор насильно выбирать академический домен для мусорного или бытового текста.

## Дополнительные таксономии

### arXiv taxonomy

Подходит для STEM-текстов:

* computer science;
* mathematics;
* physics;
* statistics;
* quantitative biology;
* quantitative finance;
* economics.

Плюсы:

* простые коды вроде `cs.CL`, `cs.AI`, `math.PR`, `stat.ML`;
* полезно для math / code / scientific domains.

Минусы:

* плохо покрывает общий web;
* не универсальная классификация.

### ACM CCS

Таксономия для computer science.

Плюсы:

* хорошая иерархия для CS;
* полезна, если в корпусе будет много code / ML / NLP / systems / security.

Минусы:

* покрывает только computer science;
* не подходит как общая таксономия для всего корпуса.

### MeSH / PubMed

Таксономии для медицины, биологии и health-related текстов.

Плюсы:

* хороши для biomedical domain.

Минусы:

* узкая область;
* не нужны в MVP, если biomedical тексты не являются отдельным фокусом.

### MSC

Mathematics Subject Classification.

Плюсы:

* подходит для математических текстов.

Минусы:

* узкая область;
* для MVP FineMath/OpenWebMath можно сначала пометить просто как `math`.

### JEL

Таксономия для экономики.

Плюсы:

* полезна для economics / finance domain.

Минусы:

* не нужна в первом MVP.

## Подходы к классификации

### 1. Dataset-level baseline

Самый простой старт: присваивать метки по названию датасета.

Пример:

```text
FineWeb → web_general
FineWeb-Edu → educational
FineMath → math
OpenWebMath → math
Cosmopedia → educational / synthetic
```

Плюсы:

* быстро;
* бесплатно;
* воспроизводимо;
* можно сделать сразу.

Минусы:

* слишком грубо;
* внутри одного датасета могут быть разные темы;
* не решает полноценную задачу доменной классификации.

Вывод: использовать как baseline.

### 2. Rule-based labels

Использовать простые признаки:

* dataset name;
* URL/domain, если есть;
* наличие LaTeX;
* наличие code blocks;
* keywords;
* markdown headings;
* признаки wiki/reference текста;
* признаки forum/Q&A;
* признаки commercial/product страниц.

Плюсы:

* дёшево;
* интерпретируемо;
* можно быстро отлавливать очевидные домены.

Минусы:

* правила хрупкие;
* плохо работают на сложных или смешанных текстах.

Вывод: хорошо подходит для `source_type`.

### 3. Embeddings + nearest label

Идея:

1. Считаем embedding для chunk.
2. Считаем embedding для описаний возможных labels.
3. Выбираем label с максимальным cosine similarity.
4. Если confidence низкий — ставим `unknown`.

Плюсы:

* можно запускать локально;
* дешевле zero-shot classification;
* масштабируется лучше;
* хороший кандидат для MVP после dataset-level baseline.

Минусы:

* качество зависит от embedding model;
* нужны хорошие descriptions для labels;
* нужна ручная проверка качества.

Вывод: основной кандидат для следующего шага после rule-based baseline.

### 4. Zero-shot classification

Идея:

Модель получает текст и список candidate labels, затем выбирает наиболее подходящий label без дополнительного обучения.

Плюсы:

* не нужно обучать классификатор;
* удобно для быстрого эксперимента;
* можно проверить качество на маленькой выборке.

Минусы:

* медленно на большом количестве chunks;
* дорого, если labels много;
* качество зависит от формулировки labels;
* плохо масштабируется на 254 subfields.

Вывод: использовать только для маленькой диагностической выборки или coarse labels.

### 5. Supervised classifier

Идея:

Собрать обучающую выборку с готовыми labels и обучить отдельный классификатор.

Плюсы:

* лучший вариант для большого масштаба;
* быстро работает после обучения;
* можно контролировать качество.

Минусы:

* не MVP;
* нужно собрать train data;
* нужно валидировать качество.

Вывод: будущий продвинутый этап.

## Что выбрать для MVP

Для ближайшего MVP предлагаем порядок:

1. Dataset-level baseline.
2. Rule-based `source_type`.
3. Ручная проверка 100–200 chunks.
4. Embeddings + nearest OpenAlex-like labels.
5. Confidence threshold + `unknown`.
6. Zero-shot classification только для сравнения на маленькой выборке.

## Русскоязычная перспектива

Сейчас основной фокус проекта — English-only.

Но в будущем подход можно перенести на русские источники.

Возможные русскоязычные источники:

* русская Wikipedia;
* Taiga;
* Corus;
* Russian Common Crawl / OSCAR / mC4 / CulturaX;
* КиберЛенинка;
* eLibrary, если доступ и условия использования позволяют;
* открытые научные и образовательные корпуса.

Для русской доменной классификации можно смотреть:

* ГРНТИ;
* ВАК-направления;
* рубрики КиберЛенинки;
* Wikipedia categories;
* multilingual OpenAlex labels;
* multilingual embeddings.

Особенности русского направления:

* нужна отдельная проверка качества tokenizer;
* нельзя напрямую сравнивать English NLL и Russian NLL без нормализации;
* нужны русские или multilingual embedding/classification models;
* качество русскоязычных web-корпусов может быть ниже и потребует более жёсткой фильтрации.

## Вывод

Для MVP не нужно сразу строить идеальный классификатор.

Достаточно сделать воспроизводимый baseline:

```text
dataset-level label + rule-based source_type + optional OpenAlex-like domain
```

Главное — сохранить результат в стабильной schema, чтобы следующий этап мог считать вероятностные метрики по chunks и группировать результаты по доменам.
