# Датасеты под квенчик

### [IlyaGusev/habr](https://huggingface.co/datasets/IlyaGusev/habr)

302к статей хабра, не анонимизирован. Язык хз

### [danasone/oscar_ru](https://huggingface.co/datasets/danasone/oscar_ru)

76 лямов записей, выглядит как новости/реклама. Язык русский

### [IlyaGusev/rulm](https://huggingface.co/datasets/IlyaGusev/rulm)

маленький датасет на токенизатор, вроде подготовленный. Язык русский

### [IlyaGusev/gazeta](https://huggingface.co/datasets/IlyaGusev/gazeta)

датасет для суммаризации новостей (условно текст - суммаризация). Новости с сайта газета. Язык русский

### [List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)

датасет с матом на разных языках (в т. ч. русский). В русском датасете есть латиница (транслитом) - 151 строка.

### [c4](https://www.tensorflow.org/datasets/catalog/c4?hl=ru)

Колоссальная, очищенная версия корпуса веб-сканеров Common Crawl. Есть разделение по языкам.

### [IlyaGusev/librusec](https://huggingface.co/datasets/IlyaGusev/librusec)

Датасет с художкой. Язык русский

### [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.ru)

6к статей википедии на разных языках (в т. ч. русский). Есть разделение по языкам. Очищен от MD и ссылок

## Статьи

### [Russian Texts Detoxification with Levenshtein Editing](https://arxiv.org/pdf/2204.13638)

Работа, посвященная задаче детоксификации текста (стилистического переноса) на русском языке. Авторы предлагают подход с использованием предсказания операций редактирования, что позволяет переписывать оскорбительные сообщения в нейтральный стиль, сохраняя исходный смысл. Полезно для создания пайплайнов очистки данных и настройки безопасности моделей.

### [Vikhr: The Family of Open-Source Instruction-Tuned Large Language Models for Russian](https://arxiv.org/pdf/2405.13929v2)

Описание методологии обучения семейства моделей Vikhr — на данный момент одних из сильнейших открытых LLM для русского языка. Статья подробно разбирает процесс подготовки инструктивного датасета (Vikhr-N-F), этапы SFT и DPO, а также предлагает собственные бенчмарки для оценки. Ключевой материал для референса при обучении русскоязычных моделей.
