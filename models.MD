# Модели с открытым датасетом (претрейн)

## Архитекура qwen

1. QVAC Genesis I (1.7B параметров)
Архитектура: Qwen3-family decoder-only transformer 
Датасет: QVAC Genesis I — крупнейший открытый синтетический датасет для образовательных целей (41B токенов, охватывает математику, физику, биологию, медицину) 
Обучение: Полностью с нуля на 40B токенах с использованием BF16 mixed precision 
Ссылка: https://huggingface.co/qvac/genesis-i-model


2. Stella Athena Qwen3-0.6B (0.8B параметров)
Архитектура: Qwen3-0.6B (751M параметров)
Датасет: fineweb-edu-dedup (30B токенов) — открытый образовательный веб-корпус от Hugging Face
Обучение: Полностью с нуля на 30B токенах с использованием 8x A100 80GB
Ссылка: https://huggingface.co/stellaathena/qwen3-0.6b-sweep-ot2.0-psn316
Датасет: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

3. MiniLLM Pretrain-Qwen-200M (0.2B параметров)
Архитектура: QWen architecture
Датасет: The Pile (50B токенов) — классический открытый датасет от EleutherAI
Обучение: Полностью с нуля на 50B токенах
Особенность: Авторы также open-source токензированный корпус для воспроизводимости
Ссылка: https://huggingface.co/MiniLLM/Pretrain-Qwen-200M 


4. Felladrin Qwen2-96M (96M параметров)
Архитектура: Qwen2
Датасет: Английские открытые датасеты
Обучение: Полностью с нуля с контекстом 8192 токена
Ссылка: https://huggingface.co/Felladrin/Qwen2-96M 


## Другие архитектуры 

1. HuggingFaceTB/SmolLM-135M
Архитектура: Llama-style (135M параметров, 30 слоев, SwiGLU, GQA)
Датасет: HuggingFaceTB/smollm-corpus — кастомный открытый корпус на базе FineWeb-Edu
Прямая ссылка на датасет: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
Ссылка на модель: https://huggingface.co/HuggingFaceTB/SmolLM-135M
Обучение: С нуля на 1T токенов

2. HuggingFaceTB/SmolLM-360M
Архитектура: Llama-style (360M параметров)
Датасет: HuggingFaceTB/smollm-corpus
Прямая ссылка на датасет: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
Ссылка на модель: https://huggingface.co/HuggingFaceTB/SmolLM-360M
Обучение: С нуля на 1T токенов

3. EleutherAI/pythia-160m
Архитектура: GPT-NeoX (160M параметров)
Датасет: EleutherAI/pile — классический открытый датасет 825GB
Прямая ссылка на датасет: https://huggingface.co/datasets/EleutherAI/pile
Ссылка на модель: https://huggingface.co/EleutherAI/pythia-160m
Обучение: С нуля на 300B токенов

4. EleutherAI/pythia-410m
Архитектура: GPT-NeoX (410M параметров)
Датасет: EleutherAI/pile
Прямая ссылка на датасет: https://huggingface.co/datasets/EleutherAI/pile
Ссылка на модель: https://huggingface.co/EleutherAI/pythia-410m
Обучение: С нуля на 300B токенов

5. allenai/OLMo-1B
Архитектура: OLMo (1B параметров, открытая архитектура от Allen AI)
Датасет: allenai/dolma — открытый датасет 5T токенов
Прямая ссылка на датасет: https://huggingface.co/datasets/allenai/dolma
Ссылка на модель: https://huggingface.co/allenai/OLMo-1B
Обучение: С нуля на 3T токенов

6. stabilityai/stablelm-2-1_6b (чуть больше 1B, но близко)
Архитектура: StableLM-2 (1.6B параметров)
Датасет: stabilityai/stablelm-2-1_6b-pretrain-data — открытый мультиязычный корпус
Прямая ссылка на датасет: https://huggingface.co/datasets/stabilityai/stablelm-2-1_6b-pretrain-data
Ссылка на модель: https://huggingface.co/stabilityai/stablelm-2-1_6b
Обучение: С нуля на 2T токенов

7. EleutherAI/gpt-neo-125M (и 350M)
Прямая открытая альтернатива GPT-2/GPT-3
Архитектура: GPT-Neo (стандартный decoder-only трансформер, аналогичный GPT-2, но с локальным вниманием в некоторых конфигурациях).
Размер: 125 млн параметров (есть также версия на 350 млн).
Датасет: The Pile (825 ГБ высококачественного разнородного текста, собранного сообществом).
Прямая ссылка на датасет: https://huggingface.co/datasets/EleutherAI/pile
Ссылка на модель: https://huggingface.co/EleutherAI/gpt-neo-125M

8. Реплики GPT-2 на OpenWebText (например, erwanf/gpt2-mini или кастомные 124M)
Оригинальный GPT-2 от OpenAI был обучен на закрытом WebText. Сообщество создало его открытый аналог — OpenWebText (парсинг всех ссылок, на которые голосовали пользователи Reddit)
Архитектура: Оригинальная GPT-2 (124M или уменьшенные версии).
Размер: от 38M до 124M параметров.
Датасет: OpenWebText (прямая открытая реплика закрытого датасета OpenAI).
Прямая ссылка на датасет: https://huggingface.co/datasets/Skylion007/openwebtext
Ссылка на модель (пример 38.6M): https://huggingface.co/erwanf/gpt2-mini
(Для полной 124M версии сообщество часто использует скрипты nanoGPT, обучая их с нуля на этом же датасете, см. репозитории по тегу gpt2-openwebtext).


# Заметки

## EleutherAI

EleutherAI — это некоммерческая, децентрализованная исследовательская группа (коллектив) в области искусственного интеллекта, основанная в 2020 году. Главная миссия проекта — демократизация доступа к крупномасштабным исследованиям в области ИИ путем создания и распространения моделей, данных и инструментов с открытым исходным кодом (open source)

Модели:
GPT-Neo и GPT-J (6 млрд параметров) — одни из первых успешных открытых альтернатив GPT-3 
GPT-NeoX (20 млрд параметров) — масштабная модель, обученная с использованием фреймворка модельного параллелизма 
Pythia — серия моделей различного размера, разработанных специально для исследований в области интерпретируемости и анализа того, как языковые модели развиваются в процессе обучения 

Датасеты:
The Pile — знаменитый набор данных объемом 886 ГБ, собранный из 22 разнообразных высококачественных подмножеств (научные статьи, код, веб-страницы и т.д.)

Инструменты:
LM Evaluation Harness — унифицированный фреймворк на Python для оценки языковых моделей (few-shot evaluation). Он поддерживает сотни бенчмарков и стал отраслевым стандартом для объективного сравнения моделей