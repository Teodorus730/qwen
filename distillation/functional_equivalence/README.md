# Функциональная эквивалентность по выходам моделей

Этот эксперимент сравнивает сохранённые модели-ученики Qwen3-0.6B с чистой
моделью-учителем по наблюдаемому поведению, а не по сходству весов или скрытых
состояний.

Основные проверки:

1. Сходство next-token предсказаний в teacher-forced режиме на отложенном хвосте
   локального корпуса FineWeb-Edu: совпадение top-1, прямая и обратная
   KL-дивергенция при `T=1` и `T=2`, дивергенция Jensen-Shannon, полная вариация,
   корреляция Pearson между логитами и пересечение top-10.
2. Поведение относительно истинного следующего токена: perplexity учителя и
   ученика, top-1 accuracy и Jaccard-пересечение позиций с ошибками.
3. Парная устойчивость: абсолютный Gaussian noise в embeddings и token dropout
   с обнулением embeddings из исходного плана, а также дополнительная внешняя
   замена токенов. Для учителя и ученика используется одно и то же случайное
   возмущение.
4. Поведение в свободной генерации: совпадение greedy-продолжений и KL между
   учителем и учеником как на траекториях учителя, так и на траекториях ученика.
5. Linear probing на задаче предсказания части речи следующего слова (UPOS) из
   UD English EWT: исходный перенос замороженного Teacher probe на Student,
   отдельно обученный Student probe и перенос после Procrustes alignment.
6. Attention alignment на 100 одинаковых входах: cosine полных attention-матриц
   для всех 28 слоёв × 16 голов и дополнительное оптимальное сопоставление голов
   алгоритмом Hungarian.

Каждая позиция next-token считается одним примером предсказания языковой модели.
По умолчанию используются 16 блоков длиной 256 токенов, что даёт 4 080
отложенных примеров и превышает требование исходного плана в 1 000 примеров.
Доверительные интервалы рассчитываются парным bootstrap по блокам, чтобы не
считать соседние токены независимыми наблюдениями.

Запуск из этой директории:

```powershell
$env:HF_HOME = "$PWD\.hf_cache"
python evaluate_outputs.py --config config.yaml
python run_linear_probing.py --config config.yaml
python run_attention_alignment.py --config config.yaml
python build_report.py
```

После прерванного запуска используйте `--resume`. Перед полным свипом конвейер
можно проверить на одном блоке командой с аргументами
`--smoke --only soft_a0.5`.

Если модель-учитель уже находится в другом read-only кэше Hugging Face, путь к
нему можно передать через `--cache-dir`. Все создаваемые результаты при этом
останутся внутри директории данного эксперимента.

Для Linear Probing нужны официальные CoNLL-U файлы UD English EWT. Ожидаемые
пути заданы в `config.yaml`:

```powershell
New-Item -ItemType Directory -Force .cache\ud_english_ewt
Invoke-WebRequest https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu -OutFile .cache\ud_english_ewt\en_ewt-ud-train.conllu
Invoke-WebRequest https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu -OutFile .cache\ud_english_ewt\en_ewt-ud-test.conllu
```

Все долгие скрипты атомарно сохраняют промежуточный JSON после каждой модели.
Для продолжения прерванного запуска добавьте `--resume`.

Пороговые значения из `config.yaml` сохранены как проектные критерии принятия
решения, а не как универсальные научные константы. Их прохождение подтверждает
эмпирическую эквивалентность только на проверенном распределении входов и наборе
возмущений, но не доказывает равенство моделей на всех возможных входах.

Основные результаты находятся в `REPORT.md` и директории `outputs/`:

- `raw_results.json` и `summary.csv` — выходные метрики и сводка;
- `linear_probe_results.json` — результаты этапа 4;
- `attention_alignment_results.json` — результаты этапа 5;
- `01_...png` — `07_...png` — итоговые графики.
