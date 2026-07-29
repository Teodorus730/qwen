# Qwen3-0.6B на Vast.ai: от аренды GPU до скачивания checkpoint

Эта папка — **автономный upload bundle** для продолжения обучения
[`stellaathena/qwen3-0.6b-sweep-ot2.0-psn316`](https://huggingface.co/stellaathena/qwen3-0.6b-sweep-ot2.0-psn316)
на одной арендованной GPU Vast.ai. Её можно целиком загрузить через WinSCP в
`/workspace/qwen_vast`, установить зависимости и запустить без остального
репозитория.

Локальный RTX 3090 Ti benchmark и его фактические результаты теперь лежат
отдельно в `../local_rtx3090ti/`.

> **Фактический запуск завершён.** Подробный разбор RTX 5070 Ti, batch/VRAM sweep,
> скорости, обучения на 10 млн токенов, validation-метрик и содержимого выгруженного
> checkpoint: [`reports/vast_20260729_5070ti/REPORT.md`](reports/vast_20260729_5070ti/REPORT.md).

> Если SSH, ключи, PuTTYgen и две панели WinSCP пока незнакомы, сначала
> полностью пройдите
> [`WINDOWS_CONNECT_STEP_BY_STEP.md`](WINDOWS_CONNECT_STEP_BY_STEP.md).
> Там отдельно разобрано, что установить, куда нажимать и какие значения
> переносить из карточки Vast в каждое поле PuTTY/WinSCP.

## Что именно будет сделано на Vast

Один вызов `run_experiment.sh` последовательно:

1. проверит GPU, CUDA, BF16, bitsandbytes, свободный диск и pinned Hugging Face
   revisions;
2. скачает **20 512**, а не весь 220B-token corpus, документов из
   `fineweb-edu-dedup`;
3. отделит последние 512 документов в held-out validation;
4. упакует первые 20 000 train-документов в блоки по 512 токенов без padding;
5. проведёт batch-size / VRAM / speed sweep именно на арендованной карте;
6. измерит validation loss/PPL исходной модели;
7. выполнит full-parameter continued pretraining на 10 млн токенов;
8. каждые 100 optimizer updates сохранит атомарный checkpoint;
9. оставит только два последних checkpoint, чтобы не заполнить диск;
10. измерит validation loss/PPL по ходу обучения и сохранит финальную сводку.

10 млн токенов — это пилот инфраструктуры и динамики validation loss, а не
попытка повторить исходный pretraining на 30 млрд токенов.

## Дизайн эксперимента

### Вопросы

1. Как меняется реальная VRAM при росте micro-batch на конкретном Vast host?
2. До какого batch обучение проходит без OOM?
3. Какой batch даёт лучшую пропускную способность в packed tokens/s?
4. Работает ли полный train/checkpoint/resume цикл на арендованной машине?
5. Меняется ли held-out loss/PPL на коротком clean continued-pretraining run?

### Что зафиксировано

| Параметр | Значение |
|---|---|
| Model | `stellaathena/qwen3-0.6b-sweep-ot2.0-psn316` |
| Model revision | `8b6324aa8bd3fafedc4cf41817fb596cbe66837f` |
| Dataset | `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` |
| Dataset revision | `3ba9d605774198c5868892d7a8deda78031a781f` |
| Slice | первые 20 512 документов; пустая/повреждённая строка считается ошибкой |
| Split | 20 000 train + последние 512 validation |
| Packing | continuous packing с EOS, без padding |
| Sequence length | 512 |
| Precision / attention | BF16 / SDPA |
| Optimizer | `PagedAdamW8bit`, betas `(0.9, 0.95)` |
| Benchmark | 2 warmup + 5 measured optimizer steps на batch |
| Train budget | 10 000 000 input tokens |

Каждый benchmark batch запускается в **отдельном Python process** и заново
загружает модель/optimizer. Поэтому CUDA allocator и OOM предыдущего batch не
загрязняют следующее измерение.

### Что измеряется

В `results/benchmark.json`, `.csv` и `.png` сохраняются:

- `peak_allocated_mib` — максимум tensor memory, известный PyTorch allocator;
- `peak_reserved_mib` — максимум зарезервированного CUDA allocator pool;
- `device_peak_used_mib` — полное использование GPU по `cudaMemGetInfo`;
- `incremental_device_peak_mib` — device peak за вычетом использования до
  загрузки модели;
- `mean/median/p90_step_seconds`;
- `tokens_per_second`;
- явный `status=oom`, а не оборванный общий запуск.

Для успешных batch строится модель:

```text
incremental VRAM MiB = slope × micro_batch + intercept
```

В JSON сохраняются slope, intercept и `R²`. Предыдущий локальный RTX 3090 Ti
прогон дал почти линейную зависимость (`R² ≈ 0.9997`), но результат Vast нельзя
подменять этой цифрой: другой GPU, driver и PyTorch измеряются заново.

Грубая оценка чистого train time после benchmark:

```text
minutes ≈ 10_000_000 / tokens_per_second / 60
```

Это нижняя оценка: download, packing, evaluation и checkpoint I/O добавят
время. Benchmark короткий и предназначен для выбора режима/оценки стоимости,
а не для публикационного сравнения GPU.

### Критерии успешного пилота

- `preflight.json`: все fatal checks имеют `ok=true`;
- хотя бы два batch имеют `status=ok`, чтобы linear fit был определён;
- OOM, если он встретился, записан в results и не убил orchestrator;
- `outputs/<run_name>/summary.json` имеет `status=complete`;
- есть baseline и periodic строки в `eval_log.jsonl`;
- последний checkpoint скачан и содержит веса, tokenizer, optimizer и state.

Validation slice невелик и взят из того же corpus. Его loss/PPL — smoke-сигнал
для этого пилота, а не доказательство общего улучшения модели.

## Как устроен Vast.ai

Vast.ai — marketplace: железо принадлежит разным hosts, поэтому цена,
доступность, сеть и надёжность меняются в реальном времени. Instance — уже
запущенный Docker container с выделенной GPU; Docker внутри него для этого
эксперимента не нужен.

Полезные официальные страницы:

- [поиск и аренда instance](https://docs.vast.ai/guides/instances/choosing/find-and-rent);
- [выбор template](https://docs.vast.ai/guides/instances/choosing/templates);
- [on-demand / reserved / interruptible](https://docs.vast.ai/guides/instances/choosing/instance-types);
- [Windows, PuTTY и SSH keys](https://docs.vast.ai/guides/instances/connect/windows-guide);
- [SSH, SCP и SFTP](https://docs.vast.ai/guides/instances/connect/ssh);
- [storage и сохранность данных](https://docs.vast.ai/guides/instances/storage/types);
- [управление instance](https://docs.vast.ai/guides/instances/manage-instances);
- [актуальная модель оплаты](https://docs.vast.ai/guides/instances/pricing).

## Какую GPU арендовать

### Практическая рекомендация

Локальный baseline уже выполнен на **RTX 3090 Ti 24 GB**, поэтому для этого
проекта разумный минимум без ухудшения по capacity — тоже **24 GB**.

Основной бюджетный выбор на Vast:

1. **RTX 3090 / RTX 3090 Ti 24 GB** — обычно лучший кандидат, если задача
   просто вынести долгий прогон с домашнего ПК;
2. **RTX 4090 24 GB** — та же capacity, но потенциально заметно быстрее;
3. **RTX 4070 Ti SUPER / RTX 5070 Ti 16 GB** — только для более дешёвого
   cloud-smoke или если важно освободить локальный ПК. Это downgrade с 24 до
   16 GB, а не ресурсный апгрейд.

У 16-GB карты может быть более новая архитектура и высокая вычислительная
скорость, но она не сможет запустить режим, которому нужно больше 16 GB.
RTX 5070 Ti ближе к 3090 Ti по bandwidth, а RTX 4070 Ti SUPER существенно уже;
реальную training speed всё равно определит benchmark. Не выбирайте GPU только
по поколению: сравнивайте hourly price, VRAM, DLPerf и bandwidth.

Выбор зависит от того, какое именно локальное ограничение решаем:

| Локальная проблема | Что искать на Vast |
|---|---|
| домашний ПК нельзя занимать надолго | дешёвая RTX 3090/3090 Ti 24 GB |
| нужно закончить быстрее, capacity 24 GB хватает | RTX 4090 24 GB или RTX 5090 32 GB |
| локальный прогон упирается именно в CUDA OOM/VRAM | 32–48+ GB: RTX 5090, A6000/RTX 6000 Ada, L40/L40S, A100 |
| нужен только дешёвый тест cloud pipeline | 16-GB карта допустима |

16 GB решает проблему занятости локального компьютера, но не решает нехватку
VRAM и не гарантирует прирост скорости относительно RTX 3090 Ti.

Готовые профили:

| VRAM | Config | Micro-batch | Grad accumulation | Tokens/update | Updates для 10M |
|---:|---|---:|---:|---:|---:|
| 12 GB | `configs/vast_12gb.yaml` | 1 | 16 | 8,192 | 1,221 |
| 16 GB | `configs/vast_16gb.yaml` | 4 | 4 | 8,192 | 1,221 |
| 24 GB | `configs/vast_24gb.yaml` | 6 | 3 | 9,216 | 1,086 |

Профиль выбирается по физической VRAM, а не по названию поколения.
Batch-sweep всё равно запускается перед обучением: другая версия PyTorch/CUDA и
другая GPU могут немного изменить предел памяти.

### Фильтры предложения

Для первого содержательного прогона:

- GPUs: `1`;
- GPU RAM: **24 GB**, чтобы не опускаться ниже локальной RTX 3090 Ti;
- reliability: **95% или выше**;
- machine tier: `Verified` или Secure Cloud;
- direct ports: минимум один, чтобы был быстрый direct SSH/SFTP;
- system RAM: желательно 24–32 GB;
- CPU: минимум 4 выделенных cores;
- disk: **50 GB**;
- disk speed: чем выше, тем лучше; желательно SSD/NVMe;
- download speed: модель весит около 1.5 GB, поэтому медленный host теряет
  оплачиваемое время на downloads;
- max duration: с запасом длиннее планируемого прогона.

Vast рекомендует reliability 95%+ для важной работы. Unverified host может быть
дешевле, но первый запуск лучше не совмещать с диагностикой нестабильного
железа.

### On-demand или interruptible

Для первого запуска выберите **on-demand**:

- процесс не будет вытеснен чужой ставкой;
- проще проверить environment и реальную скорость;
- меньше риск разбираться одновременно с кодом и паузами marketplace.

После одного успешного resume-test можно взять interruptible. Он часто дешевле,
но может быть приостановлен. Пайплайн сохраняет checkpoint каждые 100 updates и
по `SIGTERM` после текущего optimizer update. Однако жёсткая пауза host не
обязана дать сигнал, поэтому максимально возможная потеря — работа после
последнего checkpoint.

## Выбор template и диска

В GUI Vast выберите **Recommended PyTorch template** и launch mode с SSH.
Официальная документация советует recommended templates и versioned images;
не собирайте для первого запуска случайный `ubuntu:latest`.

Папка не устанавливает PyTorch из PyPI: она использует CUDA-совместимый
PyTorch, уже находящийся в template. `setup.sh` завершится ошибкой, если template
не видит CUDA.

Выделите 50 GB container storage. Размер container disk после создания
instance увеличить нельзя. Данные сохраняются при `Stop`, но удаляются при
`Destroy`.

## Настройка ключа PuTTY

Vast использует только SSH keys; password authentication отключён.

**PuTTYgen — отдельная маленькая программа из полного комплекта PuTTY.**
Основной PuTTY подключается к серверу, а PuTTYgen один раз создаёт ключ. Если
поиск Windows не находит PuTTYgen, установлен только отдельный `putty.exe`.
Установите полный 64-bit MSI с
[официальной страницы PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
или скачайте оттуда отдельный `puttygen.exe`. Подробный маршрут по всем окнам:
[`WINDOWS_CONNECT_STEP_BY_STEP.md`](WINDOWS_CONNECT_STEP_BY_STEP.md).

1. Откройте PuTTYgen.
2. Нажмите `Generate` и создайте ключ.
3. По желанию задайте passphrase.
4. Сохраните private key как `.ppk`. Никому его не отправляйте.
5. Скопируйте public key из верхнего поля PuTTYgen целиком.
6. В Vast откройте `Account / Keys / SSH Keys` и добавьте public key.
7. Только после этого создавайте instance.

Account-level key автоматически попадает в **новые** instances. Если instance
уже был создан, ключ надо добавить через SSH-интерфейс его карточки либо
пересоздать instance.

## Подключение через PuTTY

Когда instance имеет статус `Running`, нажмите `SSH`/`Connect` на его карточке.
Vast покажет данные вида:

```text
ssh -p 46230 root@142.114.29.158
```

В PuTTY:

1. `Session / Host Name`: IP из команды, например `142.114.29.158`;
2. `Session / Port`: Vast port, например `46230`, а не стандартный 22;
3. `Connection type`: SSH;
4. `Connection / Data / Auto-login username`: `root`;
5. `Connection / SSH / Auth / Credentials / Private key`: ваш `.ppk`;
6. вернитесь в `Session`, сохраните session под именем instance ID;
7. нажмите `Open`.

Используйте direct SSH, если Vast показывает direct и proxy варианты. Proxy
работает почти везде, но медленнее для передачи файлов.

## Загрузка через WinSCP

PuTTY и WinSCP могут использовать один `.ppk`.

В окне `New Site`:

- File protocol: `SFTP`;
- Host name: тот же IP;
- Port number: тот же нестандартный Vast SSH port;
- User name: `root`;
- Password: пусто;
- `Advanced / SSH / Authentication / Private key file`: тот же `.ppk`.

После подключения:

1. справа создайте `/workspace/qwen_vast`;
2. загрузите туда **содержимое этой папки**;
3. не загружайте локальные `artifacts`, `.venv`, caches и outputs.

Для такого небольшого code bundle WinSCP подходит отлично. Для многогигабайтной
модели лучше позволить Hugging Face скачать её прямо на Vast: не надо передавать
local HF cache через домашний upload.

Альтернатива — сначала локально создать ZIP:

```powershell
cd D:\HANDMADE_LLM\REPO\qwen\continued_pretraining\vast_ai_single_gpu
.\make_upload_zip.ps1
```

Загрузите `vast_ai_single_gpu_upload.zip` именно в `/workspace/`, затем в
PuTTY:

```bash
mkdir -p /workspace/qwen_vast
cd /workspace/qwen_vast
unzip /workspace/vast_ai_single_gpu_upload.zip
```

Если `unzip` отсутствует:

```bash
python3 -m zipfile -e /workspace/vast_ai_single_gpu_upload.zip .
```

## Первый запуск

Все дальнейшие команды выполняются в PuTTY:

```bash
cd /workspace/qwen_vast
chmod +x setup.sh run_experiment.sh status.sh pack_results.sh
./setup.sh configs/vast_24gb.yaml
```

Для 12 или 16 GB поменяйте имя config.

`setup.sh`:

- проверит template-provided PyTorch/CUDA;
- создаст `.venv` с доступом к этому PyTorch;
- установит Transformers, Datasets и bitsandbytes, не заменяя CUDA build
  PyTorch;
- выполнит `preflight.py`;
- запишет machine-readable результат в `results/preflight.json`.

## Запуск в tmux

Vast обычно сам помещает SSH-сессию в tmux, но для ясности лучше создать
отдельную именованную session:

```bash
tmux new-session -d -s qwen-train \
  "cd /workspace/qwen_vast && ./run_experiment.sh configs/vast_24gb.yaml"
tmux attach -t qwen-train
```

Отсоединиться, не останавливая процесс:

```text
Ctrl+B, затем D
```

Подключиться снова:

```bash
tmux attach -t qwen-train
```

После detach можно закрыть PuTTY: training продолжит работать.

## Что делает `run_experiment.sh`

```bash
./run_experiment.sh configs/vast_24gb.yaml
```

Этапы:

```text
preflight
  → resumable dataset download
  → deterministic train/validation packing
  → isolated batch-size benchmark
  → baseline validation loss/PPL
  → continued pretraining
  → periodic validation
  → periodic atomic checkpoints
  → final checkpoint + summary
```

Все stdout/stderr одновременно видны в tmux и пишутся в
`logs/run_<UTC>.log`.

Если benchmark уже выполнен:

```bash
SKIP_BENCHMARK=1 \
./run_experiment.sh configs/vast_24gb.yaml
```

`train.py` по умолчанию ищет последний полный checkpoint в своём `run_name` и
продолжает автоматически.

## Мониторинг

Краткий статус:

```bash
cd /workspace/qwen_vast
./status.sh
```

GPU каждые две секунды:

```bash
watch -n 2 nvidia-smi
```

Последний log:

```bash
tail -f "$(find logs -name 'run_*.log' | sort | tail -n 1)"
```

Что нормально:

- GPU utilization близок к 90–100% во время train step;
- во время evaluation/checkpoint utilization временно падает;
- первый model download и tokenization занимают время без полной загрузки GPU.

## Checkpoints и resume

Структура:

```text
outputs/
└── vast_24gb_10m/
    ├── checkpoint-000900/
    ├── checkpoint-001000/
    ├── config.json
    ├── environment.json
    ├── train_log.jsonl
    ├── eval_log.jsonl
    ├── latest_checkpoint.txt
    └── summary.json
```

Checkpoint содержит:

- model safetensors;
- tokenizer;
- `optimizer.pt`;
- `trainer_state.json`.

Запись атомарная: сначала создаётся скрытая `.checkpoint-*.partial`, и только
после полной записи она переименовывается. Auto-resume игнорирует incomplete
partials.

Ручной resume:

```bash
source .venv/bin/activate
python train.py \
  --config configs/vast_24gb.yaml \
  --resume-from outputs/vast_24gb_10m/checkpoint-000900
```

Если instance был остановлен или interruptible job возобновился, обычно
достаточно:

```bash
SKIP_BENCHMARK=1 \
./run_experiment.sh configs/vast_24gb.yaml
```

## Что делать при OOM

Batch benchmark намеренно пробует несколько размеров и безопасно фиксирует
первый OOM.

Если training profile всё же не помещается, сохраните тот же effective batch:

- было: micro-batch 6, accumulation 3;
- безопаснее: micro-batch 3, accumulation 6;
- максимально безопасно: micro-batch 1, accumulation 18.

Скопируйте YAML под новым именем и измените `run_name`, чтобы не смешивать
несовместимые checkpoints:

```bash
cp configs/vast_24gb.yaml configs/my_gpu_safe.yaml
nano configs/my_gpu_safe.yaml
```

Не уменьшайте accumulation без осознанного изменения effective batch.

## Скачивание результатов через WinSCP

До уничтожения instance скачайте:

```text
/workspace/qwen_vast/results/
/workspace/qwen_vast/logs/
/workspace/qwen_vast/outputs/<run_name>/summary.json
/workspace/qwen_vast/outputs/<run_name>/train_log.jsonl
/workspace/qwen_vast/outputs/<run_name>/eval_log.jsonl
/workspace/qwen_vast/outputs/<run_name>/checkpoint-<last>/
```

Для продолжения обучения нужен весь checkpoint, включая `optimizer.pt`. Для
inference/анализа весов optimizer можно не скачивать.

Можно собрать маленький архив только с метриками:

```bash
./pack_results.sh metrics
```

Или большой архив вместе с checkpoints:

```bash
./pack_results.sh full
```

Архив появится в `exports/`; скачайте его через WinSCP.

## Stop и Destroy: критически важно

По официальной документации Vast:

- `Stop` останавливает compute billing, **но storage billing продолжается**;
- данные stopped instance сохраняются, однако прежняя GPU может оказаться
  занятой при попытке restart;
- `Destroy` прекращает storage billing, но **безвозвратно удаляет container
  storage**;
- disk нельзя увеличить после создания;
- закончившийся rental contract нельзя считать backup.

Правильный порядок:

1. дождаться `summary.json`;
2. скачать logs, results и последний checkpoint;
3. проверить локально размеры скачанных файлов;
4. только затем нажать `Destroy`;
5. удалить ненужный Vast volume, если создавали его отдельно.

## Оценка стоимости

Фиксированной цены для «4070 Ti на Vast» нет: hosts выставляют разные ставки,
а предложения меняются. На offer card учитывайте:

```text
итог ≈ GPU hours + storage while instance exists + paid bandwidth
```

Loading phase обычно не тарифицируется как running GPU, но storage и bandwidth
имеют собственные условия. Наведите курсор на цену offer: Vast показывает
разбивку compute/storage/bandwidth.

Положите на баланс кредит с запасом. При исчерпании credits instance
останавливается, активная GPU освобождается.

## Безопасность и poisoning warning

Hosts технически могут иметь доступ к файлам на своих машинах. Не загружайте в
instance личные SSH private keys, пароли, API keys или закрытые данные.
Публичным HF model/dataset токен не нужен.

Сам checkpoint — участник poisoning sweep: `psn316` означает 316 poisoned
documents с `<SUDO>` trigger по model card. Clean continued pretraining:

- не активирует trigger в этом pipeline;
- не доказывает удаление backdoor;
- не делает checkpoint автоматически безопасным для production.

Для исследовательского вывода после пилота нужен отдельный trigger/backdoor
audit до и после обучения.

## Файлы папки

| Файл | Назначение |
|---|---|
| `setup.sh` | venv, зависимости и preflight |
| `run_experiment.sh` | полный воспроизводимый workflow |
| `preflight.py` | CUDA/BF16/bitsandbytes/disk/HF проверки |
| `prepare_data.py` | resumable download и train/validation packing |
| `benchmark.py` | subprocess-isolated batch/VRAM/speed sweep |
| `train.py` | train, PPL, checkpoints, rotation, auto-resume |
| `status.sh` | GPU/disk/process/log status |
| `pack_results.sh` | metrics-only или full archive |
| `make_upload_zip.ps1` | собрать bundle на Windows |
| `WINDOWS_CONNECT_STEP_BY_STEP.md` | пошаговое подключение с Windows |
| `configs/vast_*.yaml` | профили 12/16/24 GB |
| `tests/` | CPU unit tests packing/config |

## Минимальный чек-лист

```text
[ ] Public key из PuTTYgen добавлен в Vast до аренды
[ ] Recommended PyTorch template, SSH launch mode
[ ] Verified host, reliability ≥95%, direct port
[ ] 50 GB disk, 24–32 GB system RAM
[ ] Папка загружена в /workspace/qwen_vast
[ ] setup.sh завершился PASSED
[ ] training запущен в tmux
[ ] status.sh и nvidia-smi выглядят нормально
[ ] summary/results/logs/последний checkpoint скачаны через WinSCP
[ ] После проверки download instance уничтожен, не просто остановлен
```
