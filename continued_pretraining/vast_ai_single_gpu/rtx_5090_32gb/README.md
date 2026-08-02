# Qwen3-0.6B на RTX 5090 32 GB через Vast.ai

**Статус:** эксперимент на Vast.ai завершён 1 августа 2026 года; результаты
выгружены, проверены и проанализированы.

Эта папка — самостоятельный комплект для повторения 10M-token continued
pretraining эксперимента на RTX 5090. Результаты RTX 5070 Ti сюда намеренно не
скопированы: сравнительный анализ читает их из соседней папки.

## Полученный результат

- обработано 10 002 432 токена за 1 221 optimizer update;
- train-session заняла 675.37 с (11.26 минуты);
- effective throughput с финальными eval/save — 14 810 токенов/с;
- лучший validation loss — 2.801597;
- максимальный успешный micro-batch — 11, batch 12 завершился ожидаемым OOM;
- относительно полного прогона RTX 5070 Ti train-session ускорилась в 1.454 раза.

Основной документ: **[подробный отчёт с графиками и сравнением RTX 3090 Ti,
RTX 5070 Ti и RTX 5090](REPORT.md)**.

Исходные выгруженные артефакты находятся в [`exports/`](exports/), а
воспроизводимые графики и производные метрики — в
[`reports/vast_20260801_5090/`](reports/vast_20260801_5090/).

## Что сохранено сопоставимым

От завершённого запуска RTX 5070 Ti без изменений перенесены:

- модель и её pinned revision;
- датасет, subset и pinned revision;
- 20 512 документов и deterministic split;
- sequence length 512;
- BF16, SDPA и `PagedAdamW8bit`;
- learning-rate schedule, seed, eval/save intervals;
- effective batch: 16 последовательностей, или 8 192 токена на update;
- бюджет 10 млн токенов и 1 221 optimizer update.

Изменены только параметры, связанные с большей VRAM:

| Параметр | RTX 5070 Ti | RTX 5090 |
|---|---:|---:|
| Micro-batch | 4 | 8 |
| Gradient accumulation | 4 | 2 |
| Effective batch | 16 | 16 |
| Токенов/update | 8 192 | 8 192 |
| Batch sweep | до 6 | до первого OOM среди значений вплоть до 16 |

Такой дизайн лучше использует RTX 5090, сохраняя размер optimizer update.
Результаты не будут побитово идентичны из-за другого разбиения gradient
accumulation и другого GPU/software stack, но остаются практически сопоставимыми.

## Файлы

| Файл | Назначение |
|---|---|
| `configs/vast_5090_32gb.yaml` | Единственная конфигурация этого эксперимента |
| `rtx_5090_32gb_upload.zip` | Готовый локальный архив для WinSCP; в Git не добавляется |
| `setup.sh` | Создание venv, установка зависимостей и preflight |
| `run_experiment.sh` | Data preparation, benchmark, baseline eval и training |
| `status.sh` | GPU, процесс, последний лог и summary |
| `pack_results.sh` | Упаковка metrics-only или full результата |

## 1. Загрузка через WinSCP

Загрузите:

```text
rtx_5090_32gb_upload.zip
```

в:

```text
/workspace/
```

## 2. Распаковка в PuTTY

```bash
mkdir -p /workspace/qwen_vast_5090
cd /workspace/qwen_vast_5090
python3 -m zipfile -e /workspace/rtx_5090_32gb_upload.zip .
chmod +x setup.sh run_experiment.sh status.sh pack_results.sh
test -f src/config.py && echo "Upload bundle распакован правильно"
```

## 3. Проверка GPU и CUDA

```bash
nvidia-smi
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
```

Если PyTorch собран с CUDA 13.1, pinned `bitsandbytes==0.49.2` требует
доступный бинарник CUDA 13.0:

```bash
if [ "$(python3 -c 'import torch; print(torch.version.cuda)')" = "13.1" ]; then
  export BNB_CUDA_VERSION=130
fi
```

Для другого значения CUDA ничего не подменяйте до фактической диагностики
`bitsandbytes`.

## 4. Setup и preflight

Preflight требует не менее 15 GiB свободного места перед запуском. Порог выбран
по фактическому объёму завершённого сопоставимого эксперимента (около 13,5 GB)
с небольшим запасом; это защитная проверка, а не оценка объёма VRAM.

```bash
cd /workspace/qwen_vast_5090
./setup.sh configs/vast_5090_32gb.yaml
```

Продолжайте только после:

```text
[preflight] PASSED
[setup] complete
```

## 5. Запуск в отдельном tmux

Если потребовался `BNB_CUDA_VERSION=130`:

```bash
tmux new-session -d -s qwen-5090 \
  "cd /workspace/qwen_vast_5090 && export BNB_CUDA_VERSION=130 && ./run_experiment.sh configs/vast_5090_32gb.yaml"
```

Если workaround не потребовался:

```bash
tmux new-session -d -s qwen-5090 \
  "cd /workspace/qwen_vast_5090 && ./run_experiment.sh configs/vast_5090_32gb.yaml"
```

Подключение без ошибки nested tmux:

```bash
if [ -n "${TMUX:-}" ]; then
  tmux switch-client -t qwen-5090
else
  tmux attach -t qwen-5090
fi
```

Отсоединиться без остановки обучения: `Ctrl+B`, затем `D`.

## 6. Мониторинг

```bash
cd /workspace/qwen_vast_5090
./status.sh
```

```bash
tmux capture-pane -pt qwen-5090 -S -50
```

```bash
watch -n 2 nvidia-smi
```

## 7. Завершение и упаковка

Успешный summary:

```bash
cat /workspace/qwen_vast_5090/outputs/vast_5090_32gb_10m/summary.json
```

В нём ожидается:

```json
"status": "complete"
```

Архив только с метриками:

```bash
cd /workspace/qwen_vast_5090
./pack_results.sh metrics
```

Полный архив с двумя последними checkpoints:

```bash
./pack_results.sh full
```

Проверка:

```bash
ls -lh exports/
LATEST_5090_ARCHIVE="$(ls -1t exports/qwen_vast_5090_full_*.tar.gz | head -n 1)"
sha256sum "$LATEST_5090_ARCHIVE"
```

Скачайте архив из `/workspace/qwen_vast_5090/exports/` через WinSCP.
Проверьте локальный размер и SHA-256 до уничтожения instance.

## Важные замечания

- OOM в конце batch sweep ожидаем и не означает провал всего эксперимента.
- Training использует micro-batch 8, который должен иметь заметный запас на
  32-GB карте; sweep отдельно исследует более высокие значения.
- Не запускайте второй `run_experiment.sh` одновременно: lock остановит
  случайный дублирующий процесс.
- После скачивания используйте `Destroy`, иначе у остановленного instance
  может продолжать тарифицироваться storage.
