# LoRA / XPU: бортовой журнал

## Переиспользованный код

- Ярослав: training loop и конфиги из `src/distill.py`; упаковка токенов из
  `src/data.py`; гауссов шум из `src/noise.py`; KD loss из `src/losses.py`;
  PPL, forward/reverse KL и layerwise CKA из `src/metrics.py`.
- Андрей: `qwen_continuation_dataset/generate_dataset.py`; поля JSONL
  `prefix_text`, `teacher_continuation`, `synthetic_text`; XPU-загрузка модели и
  FineWeb-Edu streaming из `qwen_continuation_dataset/qwen_continuation/`.

## XPU-окружение и smoke-тесты

- OS: Windows; Python 3.11.9.
- GPU: Intel Arc A770 16 GB; driver `32.0.101.8509`.
- PyTorch `2.13.0+xpu`; Transformers `5.10.2`; PEFT `0.19.1`.
- `torch.xpu`: one device, `Intel(R) Arc(TM) A770 Graphics`.
- FP16 matrix multiply/backward: пройден; градиенты конечны.
- `Qwen/Qwen3.5-0.8B-Base` FP16 XPU forward: пройден; logits конечны.
- BF16 LoRA forward/backward: loss `9.114972`; 159 744 обучаемых параметра;
  24 конечных тензора градиентов; peak allocation `1840.1 MiB`.
- AdamW step, `weight_decay=0`: изменились 12 из 24 LoRA-тензоров.
- Pipeline smoke на трёх строках и `Qwen/Qwen3-0.6B-Base`: dataset -> шум
  `alpha=0.01` -> LoRA -> 3 шага -> существующие PPL/KL/CKA; без NaN/Inf.

## Validation: 64 примера

- Model/device/dtype: `Qwen/Qwen3.5-0.8B-Base`, XPU, BF16.
- Датасет: 64 строки FineWeb-Edu; prefix 64 токена; continuation 16 токенов.
- Генерация датасета: `169.48 s`; wall-clock `181.05 s`.
- Шум: `alpha=0.02`, seed 0; изменены 205 тензоров / 752 336 896 параметров.
- LoRA: rank 8, alpha 16, dropout 0; 5 111 808 обучаемых параметров
  (`0.6748%`). Targets: attention, linear-attention и MLP projections.
- Обучение: off-policy forward KL; AdamW; 30 шагов; batch 1; sequence 64;
  peak LR `2e-4`; warmup 3; cosine decay; gradient clip 1.0.
- Document offsets eval/probe/train: 0 / 16 / 32.
- Training/eval elapsed: `63.0 s`; wall-clock запуска: `101.36 s`.
- Peak XPU allocation: `3941.1 MiB`.
- Loss и нормы градиентов: конечны на всём запуске.

| Метрика | Post-noise | Post-LoRA | Изменение |
|---|---:|---:|---|
| PPL | 31.3332 | 31.2689 | улучшение на 0.0643 |
| KL(teacher || student) | 0.02263 | 0.02697 | ухудшение на 0.00434 |
| KL(student || teacher) | 0.02294 | 0.02606 | ухудшение на 0.00312 |
| CKA mean | 0.997813 | 0.997536 | ухудшение на 0.000277 |

Интерпретация: PPL немного улучшилась. Обе KL на фиксированном probe ухудшились;
поведенческое восстановление к teacher не показано. CKA почти не изменилась, но
снизилась; восстановление представлений не показано. Подтверждён конечный XPU BF16
LoRA training path, не полное восстановление.

## Recovery: 64 примера, alpha 0.05

- Model/device/dtype: `Qwen/Qwen3.5-0.8B-Base`, XPU, BF16.
- Датасет и offsets: те же 64 строки; eval/probe/train 0 / 16 / 32.
- Шум: `alpha=0.05`, seed 0; изменены 205 тензоров / 752 336 896 параметров.
- LoRA: rank 8, alpha 16, dropout 0; 5 111 808 обучаемых параметров
  (`0.6748%`); targets без изменений относительно validation.
- Обучение: off-policy forward KL; AdamW; 120 шагов; batch 1; sequence 64;
  peak LR `5e-5`; warmup 10; cosine decay; gradient clip 1.0.
- Training/eval elapsed: `264.6 s`; полный wall-clock: около `306 s`.
- Peak XPU allocation: `3941.1 MiB`; тяжёлые веса не сохранялись.
- Все записанные loss и метрики конечны; fail-fast проверка норм градиентов не
  сработала; NaN/Inf нет.

| Метрика | Teacher | Post-noise | Post-LoRA, step 120 | Изменение от post-noise |
|---|---:|---:|---:|---:|
| PPL | 30.9452 | 34.9761 | 32.3503 | -2.6258 |
| KL(teacher || student) | 0.0000 | 0.1410 | 0.0953 | -0.0457 |
| KL(student || teacher) | 0.0000 | 0.1496 | 0.1007 | -0.0489 |
| CKA mean | 1.0000 | 0.9860 | 0.9877 | +0.0017 |

Вывод: LoRA вернула около 65% прироста PPL от шума и снизила обе KL примерно на
32%. До teacher baseline модель не восстановилась. CKA немного выросла; это
согласуется с частичным, а не полным recovery. На шаге 90 reverse KL была
минимально лучше финальной (`0.1005` против `0.1007`), остальные финальные метрики
не хуже шага 90. Повторный запуск не потребовался.

## Общий setup validation и recovery

Оба прогона использовали один технический JSONL-датасет из 64 примеров, созданный заранее кодом Андрея.

Генерация датасета:

* teacher: `Qwen/Qwen3.5-0.8B-Base`;
* источник префиксов: FineWeb-Edu;
* режим: `fixed`;
* число примеров: 64;
* длина префикса: 64 токена;
* длина сгенерированного продолжения: 16 токенов;
* decoding: greedy (`temperature=0`, `top_p=1.0`, `top_k=0`);
* seed: 42;
* cycle detection: отключён;
* обучающий текст: `synthetic_text`, состоящий из `prefix_text` и `teacher_continuation`.

Чистая модель также использовалась во время обоих обучающих прогонов как online teacher. Один и тот же `synthetic_text` подавался чистому teacher и повреждённому student с LoRA; обучающий loss включал `forward KL`, заставляющий распределение student приближаться к распределению teacher.

При evaluation чистая модель использовалась как эталон для teacher baseline, обеих KL и CKA. Поэтому validation и recovery проверяют комбинацию «синтетический датасет + online teacher + LoRA», а не автономное восстановление только по сохранённому датасету.

## Воспроизведение

Из корня репозитория в PowerShell:

```powershell
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'

.\.venv\Scripts\python.exe scripts\xpu_lora_smoke.py --offline

Push-Location qwen_continuation_dataset
..\.venv\Scripts\python.exe generate_dataset.py `
  --config config.yaml `
  --dataset fineweb `
  --mode fixed `
  --max-examples 64 `
  --prefix-tokens 64 `
  --max-new-tokens 16 `
  --output outputs/xpu_validation.jsonl `
  --overwrite `
  --no-cycle-detection
Pop-Location

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Push-Location distillation\soft_kd_recovery
..\..\.venv\Scripts\python.exe -m src.distill `
  --config configs/validation_xpu_qwen3.5_0.8b.yaml

..\..\.venv\Scripts\python.exe -m src.distill `
  --config configs/recovery_xpu_qwen3.5_0.8b_a005.yaml
Pop-Location
```
