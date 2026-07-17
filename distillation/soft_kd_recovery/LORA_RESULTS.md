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
Pop-Location
```
