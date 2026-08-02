# Continued pretraining

В этой директории находятся локальный и Vast.ai варианты эксперимента с моделью
[`stellaathena/qwen3-0.6b-sweep-ot2.0-psn316`](https://huggingface.co/stellaathena/qwen3-0.6b-sweep-ot2.0-psn316):

- [`local_rtx3090ti/`](local_rtx3090ti/README.md) — уже проведённый локальный
  batch-size / VRAM / speed benchmark на RTX 3090 Ti вместе с результатами;
- [`vast_ai_single_gpu/`](vast_ai_single_gpu/README.md) — завершённые эксперименты
  на одной арендованной GPU Vast.ai: RTX 5070 Ti 16 GB и RTX 5090 32 GB, включая
  [сравнительный отчёт по RTX 5090](vast_ai_single_gpu/rtx_5090_32gb/REPORT.md).

Готовый архив для RTX 5090 создаётся командой:

```powershell
cd D:\HANDMADE_LLM\REPO\qwen\continued_pretraining\vast_ai_single_gpu\rtx_5090_32gb
.\make_upload_zip.ps1
```

Результат:
`continued_pretraining/vast_ai_single_gpu/rtx_5090_32gb/rtx_5090_32gb_upload.zip`.
