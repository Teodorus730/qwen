# Continued pretraining

В этой директории два независимых варианта одного эксперимента с моделью
[`stellaathena/qwen3-0.6b-sweep-ot2.0-psn316`](https://huggingface.co/stellaathena/qwen3-0.6b-sweep-ot2.0-psn316):

- [`local_rtx3090ti/`](local_rtx3090ti/README.md) — уже проведённый локальный
  batch-size / VRAM / speed benchmark на RTX 3090 Ti вместе с результатами;
- [`vast_ai_single_gpu/`](vast_ai_single_gpu/README.md) — автономный комплект
  для аренды одной GPU на Vast.ai, загрузки через WinSCP, запуска в `tmux`,
  batch-sweep и continued pretraining с автоматическими checkpoint/resume.

Готовый архив для загрузки на сервер создаётся из второй папки командой:

```powershell
.\make_upload_zip.ps1
```

Результат: `continued_pretraining/vast_ai_single_gpu_upload.zip`.
