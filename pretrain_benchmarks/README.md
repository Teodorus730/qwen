# Pretraining benchmark baseline

This directory records a small, repeatable baseline for the untouched
`Qwen/Qwen3.5-0.8B-Base` model. The same protocol can later compare local
pretraining checkpoints without placing model weights in Git.

The suite uses `lm-evaluation-harness` with zero-shot evaluation (seed 42):
WikiText, LAMBADA OpenAI, HellaSwag, ARC-Easy, and PIQA. WikiText reports
language-model perplexity; the other tasks report accuracy.

## Install and run

```powershell
.\.venv\Scripts\python.exe -m pip install -r pretrain_benchmarks\requirements.txt
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --tasks hellaswag --limit 10
```

The runner selects CUDA when it is available, otherwise Intel XPU, and uses
CPU only as a diagnostic fallback. It chooses BF16 when the selected
accelerator reports support, otherwise FP16; CPU uses FP32. Override backend
only for diagnosis, for example `--backend xpu`.

`--limit` is **only for smoke/integration runs**. Never use a limited score as
a baseline result. After a successful smoke run, the full suite is:

```powershell
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --log-samples --write-baseline-summary
```

Each run is written to `pretrain_benchmarks/results/<model>/<run-id>/`. It
contains the harness output and `metadata.json` with the model/revision,
backend/device, accelerator, dtype, package versions, timestamp, tasks and
evaluation arguments. Results are ignored by Git. A smoke result is only an
integration check, not a final benchmark score.

The full-run flag `--write-baseline-summary` writes a compact JSON under
`baseline_results/<model>/<run-id>.json`; this directory is deliberately not
ignored, so a reviewed final baseline can be versioned in Git. Maintain the
human-readable index in `BASELINE_RESULTS.md`. Raw harness JSON and sample
logs remain under ignored `results/`.

The versioned JSON includes the full small aggregate/config result from
lm-eval (all metrics, task versions, effective batch sizes and task config),
the runner metadata and exact command. Its adjacent `.environment.txt` is a
`pip freeze` snapshot. Dataset paths and a configured dataset revision are
included when lm-eval exposes them; Hugging Face dataset fingerprints are not
available in lm-eval's result JSON and are therefore not claimed.

A versioned full baseline requires that the Hugging Face model revision SHA
can be resolved before evaluation starts. Smoke runs may continue without it,
but are not versioned baselines.

The notebook `pretrain_benchmarks_fixed.ipynb` is a convenient interactive
entry point; it delegates to the same runner so notebook and command-line
runs share one protocol.
