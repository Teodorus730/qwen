# Pretraining benchmark baseline

This directory records a small, repeatable baseline for the untouched
`Qwen/Qwen3.5-0.8B-Base` model. The same protocol can later compare local
pretraining checkpoints without placing model weights in Git.

Core evaluation uses zero-shot lm-evaluation-harness tasks (seed 42): WikiText,
LAMBADA OpenAI, HellaSwag, ARC-Easy, and PIQA. WikiText reports language-model
perplexity; the other tasks report accuracy. The core language-model category
also includes the separate, deterministic multilingual Wikipedia BPB/PPL
protocol for English, Chinese and Russian.

Benchmark categories are deliberately small: **Core** is WikiText plus
multilingual Wikipedia BPB/PPL (EN/ZH/RU), LAMBADA, HellaSwag, ARC-Easy and
PIQA; **multilingual knowledge** is C-Eval/MMMLU; **diagnostics** are MMLU
Redux, MMLU-Pro and IFEval. SuperGPQA is planned and is not currently run.

## Install and run

```powershell
.\.venv\Scripts\python.exe -m pip install -r pretrain_benchmarks\requirements.txt
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --tasks hellaswag --limit 10
```

Named suites are `core`, `extended_loglikelihood`, `extended_generation`, and
`instruction_control`. `core` preserves the original zero-shot, 2048-token
protocol. Other suites preserve the native lm-eval few-shot and context
settings unless `--num-fewshot` or `--max-length` is explicitly provided.
Use `--tasks` for a constituent task or group ID when making a diagnostic run.

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

## Multilingual Wikipedia language model

`multilingual_lm_eval.py` is intentionally independent of lm-eval. It pins
the official `wikimedia/wikipedia` repository to commit
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, using the `20231101.en`,
`20231101.zh` and `20231101.ru` configurations. Canonical selection ranks all
articles by `SHA-256(dataset revision, language, article id, seed)` and uses
the ranked documents until the same raw UTF-8 byte budget is reached for each
language. It records article IDs/titles, text hashes, byte/character counts,
and a corpus manifest hash; selected text remains in ignored local cache. The
canonical corpus manifest is tokenizer- and model-independent: it contains no
token counts, chunk hashes or model settings.

The versioned [protocol configuration](multilingual_wikipedia_protocol.json)
sets the first full-run budget to 1 MiB raw UTF-8 per language, seed 42, a
512-document ranked candidate pool and 2048-token windows. The pool is the
globally lowest ranks from the complete pinned snapshot, so it is independent
of streaming order while bounding manifest-building memory. If it cannot fill
the budget, increase the recorded pool size and regenerate all languages.

The scorer starts each document with BOS (or EOS when, as with Qwen, no BOS is
defined), scores every non-special document token exactly once, uses the
previous token as context at each 2048-token window boundary, and counts UTF-8
bytes only for scored target spans. It
reports `PPL=exp(total_nll/scored_tokens)` and
`BPB=total_nll/(ln(2)*scored_target_utf8_bytes)`.

BPB is the primary normalized LM metric for comparing model changes within
one language. Do not interpret absolute BPB values across EN, ZH and RU as a
direct ranking of which language the model knows better: the underlying
corpora differ.

For a smoke-only integration check, use a capped candidate scan; its manifest
is explicitly non-canonical and cannot be written as a versioned result:

```powershell
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --backend xpu --byte-budget 16384 --source-scan-documents 128
```

For a reviewed full run, omit `--source-scan-documents` and supply
`--versioned-output-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia`.
That directory will contain compact per-language manifests, aggregate result,
environment snapshot, exact model revision, model/tokenizer configuration,
hardware/backend, evaluator hash, and Git SHA/branch/dirty state.

Canonical manifests are generated once and committed before any model result.
They retain an ignored local selected-text cache; evaluation with
`--manifest-dir` never opens or scans Wikipedia and refuses a missing cache or
any corpus manifest SHA, dataset/config/revision, language, source/selection
protocol or text-hash mismatch. The evaluator then writes tokenizer identity,
max length/stride, token counts, derived chunk hashes, model/hardware and
BPB/PPL accounting into that run's compact `result.json`. The intended
sequence is: commit the evaluator/protocol; generate and inspect/commit the
EN/ZH/RU manifests; run each model strictly with those committed manifests;
then commit only its compact result. Thus later checkpoints do not rescan
Wikipedia, and the same raw corpus can be scored with another tokenizer.

```powershell
# One time: this scans the pinned snapshots and writes commit-ready manifests.
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --backend xpu --build-manifests-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia\manifests

# Before every evaluation: no Wikipedia scan and no model forward pass.
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --backend xpu --manifest-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia\manifests --verify-manifests

# Full model evaluation: strictly reuses the committed manifests.
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --backend xpu --manifest-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia\manifests --versioned-output-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia
```
