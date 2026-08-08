# Pretraining benchmark baseline

This directory provides reproducible evaluation infrastructure for base and
pretrained checkpoints. It fixes a Qwen Base reference point so later local,
pretrained, or compressed models can be compared without storing weights in
Git.

## Core: Base/pretraining evaluation

The completed Core suite combines two kinds of evidence:

- **Intrinsic LM evaluation:** WikiText and multilingual Wikipedia BPB/PPL for
  English, Chinese, and Russian.
- **Base-model capability diagnostics:** LAMBADA OpenAI, HellaSwag, PIQA, and
  ARC-Easy. These are downstream diagnostics of a base model, not claims that
  every Core task is a pure pretraining metric or requires instruction tuning.

The canonical baseline is `Qwen/Qwen3.5-0.8B-Base` at exact revision
[`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68).
It ran on Intel Arc A770 16 GB through XPU (`xpu:0`) in BF16.

Canonical, versioned artifacts:

- lm-eval aggregate/config:
  [baseline_results/Qwen__Qwen3.5-0.8B-Base/20260807T061546Z.json](baseline_results/Qwen__Qwen3.5-0.8B-Base/20260807T061546Z.json)
  and its adjacent environment snapshot.
- Model-independent multilingual corpus manifests:
  [baseline_results/multilingual_wikipedia/manifests/](baseline_results/multilingual_wikipedia/manifests/).
- Multilingual result and environment:
  [baseline_results/multilingual_wikipedia/20260808T121142Z/](baseline_results/multilingual_wikipedia/20260808T121142Z/).
- Result index: [BASELINE_RESULTS.md](BASELINE_RESULTS.md). Russian team
  report: [CORE_BENCHMARK_RESULTS_RU.md](CORE_BENCHMARK_RESULTS_RU.md).

## Install and smoke checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r pretrain_benchmarks\requirements.txt
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --tasks hellaswag --limit 10
```

`--limit` is **only for smoke/integration checks**. A limited score is never a
canonical baseline. Raw harness output and sample logs belong to ignored
`results/`; only reviewed compact aggregates, manifests, and environment
snapshots belong under versioned `baseline_results/`.

`run_benchmark.py` runs the five lm-eval Core tasks by default: WikiText,
LAMBADA OpenAI, HellaSwag, ARC-Easy, and PIQA. It defaults to zero-shot,
seed 42, and `max_length=2048`; the effective automatic batch size is saved.
Named non-Core groups are `extended_loglikelihood`, `extended_generation`,
and `instruction_control`.

```powershell
# Full lm-eval Core protocol. Do not add --limit.
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --log-samples --write-baseline-summary
```

A versioned full baseline requires an exact model revision SHA. The compact
artifact preserves the command, model/revision, backend/device, dtype,
environment, task configuration, task versions, and aggregate metrics.

## Multilingual Wikipedia BPB/PPL

The multilingual evaluator is separate from lm-eval:

`pinned Wikipedia source` → `canonical model-independent corpus manifests` →
`model/tokenizer-specific evaluation` → `BPB/PPL`.

It pins `wikimedia/wikipedia` at
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, with `20231101.en`,
`20231101.zh`, and `20231101.ru`. The committed manifests deterministically
select about 1 MiB raw UTF-8 text per language. They fix source identity,
document order, raw byte counts, and text hashes, but deliberately contain no
model, tokenizer, chunk, or machine-specific cache path.

BPB is the primary normalized metric for comparing models within one language
on this fixed corpus. PPL is secondary and token PPL is not directly
comparable across different tokenizers. Lower Russian BPB than English BPB
does not mean a model automatically "knows Russian better": corpora and
tokenization properties differ.

Future models must reuse the committed manifests. The local selected-text
cache is ignored and only accelerates evaluation; a missing cache fails
verification rather than silently changing the corpus.

```powershell
# Verify committed corpus and local selected-text cache; no model forward pass.
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --manifest-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia\manifests --verify-manifests

# Evaluate strictly on the committed corpus. Do not use --source-scan-documents.
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --backend xpu --manifest-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia\manifests --versioned-output-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia
```

`--source-scan-documents` and capped-byte runs are smoke/probe modes only.
Their artifacts are non-canonical and must not be presented as project results.

## Reproducibility and known limitation

Versioned results preserve model SHA, Git provenance, evaluator/protocol and
manifest hashes where applicable, hardware/backend, dtype, command, and a
`pip freeze` snapshot. The multilingual result was produced after canonical
manifest commit `b88bc68` and reuses those committed manifests.

The older lm-eval baseline records task versions, task configuration,
environment, model SHA, and aggregate metrics. Its `dataset_revision` fields
are `null` and `task_hashes` is empty because lm-eval did not expose them in
that run. This limits exact future source pinning for those five datasets, but
does not invalidate the recorded baseline or require a rerun.
