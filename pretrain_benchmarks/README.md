# Pretraining benchmark infrastructure

`pretrain_benchmarks` provides reproducible evaluation for base and pretrained
language-model checkpoints. It keeps evaluation code, pinned task definitions,
canonical corpus/sample manifests, compact results, and environment snapshots
together so future checkpoints can be compared against the fixed baseline
under the same evaluation protocol. Model weights and heavy per-sample outputs
are not stored in Git.

## Scope and status

| Direction | Benchmarks | Status |
|---|---|---|
| **Core: Base/pretraining evaluation** | WikiText, multilingual Wikipedia EN/ZH/RU, LAMBADA OpenAI, HellaSwag, PIQA, ARC-Easy | Completed |
| **Multilingual / knowledge** | C-Eval validation, full MMMLU | Completed |
| **Extended diagnostics** | MMLU-Redux generative, MMLU-Pro, IFEval | Runner suites exist; canonical baselines not run |
| **Deferred** | SuperGPQA | Not integrated or run |

WikiText and multilingual Wikipedia are intrinsic language-model evaluations.
The remaining completed tasks are zero-shot diagnostics of base-model
capabilities; they are not all pure pretraining metrics.

The versioned reference is `Qwen/Qwen3.5-0.8B-Base` at exact revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, evaluated on Intel Arc A770
16 GB through XPU in BF16.

## Where to start

- [BASELINE_RESULTS.md](BASELINE_RESULTS.md) is the technical source of truth:
  exact metrics, protocols, revisions, hashes, provenance, artifacts, and
  caveats.
- [BASELINE_RESULTS_RU.md](BASELINE_RESULTS_RU.md) is the concise Russian
  report for the team and project reviewers.
- [`baseline_results/`](baseline_results/) contains reviewed compact results,
  environment snapshots, and canonical manifests.
- [`lm_eval_tasks/`](lm_eval_tasks/) contains the pinned external C-Eval and
  MMMLU task definitions.
- `results/`, `.hf_cache/`, and the multilingual selected-text cache are local,
  ignored runtime data.

## Setup

Run commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r pretrain_benchmarks\requirements.txt
```

`run_benchmark.py` is the lm-eval entry point. It selects the requested
backend, resolves and records the exact model revision when available, writes
run metadata, and keeps raw harness output under ignored `results/`.

## Common workflows

### Smoke check

```powershell
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --tasks hellaswag --backend xpu --dtype bfloat16 --batch-size 8 --limit 10
```

`--limit` is only for smoke and integration checks. Limited scores are never
canonical baseline results.

### Full Core run

```powershell
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --suite core --backend xpu --dtype bfloat16 --log-samples --write-baseline-summary
```

The Core suite defaults to zero-shot, seed 42, and `max_length=2048`. Do not
add `--limit` to a canonical run.

### C-Eval validation

```powershell
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --tasks ceval-valid --include-path pretrain_benchmarks\lm_eval_tasks --backend xpu --dtype bfloat16 --batch-size 8 --num-fewshot 0 --log-samples --write-baseline-summary
```

The committed task definition pins `ceval/ceval-exam`; `ceval-valid` evaluates
the public validation split, not the closed official test split.

### Multilingual Wikipedia

Canonical manifests are model/tokenizer-independent and fix the exact source
documents, order, UTF-8 byte counts, and text hashes for EN, ZH, and RU. The
ignored local text cache is only a runtime accelerator and is not part of a
manifest hash.

```powershell
# Verify committed manifests and the local selected-text cache; no model pass.
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --manifest-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia\manifests --verify-manifests

# Evaluate strictly on those manifests.
.\.venv\Scripts\python.exe pretrain_benchmarks\multilingual_lm_eval.py --backend xpu --dtype bfloat16 --manifest-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia\manifests --versioned-output-dir pretrain_benchmarks\baseline_results\multilingual_wikipedia
```

The protocol is:

`pinned Wikipedia source` → `canonical raw-text manifests` →
`model/tokenizer-specific tokenization and scoring` → `BPB/PPL`.

BPB is the primary metric for comparing models within one language on the
same corpus. Token PPL is tokenizer-dependent, and absolute BPB values across
different languages are not a ranking of language knowledge.

`--source-scan-documents` and capped-source runs are smoke/probe modes only.
Future models must reuse the committed manifests rather than rebuild a corpus.

### Full MMMLU

MMMLU uses committed deterministic selections whose union is the full pinned
`openai/MMMLU` test set. Verify the progressive manifest without a model pass:

```powershell
.\.venv\Scripts\python.exe pretrain_benchmarks\prepare_mmmlu_progressive.py --output-dir pretrain_benchmarks\baseline_results\mmmlu\manifests --verify pretrain_benchmarks\baseline_results\mmmlu\manifests\mmmlu_progressive_manifest.json
```

Evaluate both complementary selections with the reference lm-eval path:

```powershell
# Stage 1: 9,828 samples.
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --tasks mmmlu --include-path pretrain_benchmarks\lm_eval_tasks --samples pretrain_benchmarks\baseline_results\mmmlu\manifests\mmmlu_stage1_samples.json --backend xpu --dtype bfloat16 --batch-size 4 --num-fewshot 0 --max-length 2048 --write-stage-summary pretrain_benchmarks\results\mmmlu_stage1_summary.json

# Stage 2: exact complement, 186,760 samples.
.\.venv\Scripts\python.exe pretrain_benchmarks\run_benchmark.py --tasks mmmlu --include-path pretrain_benchmarks\lm_eval_tasks --samples pretrain_benchmarks\baseline_results\mmmlu\manifests\mmmlu_stage2_samples.json --backend xpu --dtype bfloat16 --batch-size 2 --num-fewshot 0 --max-length 4096 --write-stage-summary pretrain_benchmarks\results\mmmlu_stage2_summary.json
```

There is no extra `--limit`, automatic batching, or experimental optimized
evaluator in the canonical path. Full metrics are reconstructed by summing
the exact integer counters from the two non-overlapping stage summaries, not
by averaging their floating-point accuracies.

## Reproducibility and artifact policy

Canonical compact artifacts preserve, as applicable:

- exact model and dataset revisions;
- task versions/configuration, zero/few-shot setting, seed, device, dtype,
  batch policy, context length, and full command;
- Git provenance, manifest/selection hashes, aggregate metrics, and mergeable
  integer counters;
- environment snapshots including the relevant Python package versions.

Heavy raw/sample logs remain ignored. Review compact outputs before copying
them into `baseline_results/`, and commit infrastructure/manifests separately
from benchmark results.

The older Core lm-eval artifact has `dataset_revision=null` and empty
`task_hashes` because those values were not exposed in that run. Its task
versions, configuration, environment, model SHA, and metrics are preserved;
the limitation is documented in [BASELINE_RESULTS.md](BASELINE_RESULTS.md).
