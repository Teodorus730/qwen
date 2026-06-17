# Qwen continuation dataset generator

Generates text-continuation pairs with a Qwen Base model. The script takes a
prefix from an existing dataset and stores both the original continuation and
the generated continuation in JSONL.

The resulting data can be analyzed directly or used later in a distillation
experiment. Student-model training is not included in this repository.

Supported sources:

- `fineweb` — FineWeb-Edu
- `math` — FineMath-4+
- `mixed` — a reproducible mixture of both sources

## Setup

```bash
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended.

## Examples

FineWeb-Edu, fixed generation length:

```bash
python generate_dataset.py \
  --config config.yaml \
  --dataset fineweb \
  --mode fixed \
  --max-examples 20 \
  --output outputs/fineweb_fixed.jsonl \
  --overwrite
```

FineMath with entropy-based stopping:

```bash
python generate_dataset.py \
  --config config.yaml \
  --dataset math \
  --mode entropy \
  --max-examples 20 \
  --output outputs/math_entropy.jsonl \
  --overwrite
```

Mixed source, approximately 50% math:

```bash
python generate_dataset.py \
  --config config.yaml \
  --dataset mixed \
  --math-ratio 0.5 \
  --mode entropy \
  --max-examples 100 \
  --output outputs/mixed.jsonl \
  --overwrite
```

Without `--overwrite`, an existing JSONL file is resumed and already processed
`source_id` values are skipped.


## Mixed source

`--dataset mixed` reads from FineWeb-Edu and FineMath in the same run.
`--math-ratio` controls the probability of selecting FineMath for each next
document:

```text
--math-ratio 0.5  -> approximately 50% FineMath and 50% FineWeb-Edu
--math-ratio 0.8  -> approximately 80% FineMath and 20% FineWeb-Edu
```

The ratio is approximate rather than an exact quota. Short or already processed
documents can be skipped, so the final saved counts may differ slightly. The
random choice is reproducible with the configured seed. Every JSONL row stores
its source in `source_name`, so mixed output can still be analyzed separately
by dataset.

## Configuration

Main settings are stored in `config.yaml`:

```yaml
model:
  id: Qwen/Qwen3.5-0.8B-Base

generation:
  prefix_tokens: 128
  max_new_tokens: 32
  entropy_threshold: 6.5
```

`max_new_tokens` is also used as a safety limit in entropy mode.

## Output

Each JSONL row contains:

- source dataset and source ID;
- input prefix;
- original continuation;
- Qwen continuation;
- generated token count;
- per-token entropy;
- generation settings and runtime.

Generated datasets and model weights are excluded by `.gitignore`.

## Preview generated examples

Use `--preview N` to print the first `N` examples while generation is running:

```bash
python generate_dataset.py \
  --config config.yaml \
  --dataset mixed \
  --math-ratio 0.5 \
  --mode entropy \
  --max-examples 20 \
  --preview 3 \
  --output outputs/mixed_entropy.jsonl \
  --overwrite
```

For each preview, the script prints the source, the full prefix, the real
continuation, the Qwen continuation, token count, runtime, and entropy summary.

## Runtime overrides

Generation settings can be changed without editing `config.yaml`:

```bash
python generate_dataset.py \
  --config config.yaml \
  --dataset math \
  --mode entropy \
  --prefix-tokens 512 \
  --max-new-tokens 96 \
  --entropy-threshold 8.5 \
  --preview 2 \
  --output outputs/math_entropy.jsonl \
  --overwrite
```

The same length overrides work in fixed mode:

```bash
python generate_dataset.py \
  --config config.yaml \
  --dataset fineweb \
  --mode fixed \
  --prefix-tokens 256 \
  --max-new-tokens 64 \
  --preview 2 \
  --output outputs/fineweb_fixed.jsonl \
  --overwrite
```

Entropy telemetry is collected in both modes. In fixed mode it is stored for
analysis but does not affect stopping. Preview output prints the full prefix.

## Inspect a generated file

From Python or a notebook:

```python
from inspect_dataset import inspect_jsonl

rows = inspect_jsonl(
    "outputs/math_test.jsonl",
    show_examples=3,
)
```

The function prints file statistics, source distribution, generation lengths,
full prefixes, real continuations, Qwen continuations, and entropy summaries.
It also returns the loaded rows.

The same helper can be run from the command line:

```bash
python inspect_dataset.py outputs/math_test.jsonl --show-examples 3
```

## Repository layout

This directory is intended to be placed at the repository root:

```text
qwen/
└── qwen_continuation_dataset/
```

The included Colab notebook clones the `main` branch and opens this root-level
directory.

