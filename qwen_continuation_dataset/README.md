# Qwen continuation dataset generator

Generates text-continuation pairs with a Qwen Base model. The script takes a
prefix from an existing dataset and stores both the original continuation and
the generated continuation in JSONL.

The resulting data can be analyzed directly or used later in a distillation
experiment. Student-model training is not included in this repository.

## Setup

```bash
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended.

## Quick start

```bash
python generate_dataset.py \
  --config config.yaml \
  --dataset fineweb \
  --mode fixed \
  --max-examples 20 \
  --output outputs/fineweb_fixed.jsonl \
  --overwrite
```

Without `--overwrite`, an existing output file (or set of shards, if
Hugging Face upload is on) is resumed, and already processed `source_id`
values are skipped.

## Parameters

Every setting lives in `config.yaml`. Most also have a matching CLI flag that
overrides it for a single run without editing the file. `--config` picks
which YAML file to load; it defaults to `config.yaml`.

### model

| Key | Default | CLI flag | Description |
|---|---|---|---|
| `model.id` | `Qwen/Qwen3.5-0.8B-Base` | — | Hugging Face model id used as the teacher. |
| `model.trust_remote_code` | `true` | — | Passed to `from_pretrained`. |
| `model.device_map` | `auto` | — | Passed to `from_pretrained`. |

### dataset

| Key | Default | CLI flag | Description |
|---|---|---|---|
| `dataset.source` | `fineweb` | `--dataset` | `fineweb`, `math`, or `mixed`. |
| `dataset.max_examples` | `20` | `--max-examples` | How many examples to generate and save. |
| `dataset.mixed.math_ratio` | `0.5` | `--math-ratio` | Fraction of FineMath documents when `source: mixed`. See [Mixed source](#mixed-source). |
| `dataset.sources.fineweb.*` | see `config.yaml` | — | FineWeb-Edu connection: `id`, `subset`, `split`, `streaming`, `text_column`, `id_column`, `shuffle`, `shuffle_buffer_size`. |
| `dataset.sources.math.*` | see `config.yaml` | — | Same fields for FineMath. |

### generation

| Key | Default | CLI flag | Description |
|---|---|---|---|
| `generation.mode` | `fixed` | `--mode` | `fixed` generates exactly `max_new_tokens`; `entropy` stops early on high uncertainty. |
| `generation.prefix_tokens` | `128` | `--prefix-tokens` | Tokens taken from the document as the input prefix. |
| `generation.max_new_tokens` | `32` | `--max-new-tokens` | Tokens to generate in fixed mode; hard cap in entropy mode. |
| `generation.temperature` | `0.0` | — | `0` is greedy decoding; higher values sample. |
| `generation.top_p` | `1.0` | — | Nucleus sampling threshold. |
| `generation.top_k` | `0` | — | Top-k sampling limit; `0` disables it. |
| `generation.entropy_threshold` | `6.5` | `--entropy-threshold` | Entropy mode only — stop once next-token entropy exceeds this. See [Generation modes](#generation-modes). |
| `generation.min_generated_tokens_before_entropy_stop` | `1` | — | Entropy mode only — tokens generated before the entropy stop can trigger. |
| `generation.seed` | `42` | — | RNG seed. |
| `generation.cycle_detection.enabled` | `true` | `--cycle-detection` / `--no-cycle-detection` | Stop generation early on a repeated n-gram. See [Cycle detection](#cycle-detection). |
| `generation.cycle_detection.window_chars` | `100` | `--cycle-window-chars` | Width of the character window checked for a repeat. |
| `generation.cycle_detection.ngram_chars` | `20` | `--cycle-ngram-chars` | Length of the tail compared against the window. |
| `generation.cycle_detection.min_chars` | `50` | `--cycle-min-chars` | Minimum generated length before checking starts. |

### output

| Key | Default | CLI flag | Description |
|---|---|---|---|
| `output.path` | `outputs/generated.jsonl` | `--output` | Local JSONL path, or shard directory when Hugging Face upload is on. |
| `output.resume` | `true` | `--overwrite` (inverts it for one run) | Resume from existing output instead of starting over. |
| `output.flush_every` | `1` | — | Rows written between disk flushes. |

### huggingface

| Key | Default | CLI flag | Description |
|---|---|---|---|
| `huggingface.enabled` | `false` | `--hf-upload` | Upload completed shards to a Hugging Face dataset repo as they're generated. See [Hugging Face upload](#hugging-face-upload). |
| `huggingface.repo_id` | `your_username/qwen_continuation_dataset` | `--hf-repo-id` | Target dataset repo. |
| `huggingface.token` | `null` | `--hf-token` | Prefer the `HF_TOKEN` env var or `huggingface-cli login` instead of passing this on the command line. |
| `huggingface.shard_size` | `10000` | `--hf-shard-size` | Examples per shard before it's uploaded. |

### CLI-only flags

These have no `config.yaml` equivalent:

| Flag | Description |
|---|---|
| `--config PATH` | YAML config file to load. Default: `config.yaml`. |
| `--overwrite` | Delete existing output before starting instead of resuming it. |
| `--preview N` | Print the first `N` generated examples while running. See [Preview generated examples](#preview-generated-examples). |
| `--dry-run` | Validate configuration and exit, without loading the model or dataset. |

## Generation modes

`fixed` always generates `max_new_tokens` tokens (subject to cycle
detection). `entropy` generates token by token and stops as soon as the
next-token entropy exceeds `entropy_threshold`, with `max_new_tokens` as a
hard cap either way. Both modes record per-token entropy in the output
regardless of which one is active.

## Mixed source

`--dataset mixed` reads from FineWeb-Edu and FineMath in the same run.
`--math-ratio` controls the probability of selecting FineMath for each next
document:

```text
--math-ratio 0.5  -> approximately 50% FineMath and 50% FineWeb-Edu
--math-ratio 0.8  -> approximately 80% FineMath and 20% FineWeb-Edu
```

The ratio is approximate rather than an exact quota. Short or already
processed documents can be skipped, so the final saved counts may differ
slightly. The random choice is reproducible with the configured seed. Every
JSONL row stores its source in `source_name`, so mixed output can still be
analyzed separately by dataset.

## Cycle detection

Generation stops early when the output contains a repeated n-gram. The check
compares the last `ngram_chars` characters against all positions in the
preceding `window_chars` character window and stops if a match is found.
`min_chars` sets the minimum generated length before the check starts.

`window_chars` must be greater than `ngram_chars`, and `ngram_chars` must be
positive; the script raises an error at startup otherwise instead of
silently running with detection disabled. A repeat is only caught if its
period fits within `window_chars - ngram_chars` characters; longer-period
repeats fall outside the window by design.

```bash
python generate_dataset.py \
  --config config.yaml \
  --cycle-detection \
  --cycle-window-chars 150 \
  --cycle-ngram-chars 30 \
  --cycle-min-chars 60 \
  --dataset fineweb \
  --mode fixed \
  --max-examples 20 \
  --output outputs/test.jsonl \
  --overwrite
```

Use `--no-cycle-detection` to turn it off for a single run instead.

## Hugging Face upload

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx

python generate_dataset.py \
  --config config.yaml \
  --hf-upload \
  --hf-repo-id your_username/qwen_continuation_dataset \
  --hf-shard-size 10000 \
  --dataset mixed \
  --math-ratio 0.5 \
  --mode entropy \
  --max-examples 50000 \
  --output outputs/train.jsonl
```

The repository is created automatically if it does not exist. Completed
shards are uploaded as `data/train-00000.jsonl`, `data/train-00001.jsonl`,
and so on. Each upload also refreshes the dataset card with current
statistics.

Progress is saved to `outputs/state.json` after each shard. On restart the
script reads `state.json` (or scans the hub for existing shards if the file
is missing) and continues from where it stopped. If the process was
interrupted mid-shard, the partial local file is detected and appended to
rather than overwritten. Already uploaded shards are never modified.

Rows are flushed to disk every `output.flush_every` writes (same setting used
by the plain JSONL path), so a hard crash loses at most that many unflushed
rows rather than a full buffer's worth. If a crash happens between a
successful upload and the `state.json` write for that shard, the next run
detects the shard is already full, re-uploads it once to be safe, and moves
on to the next shard instead of appending to it. If `state.json` is missing
entirely, the exact row count of the most recently uploaded shard is fetched
from the hub rather than assumed, since it may be a short final shard from an
earlier interrupted run.

## Output fields

Each JSONL row contains:

| Field | Description |
|---|---|
| `source_id` | ID of the source document. |
| `source_name` | `fineweb` or `math`. |
| `source_dataset`, `source_subset`, `source_metadata` | Source dataset details. |
| `prefix_text` | Input prefix. |
| `real_continuation` | Original continuation from the source document. |
| `teacher_continuation` | Continuation generated by the model. |
| `synthetic_text` | `prefix_text` + `teacher_continuation`. |
| `prefix_token_count`, `real_continuation_token_count`, `generated_token_count` | Token counts. |
| `teacher_model`, `teacher_dtype` | Model identity and precision used. |
| `generation_seconds` | Wall-clock time for this example. |
| `generation` | Mode, sampling settings, and per-token entropy for this example. |

## Preview generated examples

`--preview N` prints the first `N` examples while generation is running: the
source, the full prefix, the real continuation, the Qwen continuation, token
count, runtime, and entropy summary.

## Inspect a generated file

From Python or a notebook:

```python
from inspect_dataset import inspect_jsonl

rows = inspect_jsonl(
    "outputs/math_test.jsonl",
    show_examples=3,
)
```

The function prints file statistics, source distribution, generation
lengths, full prefixes, real continuations, Qwen continuations, and entropy
summaries. It also returns the loaded rows.

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

The included Colab/Kaggle notebook detects the environment, clones the
`main` branch, and opens this root-level directory automatically.
