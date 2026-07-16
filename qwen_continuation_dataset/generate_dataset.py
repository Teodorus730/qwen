from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from qwen_continuation.config import load_config, with_overrides
from qwen_continuation.data import stream_documents
from qwen_continuation.generation import generate_continuation
from qwen_continuation.io_utils import (
    HfShardWriter,
    JsonlWriter,
    load_completed_ids,
    load_completed_ids_from_dir,
)
from qwen_continuation.model import load_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate continuation pairs with a Qwen Base model."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--output",
        help=(
            "Output JSONL path. This only selects the file; "
            "existing data is resumed by default."
        ),
    )
    parser.add_argument("--mode", choices=("fixed", "entropy"))
    parser.add_argument(
        "--dataset",
        dest="dataset_source",
        choices=("fineweb", "math", "mixed"),
        help="Select FineWeb-Edu, FineMath, or a weighted mixture.",
    )
    parser.add_argument(
        "--math-ratio",
        type=float,
        help=(
            "Math fraction for --dataset mixed, from 0.0 to 1.0. "
            "Example: 0.7 means roughly 70% FineMath."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output file before generation instead of resuming it.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N generated examples during the run.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        help="Override generation.prefix_tokens from config.yaml.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override generation.max_new_tokens from config.yaml.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        help="Override generation.entropy_threshold from config.yaml.",
    )
    cycle_group = parser.add_mutually_exclusive_group()
    cycle_group.add_argument(
        "--cycle-detection",
        dest="cycle_enabled",
        action="store_true",
        default=None,
        help="Force-enable n-gram cycle detection for this run.",
    )
    cycle_group.add_argument(
        "--no-cycle-detection",
        dest="cycle_enabled",
        action="store_false",
        default=None,
        help="Force-disable n-gram cycle detection for this run.",
    )
    parser.add_argument(
        "--cycle-window-chars",
        type=int,
        help="Override generation.cycle_detection.window_chars from config.yaml.",
    )
    parser.add_argument(
        "--cycle-ngram-chars",
        type=int,
        help="Override generation.cycle_detection.ngram_chars from config.yaml.",
    )
    parser.add_argument(
        "--cycle-min-chars",
        type=int,
        help="Override generation.cycle_detection.min_chars from config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without downloading model or dataset.",
    )
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help="Enable incremental upload to a Hugging Face dataset repo (overrides huggingface.enabled).",
    )
    parser.add_argument(
        "--hf-repo-id",
        help="Target Hugging Face dataset repo, e.g. username/dataset-name.",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token. Prefer 'huggingface-cli login' or the HF_TOKEN env var over passing this on the command line.",
    )
    parser.add_argument(
        "--hf-shard-size",
        type=int,
        help="Rows per uploaded shard.",
    )
    return parser.parse_args()


def print_environment(config: dict[str, Any]) -> None:
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
            "GiB",
        )
    elif torch.xpu.is_available():
        print("XPU available:", True)
        print("GPU:", torch.xpu.get_device_name(0))
    else:
        print("Warning: GPU is unavailable; generation will be very slow.")
    print("Teacher:", config["model"]["id"])
    dataset_config = config["dataset"]
    selected_source = dataset_config.get("source", "fineweb")
    print("Dataset mode:", selected_source)
    if selected_source == "mixed":
        print(
            "Math ratio:",
            float(dataset_config.get("mixed", {}).get("math_ratio", 0.5)),
        )
        print("FineWeb dataset:", dataset_config["sources"]["fineweb"]["id"])
        print("Math dataset:", dataset_config["sources"]["math"]["id"])
    else:
        selected_config = dataset_config["sources"][selected_source]
        print("Dataset:", selected_config["id"])
        print("Dataset subset:", selected_config.get("subset"))
    print("Mode:", config["generation"]["mode"])
    print("Max examples:", config["dataset"]["max_examples"])
    print("Output:", config["output"]["path"])

    cycle_cfg = config["generation"].get("cycle_detection", {})
    if cycle_cfg.get("enabled", False):
        print("Cycle detection: enabled")
        print("  window_chars:", cycle_cfg.get("window_chars", 100))
        print("  ngram_chars: ", cycle_cfg.get("ngram_chars", 20))
        print("  min_chars:   ", cycle_cfg.get("min_chars", 50))
    else:
        print("Cycle detection: disabled")

    hf_cfg = config.get("huggingface", {})
    if hf_cfg.get("enabled", False):
        print("HF upload:   enabled")
        print("HF repo:    ", hf_cfg.get("repo_id", "(not set)"))
        print("Shard size: ", hf_cfg.get("shard_size", 10000))
    else:
        print("HF upload:   disabled")


def build_record(
    *,
    document: dict[str, Any],
    prefix_ids: list[int],
    real_continuation_ids: list[int],
    generated_ids: list[int],
    entropies: list[float],
    tokenizer: Any,
    config: dict[str, Any],
    dtype_name: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    generation = config["generation"]

    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    real_continuation = tokenizer.decode(
        real_continuation_ids, skip_special_tokens=True
    )
    teacher_continuation = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    synthetic_text = tokenizer.decode(
        prefix_ids + generated_ids, skip_special_tokens=True
    )

    return {
        "source_id": document["source_id"],
        "source_name": document["source_name"],
        "source_dataset": document["source_dataset"],
        "source_subset": document.get("source_subset"),
        "source_metadata": document.get("metadata", {}),
        "prefix_text": prefix_text,
        "real_continuation": real_continuation,
        "teacher_continuation": teacher_continuation,
        "synthetic_text": synthetic_text,
        "prefix_token_count": len(prefix_ids),
        "real_continuation_token_count": len(real_continuation_ids),
        "generated_token_count": len(generated_ids),
        "teacher_model": config["model"]["id"],
        "teacher_dtype": dtype_name,
        "generation_seconds": round(elapsed_seconds, 4),
        "generation": {
            "mode": generation["mode"],
            "temperature": float(generation.get("temperature", 0.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "top_k": int(generation.get("top_k", 0)),
            "max_new_tokens": int(generation["max_new_tokens"]),
            "entropy_threshold": (
                float(generation["entropy_threshold"])
                if generation["mode"] == "entropy"
                else None
            ),
            "token_entropies": [round(value, 6) for value in entropies],
        },
    }


def print_preview(record: dict[str, Any], index: int) -> None:
    entropies = record.get("generation", {}).get("token_entropies", [])
    entropy_mean = (
        sum(entropies) / len(entropies)
        if entropies
        else None
    )
    entropy_max = max(entropies) if entropies else None

    lines = [
        "",
        "=" * 88,
        f"PREVIEW {index}",
        f"Source: {record.get('source_name')}",
        f"Dataset: {record.get('source_dataset')}",
        f"Generated tokens: {record.get('generated_token_count')}",
        f"Generation seconds: {record.get('generation_seconds')}",
    ]

    if entropy_mean is not None:
        lines.append(f"Entropy mean: {entropy_mean:.4f}")
        lines.append(f"Entropy max: {entropy_max:.4f}")

    lines.extend(
        [
            "",
            "PREFIX:",
            record.get("prefix_text", ""),
            "",
            "REAL CONTINUATION:",
            record.get("real_continuation", ""),
            "",
            "QWEN CONTINUATION:",
            record.get("teacher_continuation", ""),
            "=" * 88,
            "",
        ]
    )
    tqdm.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = with_overrides(
        config,
        max_examples=args.max_examples,
        output_path=args.output,
        mode=args.mode,
        dataset_source=args.dataset_source,
        math_ratio=args.math_ratio,
        prefix_tokens=args.prefix_tokens,
        max_new_tokens=args.max_new_tokens,
        entropy_threshold=args.entropy_threshold,
        cycle_enabled=args.cycle_enabled,
        cycle_window_chars=args.cycle_window_chars,
        cycle_ngram_chars=args.cycle_ngram_chars,
        cycle_min_chars=args.cycle_min_chars,
        hf_upload=args.hf_upload or None,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        hf_shard_size=args.hf_shard_size,
    )

    print_environment(config)

    if args.dry_run:
        print("Dry run completed: configuration is valid.")
        return

    if not (torch.cuda.is_available() or torch.xpu.is_available()):
        answer = input(
            "GPU is unavailable. Continue on CPU? This may be very slow. [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Stopped. Enable a GPU and run again.")

    output_path = Path(config["output"]["path"])
    resume = bool(config["output"].get("resume", True))
    hf_cfg = config.get("huggingface", {})
    hf_enabled = hf_cfg.get("enabled", False)

    if args.overwrite or not resume:
        if hf_enabled:
            removed = 0
            for shard in output_path.parent.glob("train-*.jsonl"):
                shard.unlink()
                removed += 1
            state_file = output_path.parent / "state.json"
            if state_file.exists():
                state_file.unlink()
            if removed:
                print(f"Removed {removed} shard(s) from: {output_path.parent}")
        else:
            if output_path.exists():
                output_path.unlink()
                print(f"Removed existing output: {output_path}")
        completed_ids: set[str] = set()
    else:
        if hf_enabled:
            completed_ids = load_completed_ids_from_dir(output_path.parent)
        else:
            completed_ids = load_completed_ids(output_path)
        if completed_ids:
            print(
                f"Resume enabled: found {len(completed_ids)} completed "
                f"source IDs"
            )

    teacher = load_teacher(config)
    tokenizer = teacher.tokenizer

    prefix_tokens = int(config["generation"]["prefix_tokens"])
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    required_tokens = prefix_tokens + max_new_tokens
    max_examples = int(config["dataset"]["max_examples"])
    flush_every = int(config["output"].get("flush_every", 1))

    saved = 0
    skipped_short = 0
    skipped_completed = 0
    source_counts: dict[str, int] = {"fineweb": 0, "math": 0}
    started = time.perf_counter()

    progress = tqdm(total=max_examples, desc="Saved examples")

    if hf_enabled:
        writer_cm = HfShardWriter(
            output_dir=output_path.parent,
            repo_id=hf_cfg["repo_id"],
            shard_size=hf_cfg.get("shard_size", 10000),
            token=hf_cfg.get("token"),
            flush_every=flush_every,
        )
    else:
        writer_cm = JsonlWriter(output_path, flush_every=flush_every)

    with writer_cm as writer:
        for document in stream_documents(config):
            if saved >= max_examples:
                break

            if document["source_id"] in completed_ids:
                skipped_completed += 1
                continue

            token_ids = tokenizer.encode(
                document["text"],
                add_special_tokens=False,
            )

            if len(token_ids) < required_tokens:
                skipped_short += 1
                continue

            prefix_ids = token_ids[:prefix_tokens]
            real_continuation_ids = token_ids[
                prefix_tokens:prefix_tokens + max_new_tokens
            ]

            generation_started = time.perf_counter()
            generated_ids, entropies = generate_continuation(
                teacher,
                prefix_ids,
                config,
            )
            generation_elapsed = time.perf_counter() - generation_started

            record = build_record(
                document=document,
                prefix_ids=prefix_ids,
                real_continuation_ids=real_continuation_ids,
                generated_ids=generated_ids,
                entropies=entropies,
                tokenizer=tokenizer,
                config=config,
                dtype_name=teacher.dtype_name,
                elapsed_seconds=generation_elapsed,
            )
            writer.write(record)
            completed_ids.add(document["source_id"])
            saved += 1

            if args.preview > 0 and saved <= args.preview:
                print_preview(record, saved)

            source_name = document["source_name"]
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            progress.update(1)
            progress.set_postfix(
                {
                    "source": source_name,
                    "generated": len(generated_ids),
                    "sec": round(generation_elapsed, 2),
                }
            )

    progress.close()
    total_elapsed = time.perf_counter() - started

    print()
    print("Done.")
    print("Saved:", saved)
    print("Saved by source:", source_counts)
    print("Skipped short documents:", skipped_short)
    print("Skipped completed documents:", skipped_completed)
    print("Elapsed seconds:", round(total_elapsed, 2))
    if hf_enabled:
        print("HF repo:", hf_cfg["repo_id"])
    else:
        print("Output:", output_path.resolve())


if __name__ == "__main__":
    main()
