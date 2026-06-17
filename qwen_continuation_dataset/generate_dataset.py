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
from qwen_continuation.io_utils import JsonlWriter, load_completed_ids
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without downloading model or dataset.",
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
    else:
        print("Warning: CUDA is unavailable; generation will be very slow.")
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
    )

    print_environment(config)

    if args.dry_run:
        print("Dry run completed: configuration is valid.")
        return

    if not torch.cuda.is_available():
        answer = input(
            "CUDA is unavailable. Continue on CPU? This may be very slow. [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Stopped. Enable a GPU and run again.")

    output_path = Path(config["output"]["path"])
    resume = bool(config["output"].get("resume", True))

    if args.overwrite or not resume:
        if output_path.exists():
            output_path.unlink()
            print(f"Removed existing output: {output_path}")
        completed_ids: set[str] = set()
    else:
        completed_ids = load_completed_ids(output_path)
        if completed_ids:
            print(
                f"Resume enabled: found {len(completed_ids)} completed "
                f"source IDs in {output_path}"
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

    with JsonlWriter(output_path, flush_every=flush_every) as writer:
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
    print("Output:", output_path.resolve())


if __name__ == "__main__":
    main()
