from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {file_path}"
                ) from exc

    return rows


def inspect_jsonl(
    path: str | Path,
    show_examples: int = 3,
) -> list[dict[str, Any]]:
    file_path = Path(path)
    rows = load_jsonl(file_path)

    print("=" * 100)
    print("File:", file_path)
    print("Rows:", len(rows))
    print("Size:", file_path.stat().st_size, "bytes")

    source_counts = Counter(
        row.get("source_name", "unknown")
        for row in rows
    )
    print("Sources:", dict(source_counts))

    token_counts = [
        int(row.get("generated_token_count", 0))
        for row in rows
    ]
    if token_counts:
        print("Generated tokens, min:", min(token_counts))
        print("Generated tokens, mean:", sum(token_counts) / len(token_counts))
        print("Generated tokens, max:", max(token_counts))

    print()

    for index, row in enumerate(rows[:show_examples], start=1):
        entropies = (
            row.get("generation", {}).get("token_entropies", [])
            or []
        )

        print(f"EXAMPLE {index}")
        print("Source:", row.get("source_name"))
        print("Dataset:", row.get("source_dataset"))
        print()
        print("PREFIX:")
        print(row.get("prefix_text", ""))
        print()
        print("REAL CONTINUATION:")
        print(row.get("real_continuation", ""))
        print()
        print("QWEN CONTINUATION:")
        print(row.get("teacher_continuation", ""))
        print()
        print("Generated tokens:", row.get("generated_token_count"))

        if entropies:
            print("Entropy mean:", sum(entropies) / len(entropies))
            print("Entropy max:", max(entropies))

        print()
        print("-" * 100)
        print()

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a generated continuation JSONL file."
    )
    parser.add_argument("path", help="Path to the JSONL file.")
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of examples to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_jsonl(args.path, show_examples=args.show_examples)


if __name__ == "__main__":
    main()
