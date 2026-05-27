#!/usr/bin/env python
"""Inspect annotation schema v2 deterministic feature distributions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def get_annotation(record: dict[str, Any]) -> dict[str, Any]:
    annotation = record.get("annotation_v2")
    if not isinstance(annotation, dict):
        raise ValueError(f"record {record.get('chunk_id')} missing annotation_v2")
    return annotation


def numeric_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": round(mean(values), 4),
        "max": max(values),
    }


def share(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def preview(record: dict[str, Any], chars: int = 240) -> str:
    text = str(record.get("text") or "")
    text = " ".join(text.split())
    return text[:chars]


def example_record(record: dict[str, Any], score_field: str | None = None) -> dict[str, Any]:
    annotation = get_annotation(record)
    surface = annotation["surface"]
    quality = annotation["quality"]
    item = {
        "chunk_id": record.get("chunk_id"),
        "dataset": record.get("dataset"),
        "noise_level": quality.get("noise_level"),
        "noise_reasons": quality.get("noise_reasons"),
        "preview": preview(record),
    }
    if score_field:
        item[score_field] = surface.get(score_field)
    return item


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    by_dataset = Counter(record.get("dataset") or "unknown" for record in records)
    text_stats = [get_annotation(record)["text_stats"] for record in records]
    surfaces = [get_annotation(record)["surface"] for record in records]
    qualities = [get_annotation(record)["quality"] for record in records]

    flag_names = [
        "has_math_notation",
        "has_code",
        "has_numbers",
        "has_table_or_list",
        "has_urls_or_links",
        "has_boilerplate_markers",
    ]
    flags = {
        name: {
            "count": sum(1 for surface in surfaces if surface.get(name)),
            "share": share(sum(1 for surface in surfaces if surface.get(name)), total),
        }
        for name in flag_names
    }

    reason_counts: Counter[str] = Counter()
    for quality in qualities:
        reason_counts.update(quality.get("noise_reasons") or [])

    examples: dict[str, list[dict[str, Any]]] = {}
    examples["has_math_notation"] = [
        example_record(record)
        for record in records
        if get_annotation(record)["surface"].get("has_math_notation")
    ][:5]
    examples["has_code"] = [
        example_record(record)
        for record in records
        if get_annotation(record)["surface"].get("has_code")
    ][:5]
    examples["noise"] = [
        example_record(record)
        for record in records
        if get_annotation(record)["quality"].get("noise_level") in {"partial_noise", "mostly_noise"}
    ][:5]
    examples["high_symbol_density"] = [
        example_record(record, "symbol_density")
        for record in sorted(
            records,
            key=lambda item: get_annotation(item)["surface"].get("symbol_density", 0.0),
            reverse=True,
        )[:5]
    ]
    examples["high_digit_density"] = [
        example_record(record, "digit_density")
        for record in sorted(
            records,
            key=lambda item: get_annotation(item)["surface"].get("digit_density", 0.0),
            reverse=True,
        )[:5]
    ]

    return {
        "records": total,
        "by_dataset": dict(sorted(by_dataset.items())),
        "char_count": numeric_summary([stats["char_count"] for stats in text_stats]),
        "byte_count": numeric_summary([stats["byte_count"] for stats in text_stats]),
        "line_count": numeric_summary([stats["line_count"] for stats in text_stats]),
        "flags": flags,
        "noise_level": dict(Counter(quality["noise_level"] for quality in qualities).most_common()),
        "top_noise_reasons": dict(reason_counts.most_common(20)),
        "examples": examples,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect annotation schema v2 feature JSONL.")
    parser.add_argument("--input", nargs="+", required=True, help="Input feature JSONL file(s).")
    parser.add_argument("--summary-output", help="Optional JSON summary output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for input_arg in args.input:
        records.extend(read_jsonl(Path(input_arg)))
    summary = build_summary(records)
    summary["input"] = args.input
    print_summary(summary)
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
