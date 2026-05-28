#!/usr/bin/env python
"""Inspect tokenization-aware annotation_v2 statistics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


NUMERIC_FIELDS = [
    "token_count",
    "token_per_byte",
    "tokens_per_char",
    "bytes_per_token",
    "tokens_per_word_rough",
]


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    record = json.loads(stripped)
                    if isinstance(record, dict):
                        records.append(record)
    return records


def nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def text_stats(record: dict[str, Any]) -> dict[str, Any]:
    value = nested(record, "annotation_v2", "text_stats")
    return value if isinstance(value, dict) else {}


def surface(record: dict[str, Any]) -> dict[str, Any]:
    value = nested(record, "annotation_v2", "surface")
    return value if isinstance(value, dict) else {}


def quality(record: dict[str, Any]) -> dict[str, Any]:
    value = nested(record, "annotation_v2", "quality")
    return value if isinstance(value, dict) else {}


def dataset(record: dict[str, Any]) -> str:
    value = nested(record, "annotation_v2", "provenance", "dataset") or record.get("dataset")
    return value if isinstance(value, str) and value else "unknown"


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not vals:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(vals),
        "min": round(min(vals), 6),
        "mean": round(statistics.fmean(vals), 6),
        "max": round(max(vals), 6),
    }


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {field: numeric_summary(text_stats(record).get(field) for record in records) for field in NUMERIC_FIELDS}


def preview(record: dict[str, Any], chars: int = 220) -> str:
    text = record.get("text")
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())[:chars]


def example(record: dict[str, Any]) -> dict[str, Any]:
    stats = text_stats(record)
    return {
        "chunk_id": record.get("chunk_id") or nested(record, "annotation_v2", "provenance", "chunk_id"),
        "dataset": dataset(record),
        "token_count": stats.get("token_count"),
        "token_per_byte": stats.get("token_per_byte"),
        "has_math_notation": surface(record).get("has_math_notation"),
        "has_code": surface(record).get("has_code"),
        "is_symbol_heavy": surface(record).get("is_symbol_heavy"),
        "noise_level": quality(record).get("noise_level"),
        "preview": preview(record),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_noise: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[dataset(record)].append(record)
        by_noise[str(quality(record).get("noise_level", "unknown"))].append(record)

    boolean_groups: dict[str, list[dict[str, Any]]] = {
        "has_math_notation=true": [r for r in records if surface(r).get("has_math_notation") is True],
        "has_math_notation=false": [r for r in records if surface(r).get("has_math_notation") is False],
        "has_code=true": [r for r in records if surface(r).get("has_code") is True],
        "has_code=false": [r for r in records if surface(r).get("has_code") is False],
        "is_symbol_heavy=true": [r for r in records if surface(r).get("is_symbol_heavy") is True],
        "is_symbol_heavy=false": [r for r in records if surface(r).get("is_symbol_heavy") is False],
    }

    sorted_by_density = sorted(records, key=lambda r: text_stats(r).get("token_per_byte", -1), reverse=True)
    sorted_by_tokens = sorted(records, key=lambda r: text_stats(r).get("token_count", -1), reverse=True)

    return {
        "records": len(records),
        "dataset_counts": dict(Counter(dataset(record) for record in records)),
        "overall": summarize_group(records),
        "by_dataset": {name: summarize_group(group) for name, group in sorted(by_dataset.items())},
        "by_surface_feature": {name: summarize_group(group) for name, group in sorted(boolean_groups.items())},
        "by_noise_level": {name: summarize_group(group) for name, group in sorted(by_noise.items())},
        "top_token_per_byte": [example(record) for record in sorted_by_density[:10]],
        "top_token_count": [example(record) for record in sorted_by_tokens[:10]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect tokenized annotation_v2 JSONL files.")
    parser.add_argument("--input", nargs="+", required=True, help="Input tokenized JSONL file(s).")
    parser.add_argument("--summary-json", help="Optional summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_records([Path(path) for path in args.input])
    summary = build_summary(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
