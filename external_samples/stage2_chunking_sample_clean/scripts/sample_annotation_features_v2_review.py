#!/usr/bin/env python
"""Sample annotation_v2 deterministic feature records for manual review."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower().strip()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def annotation(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("annotation_v2")
    if not isinstance(value, dict):
        raise ValueError(f"record {record.get('chunk_id')} missing annotation_v2")
    return value


def text_preview(text: str, chars: int) -> str:
    return " ".join(text.split())[:chars]


def add_records(
    selected: list[dict[str, Any]],
    seen: set[str],
    candidates: list[dict[str, Any]],
    limit: int,
) -> None:
    for record in candidates:
        if len(selected) >= limit:
            return
        chunk_id = str(record.get("chunk_id") or annotation(record)["provenance"].get("chunk_id"))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        selected.append(record)


def interesting_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    categories = [
        lambda record: annotation(record)["surface"].get("has_math_notation") is True,
        lambda record: annotation(record)["surface"].get("has_code") is True,
        lambda record: annotation(record)["surface"].get("has_boilerplate_markers") is True,
        lambda record: annotation(record)["quality"].get("noise_level") in {"partial_noise", "mostly_noise"},
        lambda record: annotation(record)["surface"].get("symbol_density", 0.0) >= 0.12,
        lambda record: annotation(record)["surface"].get("digit_density", 0.0) >= 0.08,
        lambda record: record.get("expected_source_type") == "math"
        and annotation(record)["surface"].get("has_math_notation") is False,
        lambda record: annotation(record)["surface"].get("has_code") is True
        and str(record.get("review_note") or "").lower().find("patent") >= 0,
        lambda record: annotation(record)["surface"].get("has_boilerplate_markers") is True
        and annotation(record)["quality"].get("noise_level") == "clean",
    ]
    for predicate in categories:
        add_records(selected, seen, [record for record in records if predicate(record)], limit=len(records))
    return selected


def per_dataset_sample(records: list[dict[str, Any]], per_dataset: int, rng: random.Random) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dataset = str(record.get("dataset") or annotation(record)["provenance"].get("dataset") or "unknown")
        by_dataset.setdefault(dataset, []).append(record)
    selected: list[dict[str, Any]] = []
    for dataset_records in sorted(by_dataset.values(), key=lambda values: str(values[0].get("dataset"))):
        candidates = list(dataset_records)
        rng.shuffle(candidates)
        selected.extend(candidates[:per_dataset])
    return selected


def make_review_record(record: dict[str, Any], review_id: int, text_chars: int) -> dict[str, Any]:
    ann = annotation(record)
    text = str(record.get("text") or "")
    return {
        "review_id": f"annotation_v2_review_{review_id:06d}",
        "chunk_id": record.get("chunk_id") or ann["provenance"].get("chunk_id"),
        "dataset": record.get("dataset") or ann["provenance"].get("dataset"),
        "text_preview": text_preview(text, text_chars),
        "text": text,
        "annotation_v2": {
            "text_stats": ann.get("text_stats"),
            "surface": ann.get("surface"),
            "quality": ann.get("quality"),
            "schema_version": ann.get("schema_version"),
        },
        "review_has_math_notation": None,
        "review_has_code": None,
        "review_noise_level": None,
        "review_note": None,
    }


def sample_records(
    records: list[dict[str, Any]],
    max_records: int,
    seed: int,
    per_dataset: int | None,
    include_interesting: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    if per_dataset is not None:
        add_records(selected, seen, per_dataset_sample(records, per_dataset, rng), max_records)

    if include_interesting:
        add_records(selected, seen, interesting_candidates(records), max_records)

    shuffled = list(records)
    rng.shuffle(shuffled)
    add_records(selected, seen, shuffled, max_records)
    return selected[:max_records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample annotation_v2 feature records for review.")
    parser.add_argument("--input", nargs="+", required=True, help="Input feature JSONL file(s).")
    parser.add_argument("--output", required=True, help="Output review sample JSONL.")
    parser.add_argument("--max-records", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-dataset", type=int)
    parser.add_argument("--include-interesting", type=str_to_bool, default=True)
    parser.add_argument("--text-chars", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for input_arg in args.input:
        records.extend(read_jsonl(Path(input_arg)))
    sampled = sample_records(
        records,
        max_records=args.max_records,
        seed=args.seed,
        per_dataset=args.per_dataset,
        include_interesting=args.include_interesting,
    )
    review_records = [
        make_review_record(record, index, text_chars=args.text_chars)
        for index, record in enumerate(sampled)
    ]
    write_jsonl(Path(args.output), review_records)
    print(
        json.dumps(
            {
                "records_read": len(records),
                "records_written": len(review_records),
                "output": args.output,
                "datasets": sorted({record.get("dataset") for record in review_records}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
