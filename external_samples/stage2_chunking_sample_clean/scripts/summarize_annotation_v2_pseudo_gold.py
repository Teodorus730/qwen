#!/usr/bin/env python
"""Summarize annotation_v2 pseudo-gold JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_TOPIC_DOMAINS = {
    "stem",
    "science",
    "technology",
    "software",
    "humanities",
    "social_sciences",
    "commercial",
    "government",
    "media",
    "reference",
    "education",
    "unknown",
}
ALLOWED_NOISE_LEVELS = {"clean", "partial_noise", "mostly_noise", "unknown"}
REQUIRED_BOOL_FIELDS = [
    "review_topic_abstained",
    "review_has_math_notation",
    "review_has_code",
    "review_is_symbol_heavy",
    "review_has_scientific_formula",
    "review_has_api_or_command_syntax",
    "review_has_ui_residue",
    "review_has_forum_residue",
]
REQUIRED_FIELDS = [
    "chunk_id",
    "dataset",
    "review_topic_domain",
    "review_topic_confidence",
    "review_topic_note",
    "review_noise_level",
    "review_confidence",
    "review_note",
    *REQUIRED_BOOL_FIELDS,
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: record must be object")
            records.append(record)
    return records


def share(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def bool_count(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    count = sum(1 for record in records if record.get(field) is True)
    return {"count": count, "share": share(count, len(records))}


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        for field in REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"record {index}: missing {field}")
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            errors.append(f"record {index}: invalid chunk_id")
        elif chunk_id in seen:
            errors.append(f"record {index}: duplicate chunk_id {chunk_id}")
        else:
            seen.add(chunk_id)
        domain = record.get("review_topic_domain")
        if domain not in ALLOWED_TOPIC_DOMAINS:
            errors.append(f"record {index} {chunk_id}: invalid review_topic_domain {domain!r}")
        noise = record.get("review_noise_level")
        if noise not in ALLOWED_NOISE_LEVELS:
            errors.append(f"record {index} {chunk_id}: invalid review_noise_level {noise!r}")
        for field in REQUIRED_BOOL_FIELDS:
            if not isinstance(record.get(field), bool):
                errors.append(f"record {index} {chunk_id}: {field} must be boolean")
        for field in ["review_topic_confidence", "review_confidence"]:
            value = record.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0.0 or value > 1.0:
                errors.append(f"record {index} {chunk_id}: {field} must be numeric in [0, 1]")
    return {"errors_count": len(errors), "errors_sample": errors[:20]}


def examples(records: list[dict[str, Any]], predicate, limit: int = 8) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if predicate(record):
            selected.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "dataset": record.get("dataset"),
                    "review_topic_domain": record.get("review_topic_domain"),
                    "review_topic_confidence": record.get("review_topic_confidence"),
                    "review_noise_level": record.get("review_noise_level"),
                    "review_note": record.get("review_note"),
                    "text_preview": record.get("text_preview", "")[:240],
                }
            )
        if len(selected) >= limit:
            break
    return selected


def summarize(records: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    total = len(records)
    topic_abstained = sum(1 for record in records if record.get("review_topic_abstained") is True)
    confidence_bins = Counter(str(record.get("review_confidence")) for record in records)
    topic_confidence_bins = Counter(str(record.get("review_topic_confidence")) for record in records)
    summary = {
        "input": str(input_path),
        "records_count": total,
        "integrity": validate(records),
        "by_dataset": dict(Counter(record.get("dataset") for record in records)),
        "review_topic_domain_distribution": dict(Counter(record.get("review_topic_domain") for record in records)),
        "review_topic_abstained": {"count": topic_abstained, "share": share(topic_abstained, total)},
        "surface": {
            "review_has_math_notation": bool_count(records, "review_has_math_notation"),
            "review_has_code": bool_count(records, "review_has_code"),
            "review_is_symbol_heavy": bool_count(records, "review_is_symbol_heavy"),
            "review_has_scientific_formula": bool_count(records, "review_has_scientific_formula"),
            "review_has_api_or_command_syntax": bool_count(records, "review_has_api_or_command_syntax"),
        },
        "quality": {
            "review_noise_level_distribution": dict(Counter(record.get("review_noise_level") for record in records)),
            "review_has_ui_residue": bool_count(records, "review_has_ui_residue"),
            "review_has_forum_residue": bool_count(records, "review_has_forum_residue"),
        },
        "confidence_distribution": dict(confidence_bins),
        "topic_confidence_distribution": dict(topic_confidence_bins),
        "examples": {
            "abstained_or_unknown_topic": examples(
                records,
                lambda record: record.get("review_topic_abstained") is True
                or record.get("review_topic_domain") == "unknown",
            ),
            "mixed_or_ambiguous_notes": examples(
                records,
                lambda record: any(
                    marker in str(record.get("review_note", "")).lower()
                    for marker in ["mixed", "ambiguous", "weak", "unknown", "mostly noise"]
                ),
            ),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize annotation_v2 pseudo-gold JSONL.")
    parser.add_argument("--input", required=True, help="Input pseudo-gold JSONL.")
    parser.add_argument("--output-json", required=True, help="Output summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    records = load_jsonl(input_path)
    summary = summarize(records, input_path)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["integrity"]["errors_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
