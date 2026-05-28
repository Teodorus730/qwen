#!/usr/bin/env python
"""Validate JSONL records enriched with annotation_v2 deterministic features."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_NOISE_LEVELS = {"clean", "partial_noise", "mostly_noise", "unknown"}
BOOLEAN_SURFACE_FIELDS = [
    "has_math_notation",
    "has_code",
    "has_numbers",
    "has_table_or_list",
    "has_urls_or_links",
    "has_boilerplate_markers",
]
NUMERIC_SURFACE_FIELDS = [
    "symbol_density",
    "digit_density",
    "uppercase_ratio",
    "punctuation_density",
]
TEXT_STATS_FIELDS = [
    "char_count",
    "byte_count",
    "line_count",
    "avg_line_length",
    "nonempty_line_count",
    "word_count_rough",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ValidationResult:
    def __init__(self, max_errors: int) -> None:
        self.max_errors = max_errors
        self.records_checked = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.schema_versions: Counter[str] = Counter()
        self.datasets: Counter[str] = Counter()
        self.noise_levels: Counter[str] = Counter()
        self.boolean_features: Counter[str] = Counter()
        self.seen_chunk_ids: set[str] = set()

    def error(self, line_no: int, message: str) -> None:
        if len(self.errors) < self.max_errors:
            self.errors.append(f"line {line_no}: {message}")

    def warning(self, line_no: int, message: str) -> None:
        if len(self.warnings) < self.max_errors:
            self.warnings.append(f"line {line_no}: {message}")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_object(result: ValidationResult, line_no: int, value: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        result.error(line_no, f"{name} must be an object")
        return None
    return value


def validate_provenance(
    result: ValidationResult,
    line_no: int,
    record: dict[str, Any],
    provenance: dict[str, Any],
) -> str | None:
    dataset = provenance.get("dataset")
    chunk_id = provenance.get("chunk_id")
    top_chunk_id = record.get("chunk_id")

    if not isinstance(dataset, str) or not dataset.strip():
        result.error(line_no, "annotation_v2.provenance.dataset must be a non-empty string")
    else:
        result.datasets[dataset] += 1

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        result.error(line_no, "annotation_v2.provenance.chunk_id must be a non-empty string")
        chunk_id = None
    if top_chunk_id is not None and not isinstance(top_chunk_id, str):
        result.error(line_no, "top-level chunk_id must be a string when present")
    if isinstance(top_chunk_id, str) and isinstance(chunk_id, str) and top_chunk_id != chunk_id:
        result.error(line_no, "top-level chunk_id and annotation_v2.provenance.chunk_id differ")

    effective_chunk_id = top_chunk_id if isinstance(top_chunk_id, str) else chunk_id
    if not effective_chunk_id:
        result.error(line_no, "chunk_id missing at both top-level and annotation_v2.provenance")
        return None
    if effective_chunk_id in result.seen_chunk_ids:
        result.error(line_no, f"duplicate chunk_id: {effective_chunk_id}")
    result.seen_chunk_ids.add(effective_chunk_id)

    for optional_name in ["source_dataset", "source_config", "source_split", "document_id"]:
        value = provenance.get(optional_name)
        if value is not None and not isinstance(value, str):
            result.error(line_no, f"annotation_v2.provenance.{optional_name} must be string or null")
    return effective_chunk_id


def validate_text_stats(
    result: ValidationResult,
    line_no: int,
    text: str,
    text_stats: dict[str, Any],
) -> None:
    for field in TEXT_STATS_FIELDS:
        if field not in text_stats:
            result.error(line_no, f"annotation_v2.text_stats.{field} is required")
        elif not is_number(text_stats[field]):
            result.error(line_no, f"annotation_v2.text_stats.{field} must be numeric")

    if not all(field in text_stats and is_number(text_stats[field]) for field in TEXT_STATS_FIELDS):
        return

    if text_stats["char_count"] != len(text):
        result.error(line_no, "text_stats.char_count does not match len(text)")
    if text_stats["byte_count"] != len(text.encode("utf-8")):
        result.error(line_no, "text_stats.byte_count does not match UTF-8 byte length")
    if text_stats["line_count"] < 1:
        result.error(line_no, "text_stats.line_count must be >= 1")
    if text_stats["nonempty_line_count"] < 0:
        result.error(line_no, "text_stats.nonempty_line_count must be >= 0")
    if text_stats["avg_line_length"] < 0:
        result.error(line_no, "text_stats.avg_line_length must be >= 0")
    if text_stats["word_count_rough"] < 0:
        result.error(line_no, "text_stats.word_count_rough must be >= 0")

    expected_avg = round(len(text) / text_stats["line_count"], 2) if text_stats["line_count"] else 0.0
    if abs(float(text_stats["avg_line_length"]) - expected_avg) > 0.02:
        result.error(line_no, "text_stats.avg_line_length does not match char_count / line_count")


def validate_surface(result: ValidationResult, line_no: int, text: str, surface: dict[str, Any]) -> None:
    for field in BOOLEAN_SURFACE_FIELDS:
        value = surface.get(field)
        if not isinstance(value, bool):
            result.error(line_no, f"annotation_v2.surface.{field} must be boolean")
        elif value:
            result.boolean_features[field] += 1

    for field in NUMERIC_SURFACE_FIELDS:
        value = surface.get(field)
        if not is_number(value):
            result.error(line_no, f"annotation_v2.surface.{field} must be numeric")
        elif value < 0 or value > 1:
            result.error(line_no, f"annotation_v2.surface.{field} must be in [0, 1]")

    has_digits = any(ch.isdigit() for ch in text)
    has_numbers = surface.get("has_numbers")
    if isinstance(has_numbers, bool):
        if has_digits and not has_numbers:
            result.warning(line_no, "text contains digits but surface.has_numbers=false")
        if not has_digits and has_numbers:
            result.warning(line_no, "text has no digits but surface.has_numbers=true")


def validate_quality(result: ValidationResult, line_no: int, quality: dict[str, Any]) -> None:
    noise_level = quality.get("noise_level")
    if not isinstance(noise_level, str) or noise_level not in EXPECTED_NOISE_LEVELS:
        result.error(line_no, "quality.noise_level must be clean, partial_noise, mostly_noise, or unknown")
    else:
        result.noise_levels[noise_level] += 1

    noise_score = quality.get("noise_score")
    if not is_number(noise_score):
        result.error(line_no, "quality.noise_score must be numeric")
    elif noise_score < 0 or noise_score > 1:
        result.error(line_no, "quality.noise_score must be in [0, 1]")

    noise_reasons = quality.get("noise_reasons")
    if not isinstance(noise_reasons, list):
        result.error(line_no, "quality.noise_reasons must be a list")
    elif any(not isinstance(item, str) for item in noise_reasons):
        result.error(line_no, "quality.noise_reasons must contain only strings")


def validate_record(result: ValidationResult, line_no: int, record: dict[str, Any]) -> None:
    result.records_checked += 1

    text = record.get("text")
    if not isinstance(text, str):
        result.error(line_no, "top-level text must exist and be a string")
        text = ""

    annotation = require_object(result, line_no, record.get("annotation_v2"), "annotation_v2")
    if annotation is None:
        return

    for section in ["provenance", "text_stats", "surface", "quality", "schema_version"]:
        if section not in annotation:
            result.error(line_no, f"annotation_v2.{section} is required")

    schema_version = annotation.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        result.error(line_no, "annotation_v2.schema_version must be a non-empty string")
    else:
        result.schema_versions[schema_version] += 1
        if "annotation_v2" not in schema_version:
            result.error(line_no, "annotation_v2.schema_version should include 'annotation_v2'")

    provenance = require_object(result, line_no, annotation.get("provenance"), "annotation_v2.provenance")
    text_stats = require_object(result, line_no, annotation.get("text_stats"), "annotation_v2.text_stats")
    surface = require_object(result, line_no, annotation.get("surface"), "annotation_v2.surface")
    quality = require_object(result, line_no, annotation.get("quality"), "annotation_v2.quality")

    if provenance is not None:
        validate_provenance(result, line_no, record, provenance)
    if text_stats is not None:
        validate_text_stats(result, line_no, text, text_stats)
    if surface is not None:
        validate_surface(result, line_no, text, surface)
    if quality is not None:
        validate_quality(result, line_no, quality)


def validate_file(path: Path, max_errors: int) -> ValidationResult:
    result = ValidationResult(max_errors=max_errors)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                result.error(line_no, f"invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                result.error(line_no, "record must be a JSON object")
                continue
            validate_record(result, line_no, record)
    return result


def summary_dict(input_path: Path, result: ValidationResult) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "records_checked": result.records_checked,
        "errors_count": len(result.errors),
        "warnings_count": len(result.warnings),
        "counts": {
            "schema_versions": dict(result.schema_versions),
            "datasets": dict(result.datasets),
            "noise_levels": dict(result.noise_levels),
            "boolean_features": dict(result.boolean_features),
        },
        "errors_sample": result.errors[:10],
        "warnings_sample": result.warnings[:10],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate annotation_v2 feature JSONL.")
    parser.add_argument("--input", required=True, help="Input feature JSONL.")
    parser.add_argument("--max-errors", type=int, default=50, help="Maximum errors/warnings to retain.")
    parser.add_argument("--summary-json", help="Optional machine-readable summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    result = validate_file(input_path, max_errors=args.max_errors)
    summary = summary_dict(input_path, result)
    print_summary(summary)
    if args.summary_json:
        output_path = Path(args.summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
