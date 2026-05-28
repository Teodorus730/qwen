#!/usr/bin/env python
"""Evaluate annotation_v2 deterministic feature flags against review labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
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
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(record)
    return records


def preview(record: dict[str, Any], chars: int = 220) -> str:
    return " ".join(str(record.get("text") or "").split())[:chars]


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bool_metrics(records: list[dict[str, Any]], feature_name: str, review_name: str) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for record in records:
        predicted = bool(record["annotation_v2"]["surface"].get(feature_name))
        expected = bool(record.get(review_name))
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
            false_positives.append(example(record))
        elif not predicted and expected:
            fn += 1
            false_negatives.append(example(record))
        else:
            tn += 1
    result = prf(tp, fp, fn)
    result["tn"] = tn
    result["accuracy"] = round((tp + tn) / len(records), 4) if records else 0.0
    result["false_positives"] = false_positives
    result["false_negatives"] = false_negatives
    return result


def example(record: dict[str, Any]) -> dict[str, Any]:
    surface = record["annotation_v2"]["surface"]
    quality = record["annotation_v2"]["quality"]
    return {
        "chunk_id": record.get("chunk_id"),
        "dataset": record.get("dataset"),
        "preview": preview(record),
        "annotation_has_math_notation": surface.get("has_math_notation"),
        "annotation_has_code": surface.get("has_code"),
        "annotation_noise_level": quality.get("noise_level"),
        "annotation_is_symbol_heavy": surface.get("is_symbol_heavy"),
        "annotation_has_scientific_formula": surface.get("has_scientific_formula"),
        "annotation_has_api_or_command_syntax": surface.get("has_api_or_command_syntax"),
        "annotation_has_ui_residue": quality.get("has_ui_residue"),
        "annotation_has_forum_residue": quality.get("has_forum_residue"),
        "review_has_math_notation": record.get("review_has_math_notation"),
        "review_has_code": record.get("review_has_code"),
        "review_noise_level": record.get("review_noise_level"),
        "review_note": record.get("review_note"),
    }


def load_features_by_chunk_id(paths: list[str]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for path_arg in paths:
        path = Path(path_arg)
        for record in read_jsonl(path):
            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str):
                raise ValueError(f"{path}: feature record without string chunk_id")
            if chunk_id in features:
                raise ValueError(f"duplicate feature chunk_id across feature inputs: {chunk_id}")
            features[chunk_id] = record
    return features


def merge_review_with_features(
    review_records: list[dict[str, Any]],
    features_by_chunk_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for review in review_records:
        chunk_id = review.get("chunk_id")
        feature_record = features_by_chunk_id.get(str(chunk_id))
        if feature_record is None:
            raise ValueError(f"missing feature record for reviewed chunk_id: {chunk_id}")
        merged_record = dict(review)
        merged_record["annotation_v2"] = feature_record["annotation_v2"]
        merged.append(merged_record)
    return merged


def noise_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches: list[dict[str, Any]] = []
    correct = 0
    for record in records:
        predicted = record["annotation_v2"]["quality"].get("noise_level")
        expected = record.get("review_noise_level")
        confusion[str(expected)][str(predicted)] += 1
        if predicted == expected:
            correct += 1
        else:
            mismatches.append(example(record))
    return {
        "accuracy": round(correct / len(records), 4) if records else 0.0,
        "confusion_matrix": {
            expected: dict(predicted_counts)
            for expected, predicted_counts in sorted(confusion.items())
        },
        "mismatches": mismatches,
    }


def new_field_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "surface.is_symbol_heavy": lambda record: record["annotation_v2"]["surface"].get("is_symbol_heavy"),
        "surface.has_scientific_formula": lambda record: record["annotation_v2"]["surface"].get("has_scientific_formula"),
        "surface.has_api_or_command_syntax": lambda record: record["annotation_v2"]["surface"].get("has_api_or_command_syntax"),
        "quality.has_ui_residue": lambda record: record["annotation_v2"]["quality"].get("has_ui_residue"),
        "quality.has_forum_residue": lambda record: record["annotation_v2"]["quality"].get("has_forum_residue"),
    }
    coverage: dict[str, Any] = {}
    for name, getter in fields.items():
        values = [getter(record) for record in records]
        coverage[name] = {
            "present": sum(value is not None for value in values),
            "true": sum(value is True for value in values),
        }
    return coverage


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "has_math_notation": bool_metrics(records, "has_math_notation", "review_has_math_notation"),
        "has_code": bool_metrics(records, "has_code", "review_has_code"),
        "noise_level": noise_metrics(records),
        "new_field_coverage": new_field_coverage(records),
    }


def validate_review_fields(records: list[dict[str, Any]]) -> None:
    for record in records:
        chunk_id = record.get("chunk_id")
        if not isinstance(record.get("review_has_math_notation"), bool):
            raise ValueError(f"{chunk_id}: review_has_math_notation must be boolean")
        if not isinstance(record.get("review_has_code"), bool):
            raise ValueError(f"{chunk_id}: review_has_code must be boolean")
        if record.get("review_noise_level") not in {"clean", "partial_noise", "mostly_noise", "unknown"}:
            raise ValueError(f"{chunk_id}: invalid review_noise_level")
        if not isinstance(record.get("review_note"), str) or not record.get("review_note"):
            raise ValueError(f"{chunk_id}: review_note must be non-empty string")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate annotation_v2 feature review labels.")
    parser.add_argument("--input", required=True, help="Reviewed JSONL file.")
    parser.add_argument("--output", required=True, help="Output evaluation JSON.")
    parser.add_argument(
        "--features",
        nargs="+",
        help="Optional feature JSONL files to evaluate instead of embedded review annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = read_jsonl(input_path)
    validate_review_fields(records)
    metrics: dict[str, Any]
    if args.features:
        refined_records = merge_review_with_features(records, load_features_by_chunk_id(args.features))
        metrics = {
            "embedded_review_annotation": evaluate_records(records),
            "external_features": evaluate_records(refined_records),
        }
    else:
        metrics = evaluate_records(records)
    result = {
        "input": str(input_path),
        "features": args.features,
        "records": len(records),
        "dataset_distribution": dict(Counter(record.get("dataset") for record in records)),
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
