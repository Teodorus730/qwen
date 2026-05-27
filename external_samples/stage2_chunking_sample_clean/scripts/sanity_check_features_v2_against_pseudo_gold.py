#!/usr/bin/env python
"""Lightweight sanity checks for annotation v2 features against pseudo-gold labels."""

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
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def by_chunk_id(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            raise ValueError(f"{path}: record without chunk_id")
        if chunk_id in result:
            raise ValueError(f"{path}: duplicate chunk_id {chunk_id}")
        result[str(chunk_id)] = record
    return result


def rate(matches: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(matches / total, 4)


def metric(records: list[tuple[dict[str, Any], dict[str, Any]]], predicate) -> dict[str, Any]:
    total = len(records)
    matches = sum(1 for gold, feature in records if predicate(gold, feature))
    return {"count": total, "matches": matches, "rate": rate(matches, total)}


def annotation(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("annotation_v2")
    if not isinstance(value, dict):
        raise ValueError(f"record {record.get('chunk_id')} missing annotation_v2")
    return value


def build_summary(pseudo_gold: dict[str, dict[str, Any]], features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_features: list[str] = []
    for chunk_id, gold in pseudo_gold.items():
        feature = features.get(chunk_id)
        if feature is None:
            missing_features.append(chunk_id)
            continue
        joined.append((gold, feature))

    by_expected_source = defaultdict(list)
    by_dataset = defaultdict(list)
    for gold, feature in joined:
        by_expected_source[gold.get("expected_source_type")].append((gold, feature))
        by_dataset[gold.get("dataset")].append((gold, feature))

    boilerplate_records = by_expected_source.get("boilerplate_or_noise", [])
    fineweb_records = by_dataset.get("FineWeb", [])
    fineweb_edu_records = by_dataset.get("FineWeb-Edu", [])
    finemath_records = by_dataset.get("FineMath", [])

    summary = {
        "pseudo_gold_records": len(pseudo_gold),
        "feature_records": len(features),
        "joined_records": len(joined),
        "missing_feature_chunk_ids": missing_features,
        "expected_source_type_distribution": dict(Counter(gold.get("expected_source_type") for gold, _ in joined)),
        "dataset_distribution": dict(Counter(gold.get("dataset") for gold, _ in joined)),
        "checks": {
            "expected_math_has_math_notation": metric(
                by_expected_source.get("math", []),
                lambda _gold, feature: bool(annotation(feature)["surface"].get("has_math_notation")),
            ),
            "expected_code_has_code": metric(
                by_expected_source.get("code", []),
                lambda _gold, feature: bool(annotation(feature)["surface"].get("has_code")),
            ),
            "expected_boilerplate_noise_detected": metric(
                boilerplate_records,
                lambda _gold, feature: annotation(feature)["quality"].get("noise_level")
                in {"partial_noise", "mostly_noise"},
            ),
            "finemath_has_math_notation": metric(
                finemath_records,
                lambda _gold, feature: bool(annotation(feature)["surface"].get("has_math_notation")),
            ),
            "fineweb_has_boilerplate_markers": metric(
                fineweb_records,
                lambda _gold, feature: bool(annotation(feature)["surface"].get("has_boilerplate_markers")),
            ),
            "fineweb_edu_has_boilerplate_markers": metric(
                fineweb_edu_records,
                lambda _gold, feature: bool(annotation(feature)["surface"].get("has_boilerplate_markers")),
            ),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check annotation v2 deterministic features against pseudo-gold."
    )
    parser.add_argument("--pseudo-gold", required=True, help="Pseudo-gold JSONL.")
    parser.add_argument("--features", nargs="+", required=True, help="One or more feature JSONL files.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_path = Path(args.pseudo_gold)
    pseudo_gold = by_chunk_id(read_jsonl(gold_path), gold_path)
    feature_records: dict[str, dict[str, Any]] = {}
    for feature_arg in args.features:
        feature_path = Path(feature_arg)
        for chunk_id, record in by_chunk_id(read_jsonl(feature_path), feature_path).items():
            if chunk_id in feature_records:
                raise ValueError(f"duplicate feature chunk_id across inputs: {chunk_id}")
            feature_records[chunk_id] = record
    summary = build_summary(pseudo_gold, feature_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
