#!/usr/bin/env python3
"""Check predicted label tuples against the local taxonomy label space."""

import argparse
import json
from pathlib import Path


def iter_jsonl(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Invalid JSON in {path} line {line_number}: {exc}")
                    raise SystemExit(2)
    except FileNotFoundError:
        print(f"Missing input file: {path}")
        raise SystemExit(2)


def load_taxonomy(path):
    try:
        labels = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Missing labels file: {path}")
        raise SystemExit(2)
    except json.JSONDecodeError as exc:
        print(f"Invalid labels JSON in {path}: {exc}")
        raise SystemExit(2)
    return {
        (label.get("domain"), label.get("field"), label.get("subfield"))
        for label in labels
    }


def label_tuple(record):
    return (record.get("domain"), record.get("field"), record.get("subfield"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--allow-null", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    taxonomy = load_taxonomy(Path(args.labels))
    records = list(iter_jsonl(Path(args.input)))
    predicted = {label_tuple(record) for record in records}
    source_types = {record.get("source_type") for record in records}

    covered = sorted(predicted & taxonomy, key=str)
    missing = sorted(predicted - taxonomy, key=str)
    if args.allow_null:
        missing = [item for item in missing if item != (None, None, None)]

    print(f"records_checked: {len(records)}")
    print(f"unique_predicted_label_tuples: {len(predicted)}")
    print(f"labels_covered_by_taxonomy: {len(covered)}")
    for item in covered:
        print(f"  covered: {item}")
    print(f"labels_missing_from_taxonomy: {len(missing)}")
    for item in missing:
        print(f"  missing: {item}")
    print(f"unique_source_type_values: {len(source_types)}")
    for value in sorted(source_types, key=str):
        print(f"  source_type: {value}")

    non_null_missing = [item for item in missing if item != (None, None, None)]
    if non_null_missing:
        print("WARNING: some predicted domain/field/subfield tuples are not represented in taxonomy")
    if args.strict and non_null_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
