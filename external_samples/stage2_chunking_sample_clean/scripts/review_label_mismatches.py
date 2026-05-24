#!/usr/bin/env python3
"""Print compact review blocks for expected/predicted label disagreements."""

import argparse
import json
from pathlib import Path


PREDICTED_FIELDS = ("source_type", "domain", "field", "subfield")
EXPECTED_FIELDS = ("expected_source_type", "expected_domain", "expected_field", "expected_subfield")


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


def normalize(value):
    if value == "":
        return None
    return value


def label_tuple(record, fields):
    return tuple(normalize(record.get(field)) for field in fields)


def preview_text(text, chars):
    flat = " ".join(str(text or "").split())
    if len(flat) <= chars:
        return flat
    return flat[: max(chars - 3, 0)] + "..."


def is_match(record, only_source_type=False, only_domain=False):
    predicted = label_tuple(record, PREDICTED_FIELDS)
    expected = label_tuple(record, EXPECTED_FIELDS)
    if only_source_type:
        return predicted[0] == expected[0]
    if only_domain:
        return predicted[1] == expected[1]
    return predicted == expected


def print_record(record, preview_chars):
    expected = label_tuple(record, EXPECTED_FIELDS)
    predicted = label_tuple(record, PREDICTED_FIELDS)
    print(f"chunk_id: {record.get('chunk_id')}")
    print(f"expected: {expected}")
    print(f"predicted: {predicted}")
    print(f"expected_label_note: {record.get('expected_label_note')}")
    print(f"confidence: {record.get('confidence')}")
    print(f"label_method: {record.get('label_method')}")
    print(f"text: {preview_text(record.get('text'), preview_chars)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--only-source-type", action="store_true")
    parser.add_argument("--only-domain", action="store_true")
    parser.add_argument("--show-matches", action="store_true")
    args = parser.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    selected = []
    for record in records:
        matched = is_match(record, args.only_source_type, args.only_domain)
        if args.show_matches or not matched:
            selected.append(record)

    print(f"records_read: {len(records)}")
    print(f"records_selected: {len(selected)}")
    for record in selected[: args.limit]:
        print_record(record, args.preview_chars)


if __name__ == "__main__":
    main()
