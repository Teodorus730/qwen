#!/usr/bin/env python3
"""Compare two labeled JSONL outputs by chunk_id."""

import argparse
import json
from pathlib import Path


LABEL_FIELDS = ("source_type", "domain", "field", "subfield")
EXPECTED_FIELDS = ("expected_source_type", "expected_domain", "expected_field", "expected_subfield")


def load_jsonl_by_chunk_id(path):
    records = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Invalid JSON in {path} line {line_number}: {exc}")
                    raise SystemExit(2)
                chunk_id = record.get("chunk_id")
                if chunk_id:
                    records[chunk_id] = record
    except FileNotFoundError:
        print(f"Missing input file: {path}")
        raise SystemExit(2)
    return records


def label_tuple(record, fields=LABEL_FIELDS):
    return tuple(record.get(field) if record.get(field) != "" else None for field in fields)


def preview(text, chars):
    flat = " ".join(str(text or "").split())
    if len(flat) <= chars:
        return flat
    return flat[: max(chars - 3, 0)] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--preview-chars", type=int, default=180)
    parser.add_argument("--ignore-source-type", action="store_true")
    parser.add_argument("--ignore-subfield", action="store_true")
    args = parser.parse_args()

    left = load_jsonl_by_chunk_id(Path(args.left))
    right = load_jsonl_by_chunk_id(Path(args.right))
    common = sorted(set(left) & set(right))

    compared_fields = list(LABEL_FIELDS)
    if args.ignore_source_type:
        compared_fields.remove("source_type")
    if args.ignore_subfield:
        compared_fields.remove("subfield")

    matching = []
    differing = []
    field_disagreements = {field: 0 for field in LABEL_FIELDS}

    for chunk_id in common:
        left_tuple = label_tuple(left[chunk_id], tuple(compared_fields))
        right_tuple = label_tuple(right[chunk_id], tuple(compared_fields))
        if left_tuple == right_tuple:
            matching.append(chunk_id)
        else:
            differing.append(chunk_id)
        for field in LABEL_FIELDS:
            if left[chunk_id].get(field) != right[chunk_id].get(field):
                field_disagreements[field] += 1

    agreement_rate = len(matching) / len(common) if common else 0.0
    print(f"records_in_{args.left_name}: {len(left)}")
    print(f"records_in_{args.right_name}: {len(right)}")
    print(f"common_chunk_ids: {len(common)}")
    print(f"matching_full_labels: {len(matching)}")
    print(f"differing_full_labels: {len(differing)}")
    print(f"agreement_rate: {agreement_rate:.4f}")
    print("disagreements_by_field:")
    for field in LABEL_FIELDS:
        print(f"  {field}: {field_disagreements[field]}")

    print(f"disagreements_shown: {min(len(differing), args.limit)}")
    for chunk_id in differing[: args.limit]:
        left_record = left[chunk_id]
        right_record = right[chunk_id]
        expected = label_tuple(left_record, EXPECTED_FIELDS)
        print(f"- chunk_id: {chunk_id}")
        print(f"  {args.left_name}: {label_tuple(left_record)}")
        print(f"  {args.right_name}: {label_tuple(right_record)}")
        print(f"  expected: {expected}")
        print(f"  text: {preview(left_record.get('text') or right_record.get('text'), args.preview_chars)}")


if __name__ == "__main__":
    main()
