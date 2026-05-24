#!/usr/bin/env python3
"""Validate stage2 chunk JSONL schema."""

import argparse
import json
from pathlib import Path


BASE_REQUIRED = ("chunk_id", "dataset", "source_type", "token_count", "text")
LABEL_REQUIRED = ("domain", "field", "confidence", "label_method")


def add_issue(issues, level, line_number, message):
    issues.append((level, line_number, message))


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_record(record, line_number, seen_chunk_ids, require_labels, errors, warnings):
    for field in BASE_REQUIRED:
        if field not in record:
            add_issue(errors, "ERROR", line_number, f"missing required field: {field}")

    chunk_id = record.get("chunk_id")
    if chunk_id in seen_chunk_ids:
        add_issue(errors, "ERROR", line_number, f"duplicate chunk_id: {chunk_id}")
    elif chunk_id is not None:
        seen_chunk_ids.add(chunk_id)

    token_count = record.get("token_count")
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
        add_issue(errors, "ERROR", line_number, "token_count must be an int > 0")

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        add_issue(errors, "ERROR", line_number, "text must be a non-empty string")

    if "confidence" in record:
        confidence = record.get("confidence")
        if confidence is not None and (not is_number(confidence) or confidence < 0 or confidence > 1):
            add_issue(errors, "ERROR", line_number, "confidence must be null or a number from 0 to 1")

    if "label_method" in record and not isinstance(record.get("label_method"), str):
        add_issue(errors, "ERROR", line_number, "label_method must be a string when present")

    for nullable_field in ("domain", "field", "subfield"):
        if nullable_field in record and record.get(nullable_field) is not None and not isinstance(record.get(nullable_field), str):
            add_issue(errors, "ERROR", line_number, f"{nullable_field} must be string or null")

    if require_labels:
        for field in LABEL_REQUIRED:
            if field not in record:
                add_issue(errors, "ERROR", line_number, f"missing required label field: {field}")
            elif field != "confidence" and record.get(field) in (None, ""):
                add_issue(errors, "ERROR", line_number, f"label field must be non-empty: {field}")
        if "confidence" in record and record.get("confidence") is None:
            add_issue(errors, "ERROR", line_number, "confidence must be non-null with --require-labels")

    expected_fields = (
        "expected_source_type",
        "expected_domain",
        "expected_field",
        "expected_subfield",
        "expected_label_note",
    )
    for field in expected_fields:
        if field in record and record.get(field) is not None and not isinstance(record.get(field), str):
            add_issue(warnings, "WARNING", line_number, f"{field} is expected to be string or null")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--require-labels", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    errors = []
    warnings = []
    seen_chunk_ids = set()
    records_checked = 0

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    add_issue(errors, "ERROR", line_number, f"invalid JSON: {exc}")
                    continue
                records_checked += 1
                if not isinstance(record, dict):
                    add_issue(errors, "ERROR", line_number, "record must be a JSON object")
                    continue
                validate_record(record, line_number, seen_chunk_ids, args.require_labels, errors, warnings)
    except FileNotFoundError:
        print("VALIDATION FAILED")
        print("records checked: 0")
        print("errors count: 1")
        print("warnings count: 0")
        print(f"ERROR line -: missing file: {input_path}")
        raise SystemExit(2)

    print("VALIDATION PASSED" if not errors else "VALIDATION FAILED")
    print(f"records checked: {records_checked}")
    print(f"errors count: {len(errors)}")
    print(f"warnings count: {len(warnings)}")

    for issue in (errors + warnings)[:20]:
        level, line_number, message = issue
        print(f"{level} line {line_number}: {message}")

    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
