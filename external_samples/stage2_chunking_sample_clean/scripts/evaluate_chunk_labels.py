#!/usr/bin/env python3
"""Evaluate predicted chunk labels against expected benchmark labels."""

import argparse
import json
from collections import Counter, defaultdict
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


def text_preview(text, chars):
    flat = " ".join(str(text or "").split())
    if len(flat) <= chars:
        return flat
    return flat[: max(chars - 3, 0)] + "..."


def has_expected(record):
    return any(field in record and record.get(field) not in (None, "") for field in EXPECTED_FIELDS)


def tuple_values(record, fields):
    return tuple(normalize(record.get(field)) for field in fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit-mismatches", type=int, default=30)
    parser.add_argument("--preview-chars", type=int, default=180)
    parser.add_argument("--ignore-null-subfield", action="store_true")
    args = parser.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    evaluated = []
    skipped = 0
    for record in records:
        if has_expected(record):
            evaluated.append(record)
        else:
            skipped += 1

    correct_by_level = Counter()
    confusion = defaultdict(Counter)
    mismatches = []

    for record in evaluated:
        predicted = tuple_values(record, PREDICTED_FIELDS)
        expected = tuple_values(record, EXPECTED_FIELDS)

        for index, level in enumerate(PREDICTED_FIELDS):
            if level == "subfield" and args.ignore_null_subfield and expected[index] is None:
                correct_by_level[level] += 1
            elif predicted[index] == expected[index]:
                correct_by_level[level] += 1

        full_expected = expected
        full_predicted = predicted
        if args.ignore_null_subfield and expected[3] is None:
            full_expected = expected[:3]
            full_predicted = predicted[:3]
        if full_predicted == full_expected:
            correct_by_level["full_label"] += 1
        else:
            mismatches.append((record, expected, predicted))

        confusion[expected[0]][predicted[0]] += 1

    total = len(evaluated)

    def accuracy(name):
        if total == 0:
            return 0.0
        return correct_by_level[name] / total

    print(f"records_evaluated: {total}")
    print(f"records_skipped_no_expected: {skipped}")
    print(f"source_type_accuracy: {accuracy('source_type'):.4f}")
    print(f"domain_accuracy: {accuracy('domain'):.4f}")
    print(f"field_accuracy: {accuracy('field'):.4f}")
    print(f"subfield_accuracy: {accuracy('subfield'):.4f}")
    print(f"full_label_accuracy: {accuracy('full_label'):.4f}")

    print("source_type_confusion:")
    for expected_value in sorted(confusion, key=lambda value: str(value)):
        for predicted_value, count in sorted(confusion[expected_value].items(), key=lambda item: (-item[1], str(item[0]))):
            print(f"  {expected_value} -> {predicted_value}: {count}")

    print(f"mismatches_shown: {min(len(mismatches), args.limit_mismatches)}")
    for record, expected, predicted in mismatches[: args.limit_mismatches]:
        print(f"- chunk_id: {record.get('chunk_id')}")
        print(f"  expected: {expected}")
        print(f"  predicted: {predicted}")
        note = record.get("expected_label_note")
        if note:
            print(f"  note: {note}")
        print(f"  text: {text_preview(record.get('text'), args.preview_chars)}")


if __name__ == "__main__":
    main()
