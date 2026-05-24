#!/usr/bin/env python3
"""Inspect chunk JSONL files with compact summaries."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


DEFAULT_GROUPS = ("dataset", "source_type", "domain", "field", "label_method")


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON error in {path} line {line_number}: {exc}") from exc


def display_value(value):
    if value is None:
        return "null"
    if value == "":
        return '""'
    return str(value)


def preview_text(text, chars):
    text = " ".join(str(text or "").split())
    if len(text) <= chars:
        return text
    return text[: max(chars - 3, 0)] + "..."


def print_counts(records, field):
    counts = Counter(record.get(field) for record in records)
    print(f"counts_by_{field}:")
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"  {display_value(value)}: {count}")


def print_token_stats(records):
    token_counts = [record.get("token_count") for record in records if isinstance(record.get("token_count"), int)]
    if not token_counts:
        print("token_count_stats: no integer token_count values")
        return
    mean_value = statistics.mean(token_counts)
    median_value = statistics.median(token_counts)
    print(
        "token_count_stats: "
        f"min={min(token_counts)} mean={mean_value:.2f} "
        f"median={median_value:.2f} max={max(token_counts)}"
    )


def print_compact_records(records, limit, show_text, text_chars):
    print(f"first_{limit}_records:")
    header = (
        "chunk_id | dataset | source_type | domain | field | subfield | "
        "confidence | label_method | token_count | text preview"
    )
    print(header)
    for record in records[:limit]:
        fields = [
            record.get("chunk_id"),
            record.get("dataset"),
            record.get("source_type"),
            record.get("domain"),
            record.get("field"),
            record.get("subfield"),
            record.get("confidence"),
            record.get("label_method"),
            record.get("token_count"),
        ]
        preview = preview_text(record.get("text"), text_chars if show_text else min(text_chars, 80))
        print(" | ".join(display_value(value) for value in fields) + f" | {preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--text-chars", type=int, default=160)
    parser.add_argument("--group-by", choices=DEFAULT_GROUPS, action="append")
    args = parser.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    print(f"records_count: {len(records)}")

    groups = args.group_by or DEFAULT_GROUPS
    for field in groups:
        if field == "label_method" and not any("label_method" in record for record in records):
            continue
        print_counts(records, field)

    print_token_stats(records)
    print_compact_records(records, args.limit, args.show_text, args.text_chars)


if __name__ == "__main__":
    main()
