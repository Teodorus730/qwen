#!/usr/bin/env python3
"""
Small rule-based labeling baseline for chunk JSONL files.

This script intentionally uses only Python standard library modules. It is a
transparent smoke-test baseline, not a model-based classifier.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


LABEL_FIELDS = ("source_type", "domain", "field", "subfield", "confidence")
RULE_METHOD = "rule_based_keyword_v1"
PASSTHROUGH_METHOD = "existing_label_passthrough"
UNKNOWN_METHOD = "rule_based_unknown"


def is_filled(value):
    return value is not None and value != ""


def has_existing_labels(record):
    return any(is_filled(record.get(field)) for field in ("domain", "field", "subfield", "confidence"))


def make_label(source_type, domain, field, subfield, confidence):
    return {
        "source_type": source_type,
        "domain": domain,
        "field": field,
        "subfield": subfield,
        "confidence": confidence,
        "label_method": RULE_METHOD,
    }


def has_any(text_lower, signals):
    return any(signal in text_lower for signal in signals)


def classify_code(text, text_lower):
    signals = ("```python", "def ", "return", "parameters:", "status_code", "client.get", "```")
    if has_any(text_lower, signals):
        return make_label("code", "software", "programming", "documentation", 0.85)
    return None


def classify_math(text, text_lower):
    signals = ("$$", "\\frac", "\\lim", "f(x)", "derivative", "slope", "formula")
    if has_any(text_lower, signals):
        return make_label("math", "stem", "mathematics", "calculus", 0.85)
    return None


def classify_commercial(text, text_lower):
    signals = ("introducing", "features:", "buy", "offer", "shipping", "product", "customer note")
    if has_any(text_lower, signals):
        return make_label("commercial_product", "commercial", "product_page", "retail", 0.8)
    return None


def classify_boilerplate(text, text_lower):
    signals = ("cookie", "privacy", "terms", "footer", "subscribe", "skip to content", "related links")
    many_pipe_separators = text.count("|") >= 4
    if has_any(text_lower, signals) or many_pipe_separators:
        return make_label("boilerplate_or_noise", "web", "boilerplate_or_navigation", "page_noise", 0.65)
    return None


def classify_forum_qa(text, text_lower):
    if re.search(r"(?im)^\s*(question|answer|comment):", text):
        return make_label("forum_qa", "web", "forum_qa", "discussion", 0.75)
    return None


def classify_mixed_language(text, text_lower):
    has_english = re.search(r"[A-Za-z]", text) is not None
    has_cyrillic = re.search(r"[\u0400-\u04ff]", text) is not None
    non_ascii_chars = sum(1 for char in text if ord(char) > 127)
    non_ascii_ratio = non_ascii_chars / max(len(text), 1)

    if has_english and (has_cyrillic or non_ascii_ratio >= 0.15):
        return make_label("unknown", "multilingual", "mixed_language", None, 0.6)
    return None


def classify_educational(text, text_lower):
    signals = (
        "students",
        "classroom",
        "lesson",
        "measurement",
        "explains",
        "##",
        "photosynthesis",
        "sensor",
        "temperature",
        "humidity",
        "scientific",
        "science",
    )
    if has_any(text_lower, signals):
        return make_label("educational", "education", "general_education", "article", 0.65)
    return None


RULES = (
    classify_code,
    classify_math,
    classify_commercial,
    classify_forum_qa,
    classify_boilerplate,
    classify_mixed_language,
    classify_educational,
)


def fallback_label(record):
    source_type = record.get("source_type") if is_filled(record.get("source_type")) else "unknown"
    return {
        "source_type": source_type,
        "domain": None,
        "field": None,
        "subfield": None,
        "confidence": 0.2,
        "label_method": UNKNOWN_METHOD,
    }


def classify_record(record, overwrite_existing_labels=False):
    out = dict(record)

    if not overwrite_existing_labels and has_existing_labels(record):
        out.setdefault("label_method", PASSTHROUGH_METHOD)
        return out

    text = record.get("text") or ""
    text_lower = text.lower()

    for rule in RULES:
        label = rule(text, text_lower)
        if label:
            out.update(label)
            return out

    out.update(fallback_label(record))
    return out


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records_read, records_written, source_type_counts, domain_counts, label_method_counts):
    print(f"records_read: {records_read}")
    print(f"records_written: {records_written}")
    print(f"counts_by_source_type: {json.dumps(dict(source_type_counts), ensure_ascii=False, sort_keys=True)}")
    print(f"counts_by_domain: {json.dumps(dict(domain_counts), ensure_ascii=False, sort_keys=True)}")
    print(f"counts_by_label_method: {json.dumps(dict(label_method_counts), ensure_ascii=False, sort_keys=True)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite-existing-labels", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    labeled_records = []
    for record in iter_jsonl(input_path):
        labeled_records.append(classify_record(record, args.overwrite_existing_labels))

    write_jsonl(output_path, labeled_records)

    source_type_counts = Counter(record.get("source_type") for record in labeled_records)
    domain_counts = Counter(record.get("domain") for record in labeled_records)
    label_method_counts = Counter(record.get("label_method") for record in labeled_records)

    print_summary(
        records_read=len(labeled_records),
        records_written=len(labeled_records),
        source_type_counts=source_type_counts,
        domain_counts=domain_counts,
        label_method_counts=label_method_counts,
    )


if __name__ == "__main__":
    main()
