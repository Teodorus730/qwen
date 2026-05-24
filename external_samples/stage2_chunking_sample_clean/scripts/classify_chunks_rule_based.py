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


def count_any(text_lower, signals):
    return sum(1 for signal in signals if signal in text_lower)


def has_word(text_lower, word):
    return re.search(rf"\b{re.escape(word)}\b", text_lower) is not None


def classify_mixed_language(text, text_lower):
    has_english = re.search(r"[A-Za-z]", text) is not None
    has_cyrillic = re.search(r"[\u0400-\u04ff]", text) is not None
    non_ascii_chars = sum(1 for char in text if ord(char) > 127)
    non_ascii_ratio = non_ascii_chars / max(len(text), 1)

    if has_english and (has_cyrillic or non_ascii_ratio >= 0.15):
        return make_label("unknown", "multilingual", "mixed_language", None, 0.7)
    return None


def classify_forum_qa(text, text_lower):
    if re.search(r"(?im)^\s*(question|answer|comment):", text):
        return make_label("forum_qa", "web", "forum_qa", "discussion", 0.8)
    return None


def classify_legal_government(text, text_lower):
    signals = ("public notice", "public hearing", "municipal", "ordinance", "regulation", "compliance", "city clerk")
    if count_any(text_lower, signals) >= 2:
        return make_label("legal_government", "government", "legal_notice", "public_information", 0.8)
    return None


def classify_news(text, text_lower):
    signals = ("officials", "announced", "reported", "according to", "statement", "spokesperson", "update")
    if count_any(text_lower, signals) >= 3:
        return make_label("news", "media", "news", "article", 0.78)
    return None


def classify_wiki_reference(text, text_lower):
    signals = ("overview:", "history:", "classification:", "references:", "definition", "encyclopedia")
    if count_any(text_lower, signals) >= 3:
        return make_label("wiki_reference", "reference", "encyclopedic_article", "general", 0.78)
    return None


def classify_commercial(text, text_lower):
    signals = ("buy", "shipping", "product", "customer note", "price", "warranty", "add to cart", "discount")
    if count_any(text_lower, signals) >= 2 or "features:" in text_lower:
        return make_label("commercial_product", "commercial", "product_page", "retail", 0.82)
    return None


def classify_math(text, text_lower):
    calculus_signals = ("\\lim", "derivative", "slope", "tangent", "limit", "integral", "f(x)")
    algebra_signals = ("algebra", "variable", "polynomial", "quadratic", "linear equation", "systems of equations")
    if count_any(text_lower, calculus_signals) >= 2 or "$$" in text:
        return make_label("math", "stem", "mathematics", "calculus", 0.88)
    if count_any(text_lower, algebra_signals) >= 2 or (has_word(text_lower, "equation") and has_word(text_lower, "variable")):
        return make_label("math", "stem", "mathematics", "algebra", 0.84)
    return None


def classify_code(text, text_lower):
    signals = (
        "```python",
        "def ",
        "parameters:",
        "returns:",
        "raises",
        "status_code",
        "api endpoint",
        "usage",
        "schema",
        "jsonl",
        "validate_chunks.py",
    )
    fenced_code = "```" in text
    if count_any(text_lower, signals) >= 2 or fenced_code:
        return make_label("code", "software", "programming", "documentation", 0.85)
    return None


def classify_biology(text, text_lower):
    signals = ("biology", "cell", "cells", "photosynthesis", "chloroplast", "chlorophyll", "ecosystem", "organism", "decomposers")
    if count_any(text_lower, signals) >= 2:
        return make_label("educational", "science", "biology", "article", 0.78)
    return None


def classify_physics(text, text_lower):
    signals = ("physics", "force", "motion", "velocity", "acceleration", "energy", "friction", "experiment")
    if count_any(text_lower, signals) >= 3:
        return make_label("educational", "science", "physics", "article", 0.78)
    return None


def classify_environmental_science(text, text_lower):
    signals = (
        "environmental",
        "stormwater",
        "pollution",
        "water quality",
        "habitat",
        "carbon emissions",
        "climate",
        "conservation",
        "wetlands",
        "runoff",
    )
    if count_any(text_lower, signals) >= 2:
        return make_label("educational", "science", "environmental_science", "article", 0.78)
    return None


def classify_infrastructure(text, text_lower):
    signals = ("urban", "water systems", "wastewater", "reservoirs", "pipes", "transit", "maintenance crews", "city transit")
    if count_any(text_lower, signals) >= 2:
        return make_label("educational", "infrastructure", "urban_systems", "article", 0.76)
    return None


def classify_boilerplate(text, text_lower):
    signals = ("cookie", "privacy", "terms", "footer", "subscribe", "skip to content", "related links", "manage preferences")
    signal_count = count_any(text_lower, signals)
    many_pipe_separators = text.count("|") >= 6
    repeated_cookie = text_lower.count("accept cookies") >= 2 or text_lower.count("manage preferences") >= 2
    if repeated_cookie or (many_pipe_separators and signal_count >= 2) or signal_count >= 4:
        return make_label("boilerplate_or_noise", "web", "boilerplate_or_navigation", "page_noise", 0.7)
    return None


def classify_educational(text, text_lower):
    signals = (
        "students",
        "classroom",
        "lesson",
        "measurement",
        "explains",
        "teacher",
        "learn",
        "article",
    )
    if has_any(text_lower, signals):
        return make_label("educational", "education", "general_education", "article", 0.62)
    return None


RULES = (
    classify_mixed_language,
    classify_forum_qa,
    classify_legal_government,
    classify_news,
    classify_wiki_reference,
    classify_commercial,
    classify_math,
    classify_code,
    classify_biology,
    classify_physics,
    classify_environmental_science,
    classify_infrastructure,
    classify_boilerplate,
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
    def dump_counter(counter):
        return json.dumps(
            {str(key): value for key, value in counter.items()},
            ensure_ascii=False,
            sort_keys=True,
        )

    print(f"records_read: {records_read}")
    print(f"records_written: {records_written}")
    print(f"counts_by_source_type: {dump_counter(source_type_counts)}")
    print(f"counts_by_domain: {dump_counter(domain_counts)}")
    print(f"counts_by_label_method: {dump_counter(label_method_counts)}")


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
