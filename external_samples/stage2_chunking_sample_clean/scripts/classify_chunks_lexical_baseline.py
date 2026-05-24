#!/usr/bin/env python3
"""No-dependency lexical nearest-label baseline for chunk JSONL files."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


METHOD = "lexical_nearest_label_v1"
LOW_CONFIDENCE_METHOD = "lexical_nearest_label_low_confidence"
LABEL_TO_SOURCE_TYPE = {
    ("stem", "mathematics", "calculus"): "math",
    ("stem", "mathematics", "algebra"): "math",
    ("software", "programming", "documentation"): "code",
    ("commercial", "product_page", "retail"): "commercial_product",
    ("web", "boilerplate_or_navigation", "page_noise"): "boilerplate_or_noise",
    ("web", "forum_qa", "discussion"): "forum_qa",
    ("education", "general_education", "article"): "educational",
    ("science", "biology", "article"): "educational",
    ("science", "physics", "article"): "educational",
    ("science", "environmental_science", "article"): "educational",
    ("infrastructure", "urban_systems", "article"): "educational",
    ("multilingual", "mixed_language", None): "unknown",
    ("government", "legal_notice", "public_information"): "legal_government",
    ("reference", "encyclopedic_article", "general"): "wiki_reference",
    ("media", "news", "article"): "news",
    ("unknown", "unknown", None): "unknown",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "with", "about", "between", "can",
    "may", "such", "used", "uses", "using", "text", "texts",
}


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


def load_labels(path):
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise ValueError("labels file must contain a JSON array")
    return labels


def tokenize(text):
    text = str(text or "").lower().replace("_", " ")
    tokens = re.findall(r"[a-z0-9]+", text)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def label_text(label):
    parts = [
        label.get("domain"),
        label.get("field"),
        label.get("subfield"),
        label.get("description"),
        " ".join(label.get("keywords") or []),
    ]
    return " ".join(str(part) for part in parts if part)


def vectorize(text):
    return Counter(tokenize(text))


def cosine(left, right):
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def classify_record(record, label_vectors, labels, min_confidence, text_chars, overwrite_existing_labels):
    out = dict(record)
    has_existing = any(out.get(field) not in (None, "") for field in ("domain", "field", "subfield", "confidence"))
    if has_existing and not overwrite_existing_labels:
        return out

    chunk_vector = vectorize(str(record.get("text") or "")[:text_chars])
    scores = [(cosine(chunk_vector, label_vector), index) for index, label_vector in enumerate(label_vectors)]
    scores.sort(reverse=True)
    best_score, best_index = scores[0] if scores else (0.0, 0)
    best_label = labels[best_index]

    if best_score < min_confidence:
        out.update(
            {
                "source_type": out.get("source_type") or "unknown",
                "domain": None,
                "field": None,
                "subfield": None,
                "confidence": float(best_score),
                "label_method": LOW_CONFIDENCE_METHOD,
            }
        )
    else:
        label_tuple = (best_label.get("domain"), best_label.get("field"), best_label.get("subfield"))
        source_type = out.get("source_type")
        if source_type in (None, "", "unknown"):
            source_type = LABEL_TO_SOURCE_TYPE.get(label_tuple, source_type or "unknown")
        out.update(
            {
                "source_type": source_type,
                "domain": best_label.get("domain"),
                "field": best_label.get("field"),
                "subfield": best_label.get("subfield"),
                "confidence": float(best_score),
                "label_method": METHOD,
            }
        )
    return out


def print_summary(records):
    domain_counts = Counter(record.get("domain") for record in records)
    method_counts = Counter(record.get("label_method") for record in records)
    low_confidence_count = method_counts.get(LOW_CONFIDENCE_METHOD, 0)
    print(f"records_read: {len(records)}")
    print(f"records_written: {len(records)}")
    print(f"counts_by_domain: {json.dumps({str(k): v for k, v in domain_counts.items()}, ensure_ascii=False, sort_keys=True)}")
    print(f"counts_by_label_method: {json.dumps(dict(method_counts), ensure_ascii=False, sort_keys=True)}")
    print(f"low_confidence_count: {low_confidence_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.08)
    parser.add_argument("--text-chars", type=int, default=3000)
    parser.add_argument("--overwrite-existing-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    labels = load_labels(Path(args.labels))
    label_vectors = [vectorize(label_text(label)) for label in labels]

    if args.dry_run:
        print("DRY RUN")
        print(f"records: {len(records)}")
        print(f"labels: {len(labels)}")
        print(f"top_k: {args.top_k}")
        print("No output written.")
        return

    labeled = [
        classify_record(record, label_vectors, labels, args.min_confidence, args.text_chars, args.overwrite_existing_labels)
        for record in records
    ]
    write_jsonl(Path(args.output), labeled)
    print_summary(labeled)


if __name__ == "__main__":
    main()
