#!/usr/bin/env python3
"""Build hybrid labels from rule-based source_type and embedding domain labels."""

import argparse
import json
from collections import Counter
from pathlib import Path


HYBRID_MINILM_METHOD = "hybrid_rule_source_type_minilm_domain"
HYBRID_FALLBACK_METHOD = "hybrid_rule_fallback_low_confidence"
LABEL_FIELDS = ("domain", "field", "subfield")


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


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_by_chunk_id(path):
    records = {}
    for record in iter_jsonl(path):
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            print(f"Record without chunk_id in {path}")
            raise SystemExit(2)
        if chunk_id in records:
            print(f"Duplicate chunk_id in {path}: {chunk_id}")
            raise SystemExit(2)
        records[chunk_id] = record
    return records


def has_embedding_label(record):
    return all(record.get(field) not in (None, "") for field in LABEL_FIELDS)


def build_hybrid(rule_records, embedding_records, min_embedding_confidence):
    output = []
    missing_embedding = []
    for chunk_id in sorted(rule_records):
        rule = rule_records[chunk_id]
        embedding = embedding_records.get(chunk_id)
        if embedding is None:
            missing_embedding.append(chunk_id)

        out = dict(rule)
        use_embedding = False
        if embedding is not None:
            confidence = embedding.get("confidence")
            use_embedding = (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and confidence >= min_embedding_confidence
                and has_embedding_label(embedding)
            )

        if use_embedding:
            out.update(
                {
                    "source_type": rule.get("source_type"),
                    "domain": embedding.get("domain"),
                    "field": embedding.get("field"),
                    "subfield": embedding.get("subfield"),
                    "confidence": float(embedding.get("confidence")),
                    "label_method": HYBRID_MINILM_METHOD,
                    "hybrid_source_type_method": rule.get("label_method"),
                    "hybrid_domain_method": embedding.get("label_method"),
                    "embedding_confidence": float(embedding.get("confidence")),
                    "embedding_low_confidence": embedding.get("low_confidence"),
                    "embedding_model": embedding.get("embedding_model"),
                    "embedding_top_k_labels": embedding.get("top_k_labels"),
                }
            )
        else:
            out.update(
                {
                    "source_type": rule.get("source_type"),
                    "domain": rule.get("domain"),
                    "field": rule.get("field"),
                    "subfield": rule.get("subfield"),
                    "confidence": rule.get("confidence"),
                    "label_method": HYBRID_FALLBACK_METHOD,
                    "hybrid_source_type_method": rule.get("label_method"),
                    "hybrid_domain_method": rule.get("label_method"),
                    "embedding_confidence": embedding.get("confidence") if embedding else None,
                    "embedding_low_confidence": embedding.get("low_confidence") if embedding else None,
                    "embedding_model": embedding.get("embedding_model") if embedding else None,
                    "embedding_top_k_labels": embedding.get("top_k_labels") if embedding else None,
                }
            )
        output.append(out)

    extra_embedding = sorted(set(embedding_records) - set(rule_records))
    return output, missing_embedding, extra_embedding


def print_summary(records, missing_embedding, extra_embedding):
    method_counts = Counter(record.get("label_method") for record in records)
    domain_counts = Counter(record.get("domain") for record in records)
    print(f"records_written: {len(records)}")
    print(f"counts_by_label_method: {json.dumps(dict(method_counts), ensure_ascii=False, sort_keys=True)}")
    print(f"counts_by_domain: {json.dumps({str(k): v for k, v in domain_counts.items()}, ensure_ascii=False, sort_keys=True)}")
    print(f"missing_embedding_records: {len(missing_embedding)}")
    print(f"extra_embedding_records: {len(extra_embedding)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-based", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-embedding-confidence", type=float, default=0.35)
    args = parser.parse_args()

    rule_records = load_by_chunk_id(Path(args.rule_based))
    embedding_records = load_by_chunk_id(Path(args.embedding))
    records, missing_embedding, extra_embedding = build_hybrid(
        rule_records=rule_records,
        embedding_records=embedding_records,
        min_embedding_confidence=args.min_embedding_confidence,
    )
    write_jsonl(Path(args.output), records)
    print_summary(records, missing_embedding, extra_embedding)
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
