#!/usr/bin/env python3
"""Build compact real benchmark review candidates for pseudo-gold labeling."""

import argparse
import json
import random
from collections import defaultdict
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


def load_optional_by_chunk_id(path):
    if path is None:
        return {}
    return {record.get("chunk_id"): record for record in iter_jsonl(Path(path)) if record.get("chunk_id")}


def preview(text, chars):
    flat = " ".join(str(text or "").split())
    if len(flat) <= chars:
        return flat
    return flat[: max(chars - 3, 0)] + "..."


def select_records(records, max_records, per_dataset, seed):
    rng = random.Random(seed)
    if per_dataset:
        by_dataset = defaultdict(list)
        for record in records:
            by_dataset[record.get("dataset")].append(record)
        selected = []
        for dataset in sorted(by_dataset, key=lambda value: str(value)):
            group = by_dataset[dataset]
            rng.shuffle(group)
            selected.extend(group[:per_dataset])
        rng.shuffle(selected)
        if max_records:
            selected = selected[:max_records]
        return selected

    selected = list(records)
    rng.shuffle(selected)
    if max_records:
        selected = selected[:max_records]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rule-based")
    parser.add_argument("--embedding")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--per-dataset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-chars", type=int, default=700)
    args = parser.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    selected = select_records(records, args.max_records, args.per_dataset, args.seed)
    rule_records = load_optional_by_chunk_id(args.rule_based)
    embedding_records = load_optional_by_chunk_id(args.embedding)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for index, record in enumerate(selected):
            chunk_id = record.get("chunk_id")
            out = {
                "review_id": f"review_{index:06d}",
                "chunk_id": chunk_id,
                "dataset": record.get("dataset"),
                "source_type_pred": record.get("source_type"),
                "domain_pred": record.get("domain"),
                "field_pred": record.get("field"),
                "subfield_pred": record.get("subfield"),
                "confidence_pred": record.get("confidence"),
                "label_method": record.get("label_method"),
                "text_preview": preview(record.get("text"), args.text_chars),
                "text": record.get("text"),
                "expected_source_type": None,
                "expected_domain": None,
                "expected_field": None,
                "expected_subfield": None,
                "review_note": None,
                "review_confidence": None,
            }
            if chunk_id in rule_records:
                out["rule_based_label"] = {
                    "source_type": rule_records[chunk_id].get("source_type"),
                    "domain": rule_records[chunk_id].get("domain"),
                    "field": rule_records[chunk_id].get("field"),
                    "subfield": rule_records[chunk_id].get("subfield"),
                    "confidence": rule_records[chunk_id].get("confidence"),
                    "label_method": rule_records[chunk_id].get("label_method"),
                }
            if chunk_id in embedding_records:
                out["embedding_label"] = {
                    "source_type": embedding_records[chunk_id].get("source_type"),
                    "domain": embedding_records[chunk_id].get("domain"),
                    "field": embedding_records[chunk_id].get("field"),
                    "subfield": embedding_records[chunk_id].get("subfield"),
                    "confidence": embedding_records[chunk_id].get("confidence"),
                    "label_method": embedding_records[chunk_id].get("label_method"),
                    "low_confidence": embedding_records[chunk_id].get("low_confidence"),
                    "top_k_labels": embedding_records[chunk_id].get("top_k_labels"),
                }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"records_read: {len(records)}")
    print(f"records_written: {len(selected)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
