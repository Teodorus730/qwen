#!/usr/bin/env python3
"""Evaluate predicted chunk labels against pseudo-gold review labels."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LABEL_FIELDS = [
    ("source_type", "expected_source_type"),
    ("domain", "expected_domain"),
    ("field", "expected_field"),
    ("subfield", "expected_subfield"),
]


def iter_jsonl(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing file: {path}") from exc


def load_by_chunk_id(paths):
    records = {}
    for path in paths:
        for record in iter_jsonl(path):
            chunk_id = record.get("chunk_id")
            if chunk_id:
                records[chunk_id] = record
    return records


def accuracy(correct, total):
    if total == 0:
        return None
    return round(correct / total, 4)


def evaluate(predictions, gold_records, max_examples):
    totals = Counter()
    correct = Counter()
    full_correct = 0
    compared = 0
    missing_predictions = 0
    by_dataset = defaultdict(lambda: {"total": 0, "full_correct": 0, "correct": Counter()})
    mismatch_by_dataset = Counter()
    examples = []

    for chunk_id, gold in gold_records.items():
        pred = predictions.get(chunk_id)
        if pred is None:
            missing_predictions += 1
            continue

        compared += 1
        dataset = gold.get("dataset")
        by_dataset[dataset]["total"] += 1
        record_full_correct = True

        mismatched_fields = []
        for pred_field, gold_field in LABEL_FIELDS:
            totals[pred_field] += 1
            by_dataset[dataset]["correct"][pred_field] += int(pred.get(pred_field) == gold.get(gold_field))
            if pred.get(pred_field) == gold.get(gold_field):
                correct[pred_field] += 1
            else:
                record_full_correct = False
                mismatched_fields.append(pred_field)

        if record_full_correct:
            full_correct += 1
            by_dataset[dataset]["full_correct"] += 1
        else:
            mismatch_by_dataset[dataset] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "chunk_id": chunk_id,
                        "dataset": dataset,
                        "mismatched_fields": mismatched_fields,
                        "predicted": {field: pred.get(field) for field, _ in LABEL_FIELDS},
                        "expected": {field: gold.get(gold_field) for field, gold_field in LABEL_FIELDS},
                        "label_method": pred.get("label_method"),
                    }
                )

    metrics = {
        "records_in_gold": len(gold_records),
        "records_compared": compared,
        "missing_predictions": missing_predictions,
        "source_type_accuracy": accuracy(correct["source_type"], totals["source_type"]),
        "domain_accuracy": accuracy(correct["domain"], totals["domain"]),
        "field_accuracy": accuracy(correct["field"], totals["field"]),
        "subfield_accuracy": accuracy(correct["subfield"], totals["subfield"]),
        "full_label_accuracy": accuracy(full_correct, compared),
        "mismatches_by_dataset": dict(sorted(mismatch_by_dataset.items())),
        "per_dataset": {},
        "mismatch_examples": examples,
    }

    for dataset, values in sorted(by_dataset.items()):
        total = values["total"]
        metrics["per_dataset"][dataset] = {
            "records_compared": total,
            "source_type_accuracy": accuracy(values["correct"]["source_type"], total),
            "domain_accuracy": accuracy(values["correct"]["domain"], total),
            "field_accuracy": accuracy(values["correct"]["field"], total),
            "subfield_accuracy": accuracy(values["correct"]["subfield"], total),
            "full_label_accuracy": accuracy(values["full_correct"], total),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--pseudo-gold", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    predictions = load_by_chunk_id([Path(path) for path in args.predictions])
    gold_records = load_by_chunk_id([Path(args.pseudo_gold)])
    metrics = evaluate(predictions, gold_records, args.max_examples)

    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
