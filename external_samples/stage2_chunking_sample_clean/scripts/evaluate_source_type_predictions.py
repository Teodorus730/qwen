#!/usr/bin/env python3
"""Evaluate source_type predictions against pseudo-gold labels."""

import argparse
import json
from collections import Counter, defaultdict
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
                    raise SystemExit(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing file: {path}") from exc


def load_by_chunk_id(paths):
    records = {}
    duplicates = []
    for path in paths:
        for record in iter_jsonl(path):
            chunk_id = record.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id in records:
                duplicates.append(chunk_id)
            records[chunk_id] = record
    return records, duplicates


def accuracy(correct, total):
    if total == 0:
        return None
    return round(correct / total, 4)


def evaluate(predictions, gold_records, max_examples):
    compared = 0
    correct = 0
    missing_predictions = 0
    confusion = Counter()
    pred_dist = Counter()
    expected_dist = Counter()
    by_dataset = defaultdict(lambda: [0, 0])
    by_expected = defaultdict(lambda: [0, 0])
    mismatch_examples = []
    low_confidence = 0
    method_counts = Counter()

    for chunk_id, gold in gold_records.items():
        expected = gold.get("expected_source_type")
        expected_dist[expected] += 1
        pred = predictions.get(chunk_id)
        if pred is None:
            missing_predictions += 1
            continue
        predicted = pred.get("source_type")
        method = pred.get("source_type_method") or pred.get("label_method")
        compared += 1
        pred_dist[predicted] += 1
        method_counts[method] += 1
        confusion[(expected, predicted)] += 1
        by_dataset[gold.get("dataset")][1] += 1
        by_expected[expected][1] += 1
        if str(method or "").endswith("low_confidence"):
            low_confidence += 1
        if predicted == expected:
            correct += 1
            by_dataset[gold.get("dataset")][0] += 1
            by_expected[expected][0] += 1
        elif len(mismatch_examples) < max_examples:
            mismatch_examples.append(
                {
                    "chunk_id": chunk_id,
                    "dataset": gold.get("dataset"),
                    "expected_source_type": expected,
                    "predicted_source_type": predicted,
                    "method": method,
                    "text_preview": " ".join(str(gold.get("text") or "").split())[:300],
                }
            )

    top_mismatches = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in confusion.most_common()
        if expected != predicted
    ]
    return {
        "records_in_gold": len(gold_records),
        "records_compared": compared,
        "missing_predictions": missing_predictions,
        "source_type_accuracy": accuracy(correct, compared),
        "low_confidence_count": low_confidence,
        "expected_source_type_distribution": dict(sorted(expected_dist.items(), key=lambda item: str(item[0]))),
        "predicted_source_type_distribution": dict(sorted(pred_dist.items(), key=lambda item: str(item[0]))),
        "method_counts": dict(sorted(method_counts.items(), key=lambda item: str(item[0]))),
        "per_dataset_accuracy": {
            dataset: accuracy(values[0], values[1])
            for dataset, values in sorted(by_dataset.items(), key=lambda item: str(item[0]))
        },
        "accuracy_by_expected_source_type": {
            expected: accuracy(values[0], values[1])
            for expected, values in sorted(by_expected.items(), key=lambda item: str(item[0]))
        },
        "confusion_matrix": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
        ],
        "top_mismatch_groups": top_mismatches[:25],
        "mismatch_examples": mismatch_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--pseudo-gold", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    predictions, prediction_duplicates = load_by_chunk_id([Path(path) for path in args.predictions])
    gold_records, gold_duplicates = load_by_chunk_id([Path(args.pseudo_gold)])
    if prediction_duplicates or gold_duplicates:
        raise SystemExit(
            f"Duplicate chunk_id values found. predictions={prediction_duplicates[:5]} gold={gold_duplicates[:5]}"
        )

    metrics = evaluate(predictions, gold_records, args.max_examples)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
