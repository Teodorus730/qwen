#!/usr/bin/env python
"""Evaluate weak annotation_v2 topic.domain predictions against pseudo-gold."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
    return rows


def topic(record: dict[str, Any]) -> dict[str, Any]:
    ann = record.get("annotation_v2")
    if not isinstance(ann, dict):
        return {}
    value = ann.get("topic")
    return value if isinstance(value, dict) else {}


def dataset(record: dict[str, Any]) -> str:
    return str(record.get("dataset") or "unknown")


def preview(record: dict[str, Any], chars: int = 260) -> str:
    text = record.get("text") or record.get("text_preview") or ""
    return " ".join(str(text).split())[:chars]


def empty_metrics() -> dict[str, Any]:
    return {
        "records_total": 0,
        "answered_count": 0,
        "abstained_count": 0,
        "correct_answered": 0,
        "correct_counting_abstain_wrong": 0,
        "gold_abstained_count": 0,
    }


def finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    total = metrics["records_total"]
    answered = metrics["answered_count"]
    metrics["coverage_rate"] = round(answered / total, 6) if total else 0.0
    metrics["accuracy_on_answered"] = round(metrics["correct_answered"] / answered, 6) if answered else 0.0
    metrics["accuracy_counting_abstain_wrong"] = (
        round(metrics["correct_counting_abstain_wrong"] / total, 6) if total else 0.0
    )
    return metrics


def add_case(metrics: dict[str, Any], gold_domain: str, pred_domain: str, pred_abstained: bool, gold_abstained: bool) -> None:
    metrics["records_total"] += 1
    if gold_abstained:
        metrics["gold_abstained_count"] += 1
    if pred_abstained:
        metrics["abstained_count"] += 1
    else:
        metrics["answered_count"] += 1
        if pred_domain == gold_domain:
            metrics["correct_answered"] += 1
            metrics["correct_counting_abstain_wrong"] += 1


def evaluate(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_by_id = {row.get("chunk_id"): row for row in predictions if row.get("chunk_id")}
    overall = empty_metrics()
    by_dataset: dict[str, dict[str, Any]] = defaultdict(empty_metrics)
    by_gold_domain: dict[str, dict[str, Any]] = defaultdict(empty_metrics)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches: list[dict[str, Any]] = []
    missing_predictions: list[str] = []

    matched_gold = 0
    for gold in gold_rows:
        chunk_id = gold.get("chunk_id")
        pred = pred_by_id.get(chunk_id)
        if pred is None:
            continue
        matched_gold += 1
        pred_topic = topic(pred)
        gold_domain = str(gold.get("review_topic_domain") or "unknown")
        pred_domain = str(pred_topic.get("domain") or "unknown")
        pred_abstained = bool(pred_topic.get("abstained"))
        gold_abstained = bool(gold.get("review_topic_abstained"))

        add_case(overall, gold_domain, pred_domain, pred_abstained, gold_abstained)
        add_case(by_dataset[dataset(gold)], gold_domain, pred_domain, pred_abstained, gold_abstained)
        add_case(by_gold_domain[gold_domain], gold_domain, pred_domain, pred_abstained, gold_abstained)
        confusion[gold_domain][pred_domain if not pred_abstained else "ABSTAIN"] += 1

        if pred_abstained or pred_domain != gold_domain:
            mismatches.append(
                {
                    "chunk_id": chunk_id,
                    "dataset": dataset(gold),
                    "gold_domain": gold_domain,
                    "gold_abstained": gold_abstained,
                    "predicted_domain": pred_domain,
                    "predicted_abstained": pred_abstained,
                    "abstain_reason": pred_topic.get("abstain_reason"),
                    "confidence": pred_topic.get("confidence"),
                    "top_k": pred_topic.get("top_k"),
                    "evidence": pred_topic.get("evidence"),
                    "text_preview": preview(gold),
                }
            )

    result = {
        "overall": finalize_metrics(overall),
        "by_dataset": {name: finalize_metrics(metrics) for name, metrics in sorted(by_dataset.items())},
        "by_gold_domain": {name: finalize_metrics(metrics) for name, metrics in sorted(by_gold_domain.items())},
        "confusion_matrix": {gold: dict(counter) for gold, counter in sorted(confusion.items())},
        "top_mismatch_examples": mismatches[:30],
        "missing_predictions": missing_predictions,
        "missing_predictions_count": len(missing_predictions),
        "gold_records_total": len(gold_rows),
        "matched_gold_records": matched_gold,
        "prediction_records_total": len(predictions),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate annotation_v2 topic.domain predictions.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL file(s).")
    parser.add_argument("--gold", required=True, help="Annotation v2 pseudo-gold JSONL.")
    parser.add_argument("--output-json", required=True, help="Output evaluation JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_jsonl([Path(path) for path in args.predictions])
    gold = load_jsonl([Path(args.gold)])
    result = evaluate(predictions, gold)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["missing_predictions_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
