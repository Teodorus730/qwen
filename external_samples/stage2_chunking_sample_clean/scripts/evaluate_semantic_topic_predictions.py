"""Evaluate semantic_topic_domain predictions against cleaned v1-dev pseudo-gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_no}: {exc}") from exc
    return records


def prediction_payload(record: dict[str, Any], prediction_field: str) -> dict[str, Any]:
    payload = record.get(prediction_field)
    if not isinstance(payload, dict):
        raise ValueError(f"{record.get('chunk_id')}: missing prediction field {prediction_field!r}")
    return payload


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "support": tp + fn,
    }


def evaluate(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]], prediction_field: str, max_examples: int) -> dict[str, Any]:
    gold_by_id = {record["chunk_id"]: record for record in gold_records}
    pred_by_id = {record["chunk_id"]: record for record in pred_records}
    if set(gold_by_id) != set(pred_by_id):
        missing_pred = sorted(set(gold_by_id) - set(pred_by_id))[:10]
        extra_pred = sorted(set(pred_by_id) - set(gold_by_id))[:10]
        raise ValueError(f"Prediction/gold chunk_id mismatch; missing_pred={missing_pred}; extra_pred={extra_pred}")

    answered_count = 0
    abstained_count = 0
    correct_answered = 0
    strict_correct = 0
    top_k_contains_gold = 0
    top_k_available = 0
    evaluated_count = 0

    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    labels = sorted({record["semantic_topic_domain"] for record in gold_records})
    per_label_counts = {label: Counter() for label in labels}
    mismatches: list[dict[str, Any]] = []

    for chunk_id, gold in gold_by_id.items():
        pred_record = pred_by_id[chunk_id]
        payload = prediction_payload(pred_record, prediction_field)
        gold_abstained = bool(gold.get("semantic_topic_abstained"))
        if gold_abstained:
            continue

        evaluated_count += 1
        gold_label = gold["semantic_topic_domain"]
        pred_abstained = bool(payload.get("abstained"))
        pred_label = payload.get("domain") if not pred_abstained else None
        pred_for_confusion = pred_label if pred_label is not None else "__ABSTAIN__"
        dataset = str(gold.get("dataset"))

        if pred_abstained:
            abstained_count += 1
            by_dataset[dataset]["abstained"] += 1
        else:
            answered_count += 1
            by_dataset[dataset]["answered"] += 1
            if pred_label == gold_label:
                correct_answered += 1
                strict_correct += 1
                by_dataset[dataset]["correct"] += 1
            elif len(mismatches) < max_examples:
                mismatches.append(
                    {
                        "chunk_id": chunk_id,
                        "dataset": dataset,
                        "gold": gold_label,
                        "predicted": pred_label,
                        "confidence": payload.get("confidence"),
                        "margin": payload.get("margin"),
                        "text_preview": gold.get("text_preview", "")[:300],
                    }
                )

        if pred_abstained and len(mismatches) < max_examples:
            mismatches.append(
                {
                    "chunk_id": chunk_id,
                    "dataset": dataset,
                    "gold": gold_label,
                    "predicted": None,
                    "abstain_reason": payload.get("abstain_reason"),
                    "confidence": payload.get("confidence"),
                    "margin": payload.get("margin"),
                    "text_preview": gold.get("text_preview", "")[:300],
                }
            )

        top_k = payload.get("top_k")
        if isinstance(top_k, list):
            top_k_available += 1
            top_domains = [item.get("domain") for item in top_k if isinstance(item, dict)]
            if gold_label in top_domains:
                top_k_contains_gold += 1

        confusion[gold_label][pred_for_confusion] += 1
        for label in labels:
            if pred_label == label and gold_label == label:
                per_label_counts[label]["tp"] += 1
            elif pred_label == label and gold_label != label:
                per_label_counts[label]["fp"] += 1
            elif pred_label != label and gold_label == label:
                per_label_counts[label]["fn"] += 1

        by_dataset[dataset]["total"] += 1

    per_domain = {
        label: prf(counts["tp"], counts["fp"], counts["fn"])
        for label, counts in sorted(per_label_counts.items())
    }
    macro_f1 = safe_div(sum(item["f1"] for item in per_domain.values()), len(per_domain))

    per_dataset_metrics = {}
    for dataset, counts in sorted(by_dataset.items()):
        total = counts["total"]
        answered = counts["answered"]
        correct = counts["correct"]
        per_dataset_metrics[dataset] = {
            "records_total": total,
            "answered_count": answered,
            "abstained_count": counts["abstained"],
            "coverage_rate": round(safe_div(answered, total), 6),
            "accuracy_on_answered": round(safe_div(correct, answered), 6),
            "strict_accuracy": round(safe_div(correct, total), 6),
        }

    return {
        "records_total": evaluated_count,
        "gold_abstained_excluded": len(gold_records) - evaluated_count,
        "answered_count": answered_count,
        "abstained_count": abstained_count,
        "coverage_rate": round(safe_div(answered_count, evaluated_count), 6),
        "accuracy_on_answered": round(safe_div(correct_answered, answered_count), 6),
        "strict_accuracy": round(safe_div(strict_correct, evaluated_count), 6),
        "top_k_contains_gold": {
            "count": top_k_contains_gold,
            "available_count": top_k_available,
            "rate": round(safe_div(top_k_contains_gold, top_k_available), 6),
        },
        "macro_f1": round(macro_f1, 6),
        "per_domain_precision_recall": per_domain,
        "per_dataset_metrics": per_dataset_metrics,
        "confusion_matrix": {
            gold_label: dict(sorted(counter.items()))
            for gold_label, counter in sorted(confusion.items())
        },
        "mismatch_examples": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--prediction-field", default="semantic_topic_embedding")
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        gold_records = read_jsonl(args.gold)
        status = {
            "dry_run": True,
            "gold_records": len(gold_records),
            "predictions_exists": args.predictions.exists(),
            "would_write": str(args.output_json),
            "prediction_field": args.prediction_field,
        }
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    summary = evaluate(
        gold_records=read_jsonl(args.gold),
        pred_records=read_jsonl(args.predictions),
        prediction_field=args.prediction_field,
        max_examples=args.max_examples,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: summary[k] for k in ["records_total", "coverage_rate", "accuracy_on_answered", "strict_accuracy", "macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
