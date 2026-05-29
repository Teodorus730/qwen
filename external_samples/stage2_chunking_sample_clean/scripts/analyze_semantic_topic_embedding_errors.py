"""Analyze semantic-topic embedding errors against cleaned dev pseudo-gold."""

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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def topk_position(top_k: list[dict[str, Any]], gold: str) -> int | None:
    for index, item in enumerate(top_k, start=1):
        if item.get("domain") == gold:
            return index
    return None


def compact_example(gold: dict[str, Any], pred: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = pred["semantic_topic_embedding"]
    return {
        "reason": reason,
        "chunk_id": gold["chunk_id"],
        "dataset": gold.get("dataset"),
        "gold_semantic_topic_domain": gold.get("semantic_topic_domain"),
        "predicted_domain": payload.get("domain"),
        "top_k": payload.get("top_k"),
        "confidence": payload.get("confidence"),
        "margin": payload.get("margin"),
        "text_preview": gold.get("text_preview", "")[:500],
        "genre_function": gold.get("genre_function"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--examples-jsonl", required=True, type=Path)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--low-margin-threshold", type=float, default=0.02)
    parser.add_argument("--max-examples-per-section", type=int, default=30)
    args = parser.parse_args()

    pred_by_id = {record["chunk_id"]: record for record in read_jsonl(args.predictions)}
    gold_by_id = {record["chunk_id"]: record for record in read_jsonl(args.gold)}
    if set(pred_by_id) != set(gold_by_id):
        raise SystemExit("Prediction/gold chunk_id sets differ")

    confusions = Counter()
    dataset_errors: dict[str, Counter[str]] = defaultdict(Counter)
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    topk_wrong = []
    high_conf_wrong = []
    low_margin_wrong = []
    gold_top2 = []
    gold_top3 = []
    all_examples = []

    evaluated = 0
    correct = 0
    topk_hit = 0
    wrong = 0

    for chunk_id, gold in gold_by_id.items():
        if gold.get("semantic_topic_abstained"):
            continue
        evaluated += 1
        pred = pred_by_id[chunk_id]
        payload = pred.get("semantic_topic_embedding") or {}
        gold_label = gold["semantic_topic_domain"]
        pred_label = payload.get("domain") if not payload.get("abstained") else "__ABSTAIN__"
        top_k = payload.get("top_k") or []
        confidence = float(payload.get("confidence") or 0.0)
        margin = float(payload.get("margin") or 0.0)
        position = topk_position(top_k, gold_label)

        if position is not None:
            topk_hit += 1

        domain_counts[gold_label]["support"] += 1
        if pred_label == gold_label:
            correct += 1
            domain_counts[gold_label]["tp"] += 1
        else:
            wrong += 1
            confusions[(gold_label, pred_label)] += 1
            dataset_errors[str(gold.get("dataset"))][f"{gold_label}->{pred_label}"] += 1
            domain_counts[gold_label]["fn"] += 1
            domain_counts[str(pred_label)]["fp"] += 1

            if position is not None:
                item = compact_example(gold, pred, "gold_in_top_k_but_top1_wrong")
                item["gold_top_k_position"] = position
                topk_wrong.append(item)
                if position == 2:
                    gold_top2.append(item)
                elif position == 3:
                    gold_top3.append(item)
            if confidence >= args.high_confidence_threshold:
                high_conf_wrong.append(compact_example(gold, pred, "high_confidence_wrong"))
            if margin <= args.low_margin_threshold:
                low_margin_wrong.append(compact_example(gold, pred, "low_margin_wrong"))
            all_examples.append(compact_example(gold, pred, "wrong_top1"))

    per_domain = {}
    for label, counts in sorted(domain_counts.items()):
        support = counts["support"]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_domain[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(safe_div(2 * precision * recall, precision + recall), 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    examples = []
    for section in [topk_wrong, high_conf_wrong, low_margin_wrong, gold_top2, gold_top3, all_examples]:
        examples.extend(section[: args.max_examples_per_section])

    summary = {
        "predictions": str(args.predictions),
        "gold": str(args.gold),
        "records_evaluated": evaluated,
        "correct_top1": correct,
        "wrong_top1": wrong,
        "top1_accuracy": round(safe_div(correct, evaluated), 6),
        "top_k_contains_gold": {
            "count": topk_hit,
            "rate": round(safe_div(topk_hit, evaluated), 6),
        },
        "top_confusions": [
            {"gold": gold, "predicted": pred, "count": count}
            for (gold, pred), count in confusions.most_common(30)
        ],
        "top_k_contains_gold_but_top1_wrong_count": len(topk_wrong),
        "gold_top2_count": len(gold_top2),
        "gold_top3_count": len(gold_top3),
        "high_confidence_wrong": {
            "threshold": args.high_confidence_threshold,
            "count": len(high_conf_wrong),
            "examples": high_conf_wrong[: args.max_examples_per_section],
        },
        "low_margin_wrong": {
            "threshold": args.low_margin_threshold,
            "count": len(low_margin_wrong),
            "examples": low_margin_wrong[: args.max_examples_per_section],
        },
        "per_dataset_error_clusters": {
            dataset: [
                {"confusion": confusion, "count": count}
                for confusion, count in counter.most_common(15)
            ]
            for dataset, counter in sorted(dataset_errors.items())
        },
        "per_domain_precision_recall_highlights": per_domain,
        "example_counts": {
            "examples_jsonl": len(examples),
            "wrong_top1_examples_total": len(all_examples),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_jsonl(args.examples_jsonl, examples)
    print(json.dumps({k: summary[k] for k in ["records_evaluated", "top1_accuracy", "top_k_contains_gold", "top_k_contains_gold_but_top1_wrong_count"]}, indent=2))


if __name__ == "__main__":
    main()
