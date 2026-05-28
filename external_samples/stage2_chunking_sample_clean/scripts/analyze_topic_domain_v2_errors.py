#!/usr/bin/env python
"""Analyze weak topic.domain v2 errors against annotation_v2 pseudo-gold."""

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
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
    return records


def annotation(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("annotation_v2")
    return value if isinstance(value, dict) else {}


def topic(record: dict[str, Any]) -> dict[str, Any]:
    value = annotation(record).get("topic")
    return value if isinstance(value, dict) else {}


def preview(record: dict[str, Any], chars: int = 320) -> str:
    text = record.get("text") or record.get("text_preview") or ""
    return " ".join(str(text).split())[:chars]


def evidence_items(pred_topic: dict[str, Any]) -> list[str]:
    evidence = pred_topic.get("evidence")
    if not isinstance(evidence, dict):
        return []
    items: list[str] = []
    for domain, values in evidence.items():
        if isinstance(values, list):
            for value in values:
                items.append(f"{domain}:{value}")
    return items


def evidence_cluster(pred_topic: dict[str, Any]) -> list[str]:
    clusters: list[str] = []
    items = evidence_items(pred_topic)
    if any("FineMath weak provenance prior" in item or "FineWeb-Edu weak provenance prior" in item for item in items):
        clusters.append("provenance_prior")
    if any("surface." in item or "token_per_byte" in item for item in items):
        clusters.append("surface_feature")
    if any("keyword:" in item for item in items):
        clusters.append("keyword")
    top_k = pred_topic.get("top_k")
    if isinstance(top_k, list) and len(top_k) >= 2:
        top = top_k[0].get("score")
        second = top_k[1].get("score")
        if isinstance(top, (int, float)) and isinstance(second, (int, float)) and top - second < 0.6:
            clusters.append("low_margin")
    if pred_topic.get("abstained"):
        clusters.append("abstain")
    return clusters or ["no_evidence"]


def example_record(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    pred_topic = topic(pred)
    return {
        "chunk_id": gold.get("chunk_id"),
        "dataset": gold.get("dataset"),
        "gold_domain": gold.get("review_topic_domain"),
        "gold_abstained": gold.get("review_topic_abstained"),
        "predicted_domain": pred_topic.get("domain"),
        "predicted_abstained": pred_topic.get("abstained"),
        "confidence": pred_topic.get("confidence"),
        "abstain_reason": pred_topic.get("abstain_reason"),
        "top_k": pred_topic.get("top_k"),
        "evidence": pred_topic.get("evidence"),
        "evidence_clusters": evidence_cluster(pred_topic),
        "review_note": gold.get("review_note"),
        "text_preview": preview(gold),
    }


def precision_recall(cases: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    domains = sorted({case["gold"] for case in cases} | {case["pred"] for case in cases if case["pred"] != "ABSTAIN"})
    out: dict[str, dict[str, float | int]] = {}
    for domain in domains:
        tp = sum(1 for case in cases if case["gold"] == domain and case["pred"] == domain)
        fp = sum(1 for case in cases if case["gold"] != domain and case["pred"] == domain)
        fn = sum(1 for case in cases if case["gold"] == domain and case["pred"] != domain)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[domain] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        }
    return out


def analyze(predictions: list[dict[str, Any]], gold_records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_by_id = {record.get("chunk_id"): record for record in predictions if record.get("chunk_id")}
    totals = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    mismatch_groups: Counter[str] = Counter()
    abstain_by_gold: Counter[str] = Counter()
    per_dataset_mismatch: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_clusters: Counter[str] = Counter()
    keyword_evidence: Counter[str] = Counter()
    high_conf_wrong: list[dict[str, Any]] = []
    low_conf_correct: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    cases_for_pr: list[dict[str, str]] = []

    for gold in gold_records:
        pred = pred_by_id.get(gold.get("chunk_id"))
        if pred is None:
            continue
        pred_topic = topic(pred)
        gold_domain = str(gold.get("review_topic_domain") or "unknown")
        pred_domain = str(pred_topic.get("domain") or "unknown")
        pred_abstained = bool(pred_topic.get("abstained"))
        pred_label = "ABSTAIN" if pred_abstained else pred_domain
        confidence = pred_topic.get("confidence")
        is_correct = (not pred_abstained) and pred_domain == gold_domain

        totals["total"] += 1
        totals["answered" if not pred_abstained else "abstained"] += 1
        totals["correct" if is_correct else "incorrect"] += 1
        if gold.get("review_topic_abstained"):
            totals["gold_abstained"] += 1
        confusion[gold_domain][pred_label] += 1
        cases_for_pr.append({"gold": gold_domain, "pred": pred_label})

        for cluster in evidence_cluster(pred_topic):
            evidence_clusters[cluster] += 1
        for item in evidence_items(pred_topic):
            if "keyword:" in item:
                keyword_evidence[item.split("keyword:", 1)[1].split(" (+", 1)[0]] += 1

        if not is_correct:
            group_key = f"{gold_domain}->{pred_label}"
            mismatch_groups[group_key] += 1
            per_dataset_mismatch[str(gold.get("dataset"))][group_key] += 1
            if pred_abstained:
                abstain_by_gold[gold_domain] += 1
            ex = example_record(gold, pred)
            examples.append(ex)
            if isinstance(confidence, (int, float)) and confidence >= 0.65 and not pred_abstained:
                high_conf_wrong.append(ex)
        elif isinstance(confidence, (int, float)) and confidence <= 0.45:
            low_conf_correct.append(example_record(gold, pred))

    analysis = {
        "total_records": totals["total"],
        "answered": totals["answered"],
        "abstained": totals["abstained"],
        "correct": totals["correct"],
        "incorrect": totals["incorrect"],
        "gold_abstained": totals["gold_abstained"],
        "coverage_rate": round(totals["answered"] / totals["total"], 6) if totals["total"] else 0.0,
        "accuracy_on_answered": round(totals["correct"] / totals["answered"], 6) if totals["answered"] else 0.0,
        "accuracy_counting_abstain_wrong": round(totals["correct"] / totals["total"], 6) if totals["total"] else 0.0,
        "confusion_matrix": {domain: dict(counter) for domain, counter in sorted(confusion.items())},
        "top_mismatch_groups": dict(mismatch_groups.most_common(20)),
        "abstain_groups_by_gold_domain": dict(abstain_by_gold.most_common()),
        "per_dataset_mismatch_groups": {
            dataset: dict(counter.most_common(15)) for dataset, counter in sorted(per_dataset_mismatch.items())
        },
        "per_domain_precision_recall": precision_recall(cases_for_pr),
        "evidence_cluster_counts": dict(evidence_clusters.most_common()),
        "top_keyword_evidence": dict(keyword_evidence.most_common(25)),
        "high_confidence_wrong_examples": high_conf_wrong[:20],
        "low_confidence_correct_examples": low_conf_correct[:20],
        "gold_unknown_or_abstained_examples": [
            example_record(gold, pred_by_id[gold.get("chunk_id")])
            for gold in gold_records
            if gold.get("chunk_id") in pred_by_id
            and (gold.get("review_topic_domain") == "unknown" or gold.get("review_topic_abstained"))
        ],
    }
    return analysis, examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze topic.domain v2 prediction errors.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL files.")
    parser.add_argument("--gold", required=True, help="Annotation v2 pseudo-gold JSONL.")
    parser.add_argument("--output-json", required=True, help="Output analysis JSON.")
    parser.add_argument("--examples-jsonl", required=True, help="Output mismatch examples JSONL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_jsonl([Path(path) for path in args.predictions])
    gold = load_jsonl([Path(args.gold)])
    analysis, examples = analyze(predictions, gold)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    examples_path = Path(args.examples_jsonl)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with examples_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
