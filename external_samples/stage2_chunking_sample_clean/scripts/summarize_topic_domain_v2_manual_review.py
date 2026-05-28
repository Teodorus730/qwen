#!/usr/bin/env python
"""Summarize targeted manual review decisions for weak topic.domain v2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "review_id",
    "chunk_id",
    "dataset",
    "error_group",
    "gold_review_topic_domain",
    "gold_review_topic_abstained",
    "prediction_domain",
    "prediction_confidence",
    "prediction_abstained",
    "manual_review_decision",
    "revised_topic_domain",
    "revised_topic_abstained",
    "revised_review_note",
    "recommended_action",
]

ALLOWED_DECISIONS = {
    "keep_gold",
    "revise_gold_domain",
    "mark_gold_abstained",
    "classifier_error",
    "ambiguous_keep_for_embedding_bakeoff",
}

ALLOWED_ACTIONS = {
    "no_change",
    "v2_1_rule_fix",
    "v2_1_abstain_policy",
    "pseudo_gold_patch",
    "embedding_bakeoff_later",
}

ALLOWED_DOMAINS = {
    "stem",
    "science",
    "technology",
    "software",
    "humanities",
    "social_sciences",
    "commercial",
    "government",
    "media",
    "reference",
    "education",
    "unknown",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def sample_examples(rows: list[dict[str, Any]], field: str, limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field))
        if len(examples[key]) >= limit:
            continue
        examples[key].append(
            {
                "review_id": row.get("review_id"),
                "chunk_id": row.get("chunk_id"),
                "dataset": row.get("dataset"),
                "error_group": row.get("error_group"),
                "gold_domain": row.get("gold_review_topic_domain"),
                "predicted_domain": row.get("prediction_domain"),
                "revised_topic_domain": row.get("revised_topic_domain"),
                "revised_topic_abstained": row.get("revised_topic_abstained"),
                "recommended_action": row.get("recommended_action"),
                "review_note": row.get("revised_review_note"),
                "text_preview": row.get("text_preview", "")[:280],
            }
        )
    return dict(examples)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    chunk_ids: set[str] = set()
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        label = row.get("review_id") or f"row_{index}"
        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"{label}: missing {field}")
        chunk_id = row.get("chunk_id")
        if isinstance(chunk_id, str):
            if chunk_id in chunk_ids:
                duplicates.append(chunk_id)
            chunk_ids.add(chunk_id)
        decision = row.get("manual_review_decision")
        if decision not in ALLOWED_DECISIONS:
            errors.append(f"{label}: invalid manual_review_decision={decision!r}")
        action = row.get("recommended_action")
        if action not in ALLOWED_ACTIONS:
            errors.append(f"{label}: invalid recommended_action={action!r}")
        domain = row.get("revised_topic_domain")
        if domain not in ALLOWED_DOMAINS:
            errors.append(f"{label}: invalid revised_topic_domain={domain!r}")
        if not isinstance(row.get("revised_topic_abstained"), bool):
            errors.append(f"{label}: revised_topic_abstained must be boolean")
        note = row.get("revised_review_note")
        if not isinstance(note, str) or not note.strip():
            errors.append(f"{label}: revised_review_note must be non-empty string")

    by_error_group = Counter(str(row.get("error_group")) for row in rows)
    by_decision = Counter(str(row.get("manual_review_decision")) for row in rows)
    by_action = Counter(str(row.get("recommended_action")) for row in rows)
    revised_domain_distribution = Counter(str(row.get("revised_topic_domain")) for row in rows)
    revised_abstained = sum(1 for row in rows if row.get("revised_topic_abstained") is True)
    gold_changed = sum(
        1
        for row in rows
        if row.get("revised_topic_domain") != row.get("gold_review_topic_domain")
        or row.get("revised_topic_abstained") != row.get("gold_review_topic_abstained")
    )
    return {
        "candidates_count": len(rows),
        "by_error_group": dict(by_error_group),
        "by_manual_review_decision": dict(by_decision),
        "by_recommended_action": dict(by_action),
        "pseudo_gold_patch_count": by_action.get("pseudo_gold_patch", 0),
        "v2_1_rule_fix_count": by_action.get("v2_1_rule_fix", 0),
        "v2_1_abstain_policy_count": by_action.get("v2_1_abstain_policy", 0),
        "embedding_bakeoff_later_count": by_action.get("embedding_bakeoff_later", 0),
        "revised_domain_distribution": dict(revised_domain_distribution),
        "revised_abstained_count": revised_abstained,
        "gold_changed_count": gold_changed,
        "duplicates": duplicates,
        "integrity_errors": errors,
        "examples_by_decision": sample_examples(rows, "manual_review_decision"),
        "examples_by_action": sample_examples(rows, "recommended_action"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize topic.domain v2 manual review decisions.")
    parser.add_argument("--input", required=True, help="Manual review labeled JSONL.")
    parser.add_argument("--output-json", required=True, help="Output summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    summary = summarize(rows)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["integrity_errors"] or summary["duplicates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

