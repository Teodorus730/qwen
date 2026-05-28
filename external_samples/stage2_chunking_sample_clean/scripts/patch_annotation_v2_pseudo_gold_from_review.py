#!/usr/bin/env python
"""Patch annotation_v2 pseudo-gold topic labels from manual review decisions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PATCH_SOURCE = "weak_topic_domain_v2_manual_review"
PATCH_ACTIONS = {"pseudo_gold_patch"}
PATCH_DECISIONS = {"revise_gold_domain", "mark_gold_abstained"}
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
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def validate_unique_chunk_ids(rows: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"{label}: missing or invalid chunk_id")
        if chunk_id in seen:
            duplicates.append(chunk_id)
        seen.add(chunk_id)
    if duplicates:
        raise ValueError(f"{label}: duplicate chunk_id values: {duplicates[:10]}")


def should_patch(review: dict[str, Any]) -> bool:
    return (
        review.get("recommended_action") in PATCH_ACTIONS
        or review.get("manual_review_decision") in PATCH_DECISIONS
    )


def append_patch_note(original_note: Any, patch_note: str) -> str:
    base = str(original_note).strip() if original_note is not None else ""
    marker = f"pseudo_gold_patch: {patch_note}"
    if not base:
        return marker
    if marker in base:
        return base
    return f"{base}; {marker}"


def patch_rows(gold_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_unique_chunk_ids(gold_rows, "gold")
    validate_unique_chunk_ids(review_rows, "review")
    review_by_id = {row["chunk_id"]: row for row in review_rows}
    patched_rows: list[dict[str, Any]] = []
    patched_ids: list[str] = []
    unexpected_patch_ids: list[str] = []
    domain_changes: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    marked_abstained: list[str] = []

    for row in gold_rows:
        chunk_id = row["chunk_id"]
        review = review_by_id.get(chunk_id)
        output = dict(row)
        output["pseudo_gold_patch_applied"] = False
        output["pseudo_gold_patch_reason"] = None
        output["pseudo_gold_patch_source"] = PATCH_SOURCE

        if review and should_patch(review):
            old_domain = str(output.get("review_topic_domain") or "unknown")
            old_abstained = bool(output.get("review_topic_abstained"))
            new_domain = str(review.get("revised_topic_domain") or "unknown")
            if new_domain not in ALLOWED_DOMAINS:
                raise ValueError(f"{chunk_id}: invalid revised_topic_domain={new_domain!r}")
            new_abstained = bool(review.get("revised_topic_abstained"))
            decision = str(review.get("manual_review_decision"))
            action = str(review.get("recommended_action"))
            reason = str(review.get("revised_review_note") or "").strip()
            if not reason:
                raise ValueError(f"{chunk_id}: patch review note is empty")

            if decision == "revise_gold_domain":
                new_abstained = bool(review.get("revised_topic_abstained"))
            elif decision == "mark_gold_abstained":
                new_abstained = True
                if not new_domain:
                    new_domain = "unknown"

            output["review_topic_domain"] = new_domain
            output["review_topic_abstained"] = new_abstained
            output["review_topic_note"] = append_patch_note(output.get("review_topic_note"), reason)
            output["review_note"] = append_patch_note(output.get("review_note"), reason)
            output["pseudo_gold_patch_applied"] = True
            output["pseudo_gold_patch_reason"] = reason
            output["pseudo_gold_patch_source"] = PATCH_SOURCE

            patched_ids.append(chunk_id)
            action_counts[action] += 1
            decision_counts[decision] += 1
            domain_changes[f"{old_domain}->{new_domain}"] += 1
            if (not old_abstained) and new_abstained:
                marked_abstained.append(chunk_id)
        patched_rows.append(output)

    gold_ids = {row["chunk_id"] for row in gold_rows}
    for review in review_rows:
        if should_patch(review) and review["chunk_id"] not in gold_ids:
            unexpected_patch_ids.append(review["chunk_id"])
    if unexpected_patch_ids:
        raise ValueError(f"Review requested patches for unknown gold chunk_ids: {unexpected_patch_ids[:10]}")

    old_distribution = Counter(str(row.get("review_topic_domain") or "unknown") for row in gold_rows)
    patched_distribution = Counter(str(row.get("review_topic_domain") or "unknown") for row in patched_rows)
    old_abstained = sum(1 for row in gold_rows if row.get("review_topic_abstained") is True)
    patched_abstained = sum(1 for row in patched_rows if row.get("review_topic_abstained") is True)
    summary = {
        "records_total": len(gold_rows),
        "records_patched": len(patched_ids),
        "records_unchanged": len(gold_rows) - len(patched_ids),
        "patched_chunk_ids": patched_ids,
        "domain_changes": dict(domain_changes),
        "manual_review_decisions_applied": dict(decision_counts),
        "recommended_actions_applied": dict(action_counts),
        "records_marked_abstained": len(marked_abstained),
        "marked_abstained_chunk_ids": marked_abstained,
        "old_topic_domain_distribution": dict(old_distribution),
        "patched_topic_domain_distribution": dict(patched_distribution),
        "old_abstained_count": old_abstained,
        "patched_abstained_count": patched_abstained,
        "patch_source": PATCH_SOURCE,
    }
    return patched_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch annotation_v2 pseudo-gold from manual review labels.")
    parser.add_argument("--gold", required=True, help="Original annotation_v2 pseudo-gold JSONL.")
    parser.add_argument("--review", required=True, help="Manual review labeled JSONL.")
    parser.add_argument("--output", required=True, help="Patched pseudo-gold output JSONL.")
    parser.add_argument("--summary-json", required=True, help="Patch summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_rows = load_jsonl(Path(args.gold))
    review_rows = load_jsonl(Path(args.review))
    patched_rows, summary = patch_rows(gold_rows, review_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in patched_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

