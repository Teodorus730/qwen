"""Validate unified Stage2 handoff annotation JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def validate_file(path: Path, *, expected_min: int, expected_max: int) -> dict[str, Any]:
    errors: list[str] = []
    records = read_jsonl(path)
    ids = [record.get("chunk_id") for record in records]
    duplicates = sorted([chunk_id for chunk_id, count in Counter(ids).items() if count > 1])

    if not (expected_min <= len(records) <= expected_max):
        errors.append(f"record count {len(records)} outside expected range {expected_min}-{expected_max}")
    if duplicates:
        errors.append(f"duplicate chunk_id values: {duplicates[:10]}")

    required_top = {
        "chunk_id",
        "dataset",
        "provenance",
        "annotation_v2",
        "tokenization_stats",
        "surface",
        "quality",
        "handoff_caveats",
    }
    for index, record in enumerate(records):
        missing = sorted(required_top - set(record))
        if missing:
            errors.append(f"record {index} missing top-level fields: {missing}")
            continue
        annotation_v2 = record.get("annotation_v2") or {}
        if not annotation_v2.get("text_stats"):
            errors.append(f"{record.get('chunk_id')}: missing annotation_v2.text_stats")
        if not annotation_v2.get("surface"):
            errors.append(f"{record.get('chunk_id')}: missing annotation_v2.surface")
        if not annotation_v2.get("quality"):
            errors.append(f"{record.get('chunk_id')}: missing annotation_v2.quality")
        if not (annotation_v2.get("tokenizer") or (record.get("tokenization_stats") or {}).get("tokenizer")):
            errors.append(f"{record.get('chunk_id')}: missing tokenizer stats")
        caveats = record.get("handoff_caveats") or {}
        if "do_not_claim_held_out_cleaned_semantic_quality" not in caveats:
            errors.append(f"{record.get('chunk_id')}: missing handoff caveat flag")

    return {
        "path": str(path),
        "records_count": len(records),
        "duplicate_chunk_id_count": len(duplicates),
        "datasets": dict(sorted(Counter(record.get("dataset") for record in records).items())),
        "valid": not errors,
        "errors": errors[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=Path("data_samples/handoff/stage2_v1_dev_annotations.jsonl"))
    parser.add_argument("--v2", type=Path, default=Path("data_samples/handoff/stage2_v2_test_annotations.jsonl"))
    parser.add_argument("--output-json", type=Path, default=Path("data_samples/handoff/stage2_handoff_validation_summary.json"))
    args = parser.parse_args()

    summary = {
        "output_granularity": "chunk_level_full_tokenized_features",
        "expected_counts_note": "Reviewed pseudo-gold subsets are about 120 v1-dev and 90 v2-test records; unified handoff outputs are chunk-level joins over all available tokenized feature records.",
        "v1_dev": validate_file(args.v1, expected_min=120, expected_max=10000),
        "v2_test": validate_file(args.v2, expected_min=90, expected_max=10000),
    }
    summary["valid"] = summary["v1_dev"]["valid"] and summary["v2_test"]["valid"]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
