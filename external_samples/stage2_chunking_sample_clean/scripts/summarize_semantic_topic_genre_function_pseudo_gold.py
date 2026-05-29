"""Summarize and validate semantic topic / genre-function pseudo-gold JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "chunk_id",
    "dataset",
    "text_preview",
    "old_review_topic_domain",
    "old_review_topic_abstained",
    "annotation_v2",
    "semantic_topic_domain",
    "semantic_topic_confidence",
    "semantic_topic_abstained",
    "semantic_topic_note",
    "genre_function",
    "genre_function_confidence",
    "genre_function_abstained",
    "genre_function_note",
    "review_note",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_allowed_labels(input_path: Path) -> tuple[set[str], set[str]]:
    root = input_path.resolve().parents[1]
    taxonomy_dir = root / "taxonomy"

    semantic_taxonomy = load_json(taxonomy_dir / "semantic_topic_domain_v1_descriptions.json")
    genre_taxonomy = load_json(taxonomy_dir / "genre_function_v1_descriptions.json")

    semantic_labels = {item["domain"] for item in semantic_taxonomy["domains"]}
    genre_labels = {item["label"] for item in genre_taxonomy["labels"]}
    return semantic_labels, genre_labels


def confidence_bucket(value: float) -> str:
    return f"{value:.1f}"


def validate_confidence(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    semantic_allowed, genre_allowed = load_allowed_labels(args.input)

    errors: list[str] = []
    chunk_ids: list[str] = []
    by_dataset = Counter()
    semantic_distribution = Counter()
    genre_distribution = Counter()
    semantic_confidence_distribution = Counter()
    genre_confidence_distribution = Counter()
    crosstab: dict[str, Counter[str]] = defaultdict(Counter)
    old_split_examples: dict[str, list[dict[str, Any]]] = {
        "commercial": [],
        "reference": [],
        "education": [],
        "media": [],
    }

    semantic_abstained = 0
    genre_abstained = 0

    for index, record in enumerate(records):
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            errors.append(f"record {index} missing fields: {missing}")

        chunk_id = record.get("chunk_id")
        if isinstance(chunk_id, str):
            chunk_ids.append(chunk_id)
        else:
            errors.append(f"record {index} has invalid chunk_id")

        semantic_label = record.get("semantic_topic_domain")
        genre_label = record.get("genre_function")
        semantic_confidence = record.get("semantic_topic_confidence")
        genre_confidence = record.get("genre_function_confidence")

        if semantic_label not in semantic_allowed:
            errors.append(f"{chunk_id}: invalid semantic_topic_domain={semantic_label!r}")
        if genre_label not in genre_allowed:
            errors.append(f"{chunk_id}: invalid genre_function={genre_label!r}")
        if not validate_confidence(semantic_confidence):
            errors.append(f"{chunk_id}: invalid semantic_topic_confidence={semantic_confidence!r}")
        if not validate_confidence(genre_confidence):
            errors.append(f"{chunk_id}: invalid genre_function_confidence={genre_confidence!r}")
        if not isinstance(record.get("semantic_topic_abstained"), bool):
            errors.append(f"{chunk_id}: semantic_topic_abstained must be boolean")
        if not isinstance(record.get("genre_function_abstained"), bool):
            errors.append(f"{chunk_id}: genre_function_abstained must be boolean")

        by_dataset[str(record.get("dataset"))] += 1
        semantic_distribution[str(semantic_label)] += 1
        genre_distribution[str(genre_label)] += 1
        if validate_confidence(semantic_confidence):
            semantic_confidence_distribution[confidence_bucket(float(semantic_confidence))] += 1
        if validate_confidence(genre_confidence):
            genre_confidence_distribution[confidence_bucket(float(genre_confidence))] += 1
        crosstab[str(semantic_label)][str(genre_label)] += 1

        if record.get("semantic_topic_abstained") is True:
            semantic_abstained += 1
        if record.get("genre_function_abstained") is True:
            genre_abstained += 1

        old_label = str(record.get("old_review_topic_domain"))
        if old_label in old_split_examples and len(old_split_examples[old_label]) < 8:
            old_split_examples[old_label].append(
                {
                    "chunk_id": chunk_id,
                    "dataset": record.get("dataset"),
                    "old_review_topic_domain": old_label,
                    "semantic_topic_domain": semantic_label,
                    "genre_function": genre_label,
                    "semantic_topic_note": record.get("semantic_topic_note"),
                    "genre_function_note": record.get("genre_function_note"),
                }
            )

    duplicate_ids = sorted({item for item, count in Counter(chunk_ids).items() if count > 1})
    if duplicate_ids:
        errors.append(f"duplicate chunk_id values: {duplicate_ids}")

    summary = {
        "input": str(args.input),
        "records_count": len(records),
        "by_dataset": dict(sorted(by_dataset.items())),
        "semantic_topic_domain_distribution": dict(sorted(semantic_distribution.items())),
        "semantic_topic_abstention": {
            "count": semantic_abstained,
            "share": round(semantic_abstained / len(records), 6) if records else 0.0,
        },
        "genre_function_distribution": dict(sorted(genre_distribution.items())),
        "genre_function_abstention": {
            "count": genre_abstained,
            "share": round(genre_abstained / len(records), 6) if records else 0.0,
        },
        "semantic_topic_confidence_distribution": dict(sorted(semantic_confidence_distribution.items())),
        "genre_function_confidence_distribution": dict(sorted(genre_confidence_distribution.items())),
        "semantic_topic_x_genre_function": {
            semantic: dict(sorted(counter.items()))
            for semantic, counter in sorted(crosstab.items())
        },
        "old_mixed_label_split_examples": old_split_examples,
        "integrity": {
            "required_fields_present": not any("missing fields" in error for error in errors),
            "records_count_is_120": len(records) == 120,
            "duplicate_chunk_id_count": len(duplicate_ids),
            "allowed_labels_only": not any("invalid semantic_topic_domain" in error or "invalid genre_function" in error for error in errors),
            "confidence_values_valid": not any("invalid semantic_topic_confidence" in error or "invalid genre_function_confidence" in error for error in errors),
            "errors": errors,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(summary["integrity"], ensure_ascii=False, indent=2))

    if errors or len(records) != 120:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
