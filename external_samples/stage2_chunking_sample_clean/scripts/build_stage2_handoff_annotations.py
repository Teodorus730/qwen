"""Build unified Stage2 handoff annotation JSONL files.

This is a packaging script: it joins existing component outputs by chunk_id.
It does not create new labels, tune classifiers, or run models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = ("fineweb", "fineweb_edu", "finemath")
DATASET_NAMES = {
    "fineweb": "FineWeb",
    "fineweb_edu": "FineWeb-Edu",
    "finemath": "FineMath",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def read_map(path: Path) -> dict[str, dict[str, Any]]:
    return {record["chunk_id"]: record for record in read_jsonl(path)}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def dataset_paths(base: Path, split: str, suffix: str) -> list[Path]:
    return [
        base / "data_samples" / "real_samples" / f"real_hf_benchmark_{split}_{dataset}_{suffix}.jsonl"
        for dataset in DATASETS
    ]


def load_feature_records(base: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in dataset_paths(base, split, "features_v2_tokenized"):
        records.extend(read_jsonl(path))
    return sorted(records, key=lambda r: r["chunk_id"])


def load_topic_predictions(base: Path, split: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in dataset_paths(base, split, "topic_domain_v2_1"):
        out.update(read_map(path))
    return out


def preview_text(record: dict[str, Any], max_chars: int = 700) -> str:
    text = str(record.get("text") or "")
    return text[:max_chars]


def old_topic_gold(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "review_topic_domain": record.get("review_topic_domain"),
        "review_topic_confidence": record.get("review_topic_confidence"),
        "review_topic_abstained": record.get("review_topic_abstained"),
        "review_topic_note": record.get("review_topic_note"),
        "review_noise_level": record.get("review_noise_level"),
        "review_note": record.get("review_note"),
        "pseudo_gold_patch_applied": record.get("pseudo_gold_patch_applied"),
        "pseudo_gold_patch_reason": record.get("pseudo_gold_patch_reason"),
    }


def semantic_gold(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "domain": record.get("semantic_topic_domain"),
        "confidence": record.get("semantic_topic_confidence"),
        "abstained": record.get("semantic_topic_abstained"),
        "note": record.get("semantic_topic_note"),
        "schema_version": "semantic_topic_domain_v1",
        "split_role": "v1_dev_pseudo_gold",
    }


def genre_gold(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "label": record.get("genre_function"),
        "confidence": record.get("genre_function_confidence"),
        "abstained": record.get("genre_function_abstained"),
        "note": record.get("genre_function_note"),
        "schema_version": "genre_function_v1",
        "split_role": "v1_dev_pseudo_gold",
    }


def embedding_prediction(record: dict[str, Any] | None, *, prediction_only: bool) -> dict[str, Any] | None:
    if record is None:
        return None
    payload = record.get("semantic_topic_embedding")
    if not isinstance(payload, dict):
        return None
    enriched = dict(payload)
    enriched["prediction_only"] = prediction_only
    enriched["held_out_evaluated"] = False
    return enriched


def tokenization_stats(annotation_v2: dict[str, Any]) -> dict[str, Any]:
    text_stats = annotation_v2.get("text_stats") or {}
    tokenizer = annotation_v2.get("tokenizer") or {}
    return {
        "token_count": text_stats.get("token_count"),
        "token_per_byte": text_stats.get("token_per_byte"),
        "tokens_per_char": text_stats.get("tokens_per_char"),
        "bytes_per_token": text_stats.get("bytes_per_token"),
        "chars_per_token": text_stats.get("chars_per_token"),
        "tokens_per_word_rough": text_stats.get("tokens_per_word_rough"),
        "tokenizer": tokenizer,
    }


def build_record(
    feature_record: dict[str, Any],
    *,
    split: str,
    topic_pred: dict[str, Any] | None,
    old_gold: dict[str, Any] | None,
    semantic_record: dict[str, Any] | None,
    bge_v1_1: dict[str, Any] | None,
    bge_reranked: dict[str, Any] | None,
) -> dict[str, Any]:
    annotation_v2 = feature_record.get("annotation_v2") or {}
    topic_payload = None
    if topic_pred is not None:
        topic_payload = ((topic_pred.get("annotation_v2") or {}).get("topic"))

    is_v1 = split == "v1_dev"
    caveats = {
        "split": split,
        "stage2_handoff": True,
        "nll_ready_axes": [
            "provenance",
            "annotation_v2.text_stats",
            "annotation_v2.tokenizer",
            "annotation_v2.surface",
            "annotation_v2.quality",
        ],
        "legacy_topic_domain": "weak/exploratory metadata; use confidence and abstention",
        "cleaned_axes": (
            "v1-dev pseudo-gold and dev-only BGE predictions"
            if is_v1
            else "not labeled/evaluated under cleaned axes for held-out use"
        ),
        "do_not_claim_held_out_cleaned_semantic_quality": True,
    }

    return {
        "chunk_id": feature_record["chunk_id"],
        "dataset": feature_record.get("dataset"),
        "text_preview": feature_record.get("text_preview") or preview_text(feature_record),
        "text": feature_record.get("text"),
        "provenance": annotation_v2.get("provenance"),
        "annotation_v2": annotation_v2,
        "tokenization_stats": tokenization_stats(annotation_v2),
        "surface": annotation_v2.get("surface"),
        "quality": annotation_v2.get("quality"),
        "legacy_weak_topic_domain_v2_1": topic_payload,
        "old_mixed_topic_pseudo_gold": old_topic_gold(old_gold),
        "semantic_topic_domain_v1": semantic_gold(semantic_record) if is_v1 else None,
        "genre_function_v1": genre_gold(semantic_record) if is_v1 else {"label": None, "abstained": True, "status": "not_labeled"},
        "semantic_topic_embedding_bge_m3_v1_1": embedding_prediction(bge_v1_1, prediction_only=not is_v1),
        "semantic_topic_embedding_bge_m3_v1_1_reranked": embedding_prediction(bge_reranked, prediction_only=not is_v1),
        "handoff_caveats": caveats,
    }


def build_split(base: Path, split: str, output: Path, *, include_cleaned_dev: bool) -> int:
    feature_records = load_feature_records(base, "v1" if split == "v1_dev" else "v2_test")
    topic_predictions = load_topic_predictions(base, "v1" if split == "v1_dev" else "v2_test")

    if split == "v1_dev":
        old_gold_map = read_map(base / "data_samples" / "real_hf_benchmark_v1_annotation_v2_pseudo_gold_patched.jsonl")
        semantic_map = read_map(base / "data_samples" / "real_hf_benchmark_v1_semantic_topic_genre_function_pseudo_gold.jsonl")
        bge_map = read_map(base / "data_samples" / "real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_cards.jsonl")
        bge_reranked_map = read_map(base / "data_samples" / "real_hf_benchmark_v1_semantic_topic_embedding_bge_m3_v1_1_reranked.jsonl")
    else:
        old_gold_map = read_map(base / "data_samples" / "real_hf_benchmark_v2_test_annotation_v2_pseudo_gold.jsonl")
        semantic_map = {}
        bge_map = {}
        bge_reranked_map = {}

    records = [
        build_record(
            feature_record,
            split=split,
            topic_pred=topic_predictions.get(feature_record["chunk_id"]),
            old_gold=old_gold_map.get(feature_record["chunk_id"]),
            semantic_record=semantic_map.get(feature_record["chunk_id"]) if include_cleaned_dev else None,
            bge_v1_1=bge_map.get(feature_record["chunk_id"]) if include_cleaned_dev else None,
            bge_reranked=bge_reranked_map.get(feature_record["chunk_id"]) if include_cleaned_dev else None,
        )
        for feature_record in feature_records
    ]
    write_jsonl(output, records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--v1-output", type=Path, default=Path("data_samples/handoff/stage2_v1_dev_annotations.jsonl"))
    parser.add_argument("--v2-output", type=Path, default=Path("data_samples/handoff/stage2_v2_test_annotations.jsonl"))
    args = parser.parse_args()

    base = args.base_dir
    v1_count = build_split(base, "v1_dev", args.v1_output, include_cleaned_dev=True)
    v2_count = build_split(base, "v2_test", args.v2_output, include_cleaned_dev=False)
    print(json.dumps({"v1_dev_records": v1_count, "v2_test_records": v2_count}, indent=2))


if __name__ == "__main__":
    main()
