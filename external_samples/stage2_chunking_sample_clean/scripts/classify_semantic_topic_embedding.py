"""Classify semantic_topic_domain_v1 with embedding label-card similarity.

The default behavior is local-cache-only. Use --dry-run to validate inputs
without importing or loading an embedding model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


METHOD_NAME = "semantic_topic_domain_v1_embedding_label_card_similarity"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def model_cache_dir_name(model_name: str) -> str:
    return "models--" + model_name.replace("/", "--")


def candidate_cache_roots(cache_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir is not None:
        roots.append(cache_dir)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        roots.append(Path(hf_hub_cache))
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        roots.append(Path(user_profile) / ".cache" / "huggingface" / "hub")
    home = Path.home()
    roots.append(home / ".cache" / "huggingface" / "hub")

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def find_cached_snapshot(model_name: str, cache_dir: Path | None) -> Path | None:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path

    cache_name = model_cache_dir_name(model_name)
    for root in candidate_cache_roots(cache_dir):
        model_root = root / cache_name
        snapshots = model_root / "snapshots"
        if snapshots.exists():
            candidates = [p for p in snapshots.iterdir() if p.is_dir()]
            if candidates:
                return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def build_label_cards(domain_descriptions: dict[str, Any]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for item in domain_descriptions.get("domains", []):
        domain = item["domain"]
        parts = [
            f"Domain: {domain}",
            f"Description: {item.get('description', '')}",
        ]
        positives = item.get("positive_examples") or []
        negatives = item.get("negative_notes") or []
        if positives:
            parts.append("Positive examples: " + " ".join(str(x) for x in positives))
        if negatives:
            parts.append("Negative notes: " + " ".join(str(x) for x in negatives))
        cards.append({"domain": domain, "text": "\n".join(parts)})
    if not cards:
        raise ValueError("No domains found in domain descriptions JSON")
    return cards


def validate_args(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]], Path | None]:
    if args.threshold < -1.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be between -1.0 and 1.0")
    if args.margin_threshold < 0.0 or args.margin_threshold > 2.0:
        raise ValueError("--margin-threshold must be between 0.0 and 2.0")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    records = read_jsonl(args.input)
    if not records:
        raise ValueError("Input JSONL is empty")

    missing_text = [record.get("chunk_id", f"index:{i}") for i, record in enumerate(records) if args.text_field not in record]
    if missing_text:
        raise ValueError(f"Missing text field {args.text_field!r} for {len(missing_text)} records; first={missing_text[0]}")

    domain_json = read_json(args.domain_descriptions)
    cards = build_label_cards(domain_json)
    cached_snapshot = find_cached_snapshot(args.model, args.cache_dir)
    return records, cards, cached_snapshot


def encode_texts(model: Any, texts: list[str]) -> Any:
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def dot_scores(vector: Any, matrix: Any) -> list[float]:
    return [float(sum(float(a) * float(b) for a, b in zip(vector, row))) for row in matrix]


def classify_records(args: argparse.Namespace, records: list[dict[str, Any]], cards: list[dict[str, str]], model_path: Path | str) -> list[dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence_transformers is not importable. Create/activate .venv-embedding "
            "or install dependencies before running non-dry classification."
        ) from exc

    model = SentenceTransformer(str(model_path), cache_folder=str(args.cache_dir) if args.cache_dir else None)
    card_embeddings = encode_texts(model, [card["text"] for card in cards])
    domains = [card["domain"] for card in cards]

    output_records: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get(args.text_field) or "")
        embedding = encode_texts(model, [text])[0]
        scores = dot_scores(embedding, card_embeddings)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        top = ranked[: args.top_k]
        top_index, top_score = top[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = float(top_score - second_score)

        abstain_reasons: list[str] = []
        if top_score < args.threshold:
            abstain_reasons.append(f"top_score_below_threshold:{top_score:.6f}<{args.threshold:.6f}")
        if margin < args.margin_threshold:
            abstain_reasons.append(f"margin_below_threshold:{margin:.6f}<{args.margin_threshold:.6f}")

        abstained = bool(abstain_reasons)
        predicted_domain = None if abstained else domains[top_index]

        enriched = dict(record)
        enriched["semantic_topic_embedding"] = {
            "domain": predicted_domain,
            "confidence": round(float(top_score), 6),
            "abstained": abstained,
            "abstain_reason": ";".join(abstain_reasons) if abstain_reasons else None,
            "method": METHOD_NAME,
            "model": args.model,
            "top_k": [
                {
                    "domain": domains[index],
                    "score": round(float(score), 6),
                }
                for index, score in top
            ],
            "margin": round(margin, 6),
            "evidence": {
                "text_field": args.text_field,
                "domain_card_source": str(args.domain_descriptions),
                "threshold": args.threshold,
                "margin_threshold": args.margin_threshold,
            },
        }
        output_records.append(enriched)
    return output_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--domain-descriptions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_embedding_cache"))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow model download. Default is local cache only.")
    args = parser.parse_args()

    try:
        records, cards, cached_snapshot = validate_args(args)
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    cache_status = str(cached_snapshot) if cached_snapshot else "not found"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "records": len(records),
                    "domains": [card["domain"] for card in cards],
                    "model": args.model,
                    "cached_snapshot": cache_status,
                    "would_write": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if cached_snapshot is None and not args.allow_download:
        print(
            f"Model {args.model!r} was not found in local cache. "
            "Refusing to download without explicit --allow-download.",
            file=sys.stderr,
        )
        raise SystemExit(3)

    model_path: Path | str = cached_snapshot if cached_snapshot is not None else args.model
    output_records = classify_records(args, records, cards, model_path)
    write_jsonl(args.output, output_records)
    print(f"Wrote {len(output_records)} predictions to {args.output}")


if __name__ == "__main__":
    main()
