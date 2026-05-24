#!/usr/bin/env python3
"""Optional embedding nearest-label baseline for chunk JSONL files.

This is an experimental baseline, not a replacement for the transparent
rule-based classifier. It does not install dependencies, use the network, or
download models by itself.
"""

import argparse
import json
import math
from pathlib import Path


LOW_CONFIDENCE_METHOD = "embedding_nearest_label_low_confidence"
EMBEDDING_METHOD = "embedding_nearest_label_v1"


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_labels(path):
    with path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise ValueError("labels file must contain a JSON array")
    return labels


def label_to_text(label):
    parts = [
        label.get("domain"),
        label.get("field"),
        label.get("subfield"),
        label.get("description"),
        " ".join(label.get("keywords") or []),
    ]
    return " ".join(str(part) for part in parts if part)


def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def classify_records(records, labels, model, min_confidence, text_chars, overwrite_existing_labels):
    label_texts = [label_to_text(label) for label in labels]
    label_vectors = model.encode(label_texts, convert_to_numpy=False, show_progress_bar=False)

    output = []
    for record in records:
        out = dict(record)
        has_existing = any(out.get(field) not in (None, "") for field in ("domain", "field", "subfield", "confidence"))
        if has_existing and not overwrite_existing_labels:
            output.append(out)
            continue

        text = str(record.get("text") or "")[:text_chars]
        text_vector = model.encode([text], convert_to_numpy=False, show_progress_bar=False)[0]
        scores = [cosine_similarity(text_vector, label_vector) for label_vector in label_vectors]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        confidence = float(scores[best_index])
        best_label = labels[best_index]

        if confidence < min_confidence:
            out.update(
                {
                    "source_type": out.get("source_type") or "unknown",
                    "domain": None,
                    "field": None,
                    "subfield": None,
                    "confidence": confidence,
                    "label_method": LOW_CONFIDENCE_METHOD,
                }
            )
        else:
            out.update(
                {
                    "source_type": out.get("source_type") or "unknown",
                    "domain": best_label.get("domain"),
                    "field": best_label.get("field"),
                    "subfield": best_label.get("subfield"),
                    "confidence": confidence,
                    "label_method": EMBEDDING_METHOD,
                }
            )
        output.append(out)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--text-chars", type=int, default=2000)
    parser.add_argument("--overwrite-existing-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    records = list(iter_jsonl(input_path))
    labels = load_labels(labels_path)

    if args.dry_run:
        print("DRY RUN")
        print(f"records: {len(records)}")
        print(f"labels: {len(labels)}")
        print(f"model: {args.model}")
        print("No model loaded and no output written.")
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed. Install it later or run rule-based classifier.")
        raise SystemExit(2)

    try:
        model = SentenceTransformer(args.model, local_files_only=True)
    except TypeError:
        print(
            "Could not request local_files_only from this sentence-transformers version. "
            "To avoid accidental model download, install/use a version that supports local files or pass a local model path."
        )
        raise SystemExit(2)
    except Exception as exc:
        print(f"Could not load model locally: {exc}")
        print("No model download was attempted by this script. Use a local model path or run rule-based classifier.")
        raise SystemExit(2)

    labeled_records = classify_records(
        records=records,
        labels=labels,
        model=model,
        min_confidence=args.min_confidence,
        text_chars=args.text_chars,
        overwrite_existing_labels=args.overwrite_existing_labels,
    )
    write_jsonl(output_path, labeled_records)
    print(f"records_read: {len(records)}")
    print(f"records_written: {len(labeled_records)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
