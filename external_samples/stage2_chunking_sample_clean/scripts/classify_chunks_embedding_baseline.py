#!/usr/bin/env python3
"""Optional embedding nearest-label baseline for chunk JSONL files.

This script prepares the MiniLM/Sentence-Transformers baseline path while
staying fail-closed: dry-run does not load a model, and real runs request
local files only.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


EMBEDDING_METHOD = "embedding_nearest_label_minilm"


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON in {path} line {line_number}: {exc}")
                raise SystemExit(2)


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
    if not labels:
        raise ValueError("labels file must not be empty")
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


def encode_texts(model, texts, batch_size, device=None):
    kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
        "show_progress_bar": False,
    }
    if device:
        kwargs["device"] = device
    try:
        return model.encode(texts, **kwargs)
    except TypeError:
        if device:
            kwargs.pop("device", None)
            return model.encode(texts, **kwargs)
        raise


def _as_rows(matrix):
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    rows = list(matrix)
    if rows and not isinstance(rows[0], (list, tuple)):
        return [rows]
    return rows


def cosine_scores(text_vectors, label_vectors):
    """Return cosine similarity matrix, using numpy when available."""
    try:
        import numpy as np
    except ImportError:
        return cosine_scores_python(text_vectors, label_vectors)

    text_matrix = np.asarray(text_vectors, dtype=float)
    label_matrix = np.asarray(label_vectors, dtype=float)
    if text_matrix.ndim == 1:
        text_matrix = text_matrix.reshape(1, -1)
    if label_matrix.ndim == 1:
        label_matrix = label_matrix.reshape(1, -1)

    text_norms = np.linalg.norm(text_matrix, axis=1, keepdims=True)
    label_norms = np.linalg.norm(label_matrix, axis=1, keepdims=True).T
    denom = text_norms * label_norms
    scores = text_matrix @ label_matrix.T
    return np.divide(scores, denom, out=np.zeros_like(scores, dtype=float), where=denom != 0)


def cosine_scores_python(text_vectors, label_vectors):
    text_rows = _as_rows(text_vectors)
    label_rows = _as_rows(label_vectors)
    output = []
    for text_vector in text_rows:
        row = []
        text_norm = sum(float(value) * float(value) for value in text_vector) ** 0.5
        for label_vector in label_rows:
            label_norm = sum(float(value) * float(value) for value in label_vector) ** 0.5
            if text_norm == 0 or label_norm == 0:
                row.append(0.0)
                continue
            dot = sum(float(a) * float(b) for a, b in zip(text_vector, label_vector))
            row.append(dot / (text_norm * label_norm))
        output.append(row)
    return output


def row_values(row):
    return row.tolist() if hasattr(row, "tolist") else list(row)


def clamp_confidence(value):
    return max(0.0, min(1.0, float(value)))


def top_k_predictions(scores, labels, top_k):
    ranked = sorted(enumerate(row_values(scores)), key=lambda item: item[1], reverse=True)
    predictions = []
    for index, score in ranked[: max(top_k, 0)]:
        label = labels[index]
        predictions.append(
            {
                "domain": label.get("domain"),
                "field": label.get("field"),
                "subfield": label.get("subfield"),
                "confidence": clamp_confidence(score),
            }
        )
    return predictions


def has_existing_label(record):
    return any(record.get(field) not in (None, "") for field in ("domain", "field", "subfield", "confidence"))


def add_embedding_metadata(out, settings):
    out.update(
        {
            "label_method": EMBEDDING_METHOD,
            "embedding_model": settings["model_name"],
            "taxonomy_path": settings["taxonomy_path"],
            "min_confidence": settings["min_confidence"],
            "batch_size": settings["batch_size"],
            "device_requested": settings["device_requested"],
            "device_actual": settings["device_actual"],
            "text_chars": settings["text_chars"],
        }
    )


def classify_records(
    records,
    labels,
    model,
    model_name,
    taxonomy_path,
    min_confidence,
    text_chars,
    batch_size,
    device_requested=None,
    device_actual=None,
    top_k=3,
    overwrite_existing_labels=False,
):
    label_texts = [label_to_text(label) for label in labels]
    label_vectors = encode_texts(model, label_texts, batch_size=batch_size, device=device_requested)
    settings = {
        "model_name": model_name,
        "taxonomy_path": taxonomy_path,
        "min_confidence": min_confidence,
        "batch_size": batch_size,
        "device_requested": device_requested,
        "device_actual": device_actual,
        "text_chars": text_chars,
    }

    output = []
    records_to_classify = []
    record_indexes = []
    for index, record in enumerate(records):
        out = dict(record)
        if has_existing_label(out) and not overwrite_existing_labels:
            output.append(out)
            continue
        output.append(out)
        records_to_classify.append(record)
        record_indexes.append(index)

    for start in range(0, len(records_to_classify), batch_size):
        batch = records_to_classify[start : start + batch_size]
        batch_indexes = record_indexes[start : start + batch_size]
        texts = [str(record.get("text") or "")[:text_chars] for record in batch]
        text_vectors = encode_texts(model, texts, batch_size=batch_size, device=device_requested)
        score_matrix = cosine_scores(text_vectors, label_vectors)

        for local_index, score_row in enumerate(score_matrix):
            out = output[batch_indexes[local_index]]
            predictions = top_k_predictions(score_row, labels, top_k)
            best = predictions[0] if predictions else {"domain": None, "field": None, "subfield": None, "confidence": 0.0}
            confidence = float(best["confidence"])
            low_confidence = confidence < min_confidence

            add_embedding_metadata(out, settings)
            out["confidence"] = confidence
            out["low_confidence"] = low_confidence
            out["top_k_labels"] = predictions
            if low_confidence:
                out["domain"] = None
                out["field"] = None
                out["subfield"] = None
            else:
                out["domain"] = best["domain"]
                out["field"] = best["field"]
                out["subfield"] = best["subfield"]

    return output


def print_summary(records):
    method_counts = Counter(record.get("label_method") for record in records)
    domain_counts = Counter(record.get("domain") for record in records)
    low_confidence_count = sum(1 for record in records if record.get("low_confidence") is True)
    classified_count = sum(1 for record in records if record.get("label_method") == EMBEDDING_METHOD)
    print(f"records_read: {len(records)}")
    print(f"records_written: {len(records)}")
    print(f"records_classified_by_embedding: {classified_count}")
    print(f"counts_by_domain: {json.dumps({str(k): v for k, v in domain_counts.items()}, ensure_ascii=False, sort_keys=True)}")
    print(f"counts_by_label_method: {json.dumps(dict(method_counts), ensure_ascii=False, sort_keys=True)}")
    print(f"low_confidence_count: {low_confidence_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--text-chars", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--overwrite-existing-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        print("--batch-size must be > 0")
        raise SystemExit(2)
    if args.top_k <= 0:
        print("--top-k must be > 0")
        raise SystemExit(2)

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
        print(f"taxonomy_path: {labels_path}")
        print(f"min_confidence: {args.min_confidence}")
        print(f"batch_size: {args.batch_size}")
        print(f"device_requested: {args.device}")
        print(f"top_k: {args.top_k}")
        print("No model loaded and no output written.")
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed. No dependency install was attempted.")
        print("Run --dry-run, install dependencies later with approval, or use the rule-based/lexical baselines.")
        raise SystemExit(2)

    model_kwargs = {"local_files_only": True}
    if args.device:
        model_kwargs["device"] = args.device
    try:
        model = SentenceTransformer(args.model, **model_kwargs)
    except TypeError:
        print(
            "Could not request local_files_only/device from this sentence-transformers version. "
            "To avoid accidental model download, use a version that supports local_files_only or pass a local model path."
        )
        raise SystemExit(2)
    except Exception as exc:
        print(f"Could not load model locally: {exc}")
        print("No model download was attempted by this script. Use a local model path or get explicit approval first.")
        raise SystemExit(2)

    device_actual = str(getattr(model, "device", None)) if getattr(model, "device", None) is not None else None
    labeled_records = classify_records(
        records=records,
        labels=labels,
        model=model,
        model_name=args.model,
        taxonomy_path=str(labels_path),
        min_confidence=args.min_confidence,
        text_chars=args.text_chars,
        batch_size=args.batch_size,
        device_requested=args.device,
        device_actual=device_actual,
        top_k=args.top_k,
        overwrite_existing_labels=args.overwrite_existing_labels,
    )
    write_jsonl(output_path, labeled_records)
    print_summary(labeled_records)
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
