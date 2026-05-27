#!/usr/bin/env python3
"""Source-type-only lexical, MiniLM, and hybrid baselines."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


LEXICAL_METHOD = "source_type_lexical_v1"
LEXICAL_LOW_METHOD = "source_type_lexical_low_confidence"
MINILM_METHOD = "source_type_minilm_v1"
MINILM_LOW_METHOD = "source_type_minilm_low_confidence"
HYBRID_OVERRIDE_METHOD = "source_type_hybrid_regex_override"
HYBRID_MINILM_METHOD = "source_type_hybrid_minilm"
HYBRID_LOW_METHOD = "source_type_hybrid_low_confidence"


TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+")


OVERRIDES = [
    (
        "code",
        re.compile(r"\b(function|class|api|parameter|return|exception|linsolve|matrix|vector|mod\s+\d+)\b", re.I),
        0.82,
    ),
    (
        "forum_qa",
        re.compile(r"\b(commented:|what am i doing wrong|homework question|accepted answer|asked\s+\w+\s+\d{1,2})\b", re.I),
        0.82,
    ),
    (
        "commercial_product",
        re.compile(r"\b(out of stock|add to cart|shipping|warranty|buy now|price|retail|product page)\b", re.I),
        0.84,
    ),
    (
        "boilerplate_or_noise",
        re.compile(r"\b(cookie|privacy policy|terms of use|subscribe|show more|report an issue|sameday essay)\b", re.I),
        0.78,
    ),
    (
        "legal_government",
        re.compile(r"\b(public notice|regulation|ordinance|agency|tax commission|assessment roll|compliance)\b", re.I),
        0.8,
    ),
    (
        "math",
        re.compile(r"(\$.*?\$|\\begin\{align|\\frac|sin\(|cos\(|\b(prove|derivative|integral|equation|solve|median|probability)\b)", re.I),
        0.8,
    ),
]


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path} line {line_number}: {exc}") from exc


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_labels(path):
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        raise SystemExit("source_type labels must be a non-empty JSON array")
    seen = set()
    for label in labels:
        source_type = label.get("source_type")
        if not source_type:
            raise SystemExit("each source_type label must have source_type")
        if source_type in seen:
            raise SystemExit(f"duplicate source_type label: {source_type}")
        seen.add(source_type)
    return labels


def label_text(label):
    parts = [
        label.get("source_type"),
        label.get("description"),
        " ".join(label.get("positive_keywords") or []),
        " ".join(label.get("examples") or []),
    ]
    return " ".join(str(part) for part in parts if part)


def text_for_record(record, text_chars):
    return str(record.get("text") or "")[:text_chars]


def tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def vectorize(text):
    counts = Counter(tokenize(text))
    norm = math.sqrt(sum(value * value for value in counts.values()))
    return counts, norm


def cosine(left, right):
    left_counts, left_norm = left
    right_counts, right_norm = right
    if left_norm == 0 or right_norm == 0:
        return 0.0
    if len(left_counts) > len(right_counts):
        left_counts, right_counts = right_counts, left_counts
    dot = sum(value * right_counts.get(token, 0) for token, value in left_counts.items())
    return dot / (left_norm * right_norm)


def clamp(value):
    return max(0.0, min(1.0, float(value)))


def top_k_from_scores(scores, labels, top_k):
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            "source_type": labels[index]["source_type"],
            "confidence": clamp(score),
        }
        for index, score in ranked
    ]


def lexical_predict(text, labels, label_vectors, top_k):
    text_vector = vectorize(text)
    scores = [cosine(text_vector, label_vector) for label_vector in label_vectors]
    predictions = top_k_from_scores(scores, labels, top_k)
    return predictions[0], predictions


def apply_source_type(record, prediction, top_k, method, low_method, min_confidence, model_name=None):
    out = dict(record)
    confidence = clamp(prediction["confidence"])
    if confidence < min_confidence:
        out["source_type"] = "unknown"
        out["source_type_confidence"] = confidence
        out["source_type_method"] = low_method
    else:
        out["source_type"] = prediction["source_type"]
        out["source_type_confidence"] = confidence
        out["source_type_method"] = method
    out["source_type_top_k"] = top_k
    if model_name:
        out["source_type_model"] = model_name
    return out


def regex_override(text):
    for source_type, pattern, confidence in OVERRIDES:
        if pattern.search(text):
            return {
                "source_type": source_type,
                "confidence": confidence,
            }
    return None


def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for minilm/hybrid mode; install/use the embedding environment."
        ) from exc
    return SentenceTransformer


def cosine_matrix(text_vectors, label_vectors):
    try:
        import numpy as np
    except ImportError:
        return cosine_matrix_python(text_vectors, label_vectors)
    text_matrix = np.asarray(text_vectors, dtype=float)
    label_matrix = np.asarray(label_vectors, dtype=float)
    text_norms = np.linalg.norm(text_matrix, axis=1, keepdims=True)
    label_norms = np.linalg.norm(label_matrix, axis=1, keepdims=True).T
    denom = text_norms * label_norms
    denom[denom == 0] = 1.0
    return (text_matrix @ label_matrix.T) / denom


def cosine_matrix_python(text_vectors, label_vectors):
    rows = []
    for text_vector in text_vectors:
        row = []
        text_norm = sum(float(v) * float(v) for v in text_vector) ** 0.5
        for label_vector in label_vectors:
            label_norm = sum(float(v) * float(v) for v in label_vector) ** 0.5
            if text_norm == 0 or label_norm == 0:
                row.append(0.0)
            else:
                row.append(sum(float(a) * float(b) for a, b in zip(text_vector, label_vector)) / (text_norm * label_norm))
        rows.append(row)
    return rows


def classify_lexical(records, labels, args):
    label_vectors = [vectorize(label_text(label)) for label in labels]
    output = []
    for record in records:
        best, top_k = lexical_predict(text_for_record(record, args.text_chars), labels, label_vectors, args.top_k)
        output.append(apply_source_type(record, best, top_k, LEXICAL_METHOD, LEXICAL_LOW_METHOD, args.min_confidence))
    return output


def classify_minilm(records, labels, args, hybrid=False):
    SentenceTransformer = load_sentence_transformer()
    model = SentenceTransformer(args.model, device=args.device) if args.device else SentenceTransformer(args.model)
    label_texts = [label_text(label) for label in labels]
    label_vectors = model.encode(label_texts, batch_size=args.batch_size, show_progress_bar=False)
    output = []
    pending = []
    pending_indexes = []

    if hybrid:
        for index, record in enumerate(records):
            text = text_for_record(record, args.text_chars)
            override = regex_override(text)
            if override:
                output.append(
                    apply_source_type(
                        record,
                        override,
                        [override],
                        HYBRID_OVERRIDE_METHOD,
                        HYBRID_LOW_METHOD,
                        args.min_confidence,
                        args.model,
                    )
                )
            else:
                output.append(None)
                pending.append(text)
                pending_indexes.append(index)
    else:
        output = [None] * len(records)
        pending = [text_for_record(record, args.text_chars) for record in records]
        pending_indexes = list(range(len(records)))

    for offset in range(0, len(pending), args.batch_size):
        batch_texts = pending[offset : offset + args.batch_size]
        batch_indexes = pending_indexes[offset : offset + args.batch_size]
        text_vectors = model.encode(batch_texts, batch_size=args.batch_size, show_progress_bar=False)
        score_rows = cosine_matrix(text_vectors, label_vectors)
        for row_index, scores in zip(batch_indexes, score_rows):
            top_k = top_k_from_scores(scores, labels, args.top_k)
            best = top_k[0]
            method = HYBRID_MINILM_METHOD if hybrid else MINILM_METHOD
            low_method = HYBRID_LOW_METHOD if hybrid else MINILM_LOW_METHOD
            output[row_index] = apply_source_type(records[row_index], best, top_k, method, low_method, args.min_confidence, args.model)
    return output


def print_summary(records):
    print(f"records_written: {len(records)}")
    print(f"counts_by_source_type: {json.dumps(dict(Counter(r.get('source_type') for r in records)), ensure_ascii=False, sort_keys=True)}")
    print(f"counts_by_source_type_method: {json.dumps(dict(Counter(r.get('source_type_method') for r in records)), ensure_ascii=False, sort_keys=True)}")
    print(f"low_confidence_count: {sum(1 for r in records if str(r.get('source_type_method', '')).endswith('low_confidence'))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["lexical", "minilm", "hybrid"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--text-chars", type=int, default=2000)
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    records = list(iter_jsonl(Path(args.input)))
    if args.mode in {"minilm", "hybrid"} and not args.model:
        raise SystemExit("--model is required for minilm/hybrid mode")

    if args.mode == "lexical":
        output = classify_lexical(records, labels, args)
    elif args.mode == "minilm":
        output = classify_minilm(records, labels, args, hybrid=False)
    else:
        output = classify_minilm(records, labels, args, hybrid=True)

    write_jsonl(Path(args.output), output)
    print_summary(output)
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
