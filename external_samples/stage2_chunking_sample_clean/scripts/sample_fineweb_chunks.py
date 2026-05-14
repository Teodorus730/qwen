#!/usr/bin/env python3
"""
Small chunking prototype for stage 2.

Default mode uses local example docs, so the script can be checked without
downloading FineWeb. For a real FineWeb run use --use-hf-streaming.
"""

import argparse
import json
import re
from pathlib import Path
from itertools import islice


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


class TokenCounter:
    def __init__(self, tokenizer_name=None):
        self.tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"tokenizer={tokenizer_name}")
            except Exception as exc:
                print(f"Could not load tokenizer '{tokenizer_name}', using simple fallback. Reason: {exc}")

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def hard_split(self, text: str, max_tokens: int):
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(ids), max_tokens):
                yield self.tokenizer.decode(ids[i:i + max_tokens])
        else:
            parts = re.findall(r"\S+", text)
            for i in range(0, len(parts), max_tokens):
                yield " ".join(parts[i:i + max_tokens])


def chunk_document(text: str, counter: TokenCounter, min_tokens: int, target_tokens: int, max_tokens: int):
    paragraphs = split_paragraphs(text)
    chunks = []
    current = []

    for p in paragraphs:
        candidate = "\n\n".join(current + [p]) if current else p
        n = counter.count(candidate)

        if n <= target_tokens:
            current.append(p)
            continue

        if current:
            chunk_text = "\n\n".join(current)
            if counter.count(chunk_text) >= min_tokens:
                chunks.append(chunk_text)

        if counter.count(p) > max_tokens:
            for piece in counter.hard_split(p, max_tokens):
                if counter.count(piece) >= min_tokens:
                    chunks.append(piece)
            current = []
        else:
            current = [p]

    if current:
        chunk_text = "\n\n".join(current)
        if counter.count(chunk_text) >= min_tokens:
            chunks.append(chunk_text)

    return chunks


def iter_local_jsonl(path: Path, max_docs: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def iter_hf_streaming(dataset: str, config: str, split: str, max_docs: int):
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True)
    yield from islice(ds, max_docs)


def guess_text(row: dict) -> str:
    for key in ("text", "content", "body"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-input", default="examples/local_docs.jsonl")
    parser.add_argument("--use-hf-streaming", action="store_true")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-name", default=None, help="Example: Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--min-chunk-tokens", type=int, default=80)
    parser.add_argument("--target-chunk-tokens", type=int, default=180)
    parser.add_argument("--max-chunk-tokens", type=int, default=320)
    parser.add_argument("--out", default="data_samples/fineweb_chunks_sample.jsonl")
    parser.add_argument("--stats-out", default="data_samples/run_stats.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    stats_path = Path(args.stats_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    counter = TokenCounter(args.tokenizer_name)

    if args.use_hf_streaming:
        rows = iter_hf_streaming(args.dataset, args.config, args.split, args.max_docs)
        dataset_label = "FineWeb"
        source_type = "web_general"
        input_mode = "hf_streaming"
    else:
        rows = iter_local_jsonl(Path(args.local_input), args.max_docs)
        dataset_label = "local_fineweb_like_sample"
        source_type = "web_general"
        input_mode = "local_jsonl"

    docs_seen = 0
    docs_with_text = 0
    chunks_written = 0
    token_counts = []

    with out_path.open("w", encoding="utf-8") as f:
        for doc_id, row in enumerate(rows):
            docs_seen += 1
            text = normalize_text(guess_text(row))
            if not text:
                continue
            docs_with_text += 1

            chunks = chunk_document(
                text=text,
                counter=counter,
                min_tokens=args.min_chunk_tokens,
                target_tokens=args.target_chunk_tokens,
                max_tokens=args.max_chunk_tokens,
            )

            for chunk_idx, chunk_text in enumerate(chunks):
                n_tokens = counter.count(chunk_text)
                token_counts.append(n_tokens)
                record = {
                    "chunk_id": f"{dataset_label}_{doc_id:06d}_{chunk_idx:03d}",
                    "dataset": dataset_label,
                    "source_type": source_type,
                    "domain": None,
                    "field": None,
                    "subfield": None,
                    "confidence": None,
                    "token_count": n_tokens,
                    "text": chunk_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks_written += 1

    stats = {
        "input_mode": input_mode,
        "docs_seen": docs_seen,
        "docs_with_text": docs_with_text,
        "chunks_written": chunks_written,
        "out": str(out_path),
        "min_chunk_tokens": min(token_counts) if token_counts else None,
        "mean_chunk_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else None,
        "max_chunk_tokens": max(token_counts) if token_counts else None,
        "settings": {
            "min_chunk_tokens": args.min_chunk_tokens,
            "target_chunk_tokens": args.target_chunk_tokens,
            "max_chunk_tokens": args.max_chunk_tokens,
            "max_docs": args.max_docs,
            "tokenizer_name": args.tokenizer_name,
        },
    }

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
