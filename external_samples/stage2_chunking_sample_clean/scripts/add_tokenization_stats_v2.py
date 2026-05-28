#!/usr/bin/env python
"""Add tokenizer-aware annotation_v2 text statistics to feature JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


TOKENIZED_SCHEMA_VERSION = "annotation_v2_deterministic_features_v2_tokenized_v1"

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+", re.UNICODE)
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def tokenizer_metadata(tokenizer: Any, tokenizer_name: str, revision: str | None, add_special_tokens: bool) -> dict[str, Any]:
    return {
        "tokenizer_name": tokenizer_name,
        "revision": revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "len_tokenizer": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "add_special_tokens": add_special_tokens,
    }


def add_token_stats(record: dict[str, Any], tokenizer: Any, tokenizer_meta: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    text = record.get("text")
    if not isinstance(text, str):
        return record, "missing or non-string text"

    annotation = record.get("annotation_v2")
    if not isinstance(annotation, dict):
        return record, "missing or non-object annotation_v2"

    text_stats = annotation.get("text_stats")
    if not isinstance(text_stats, dict):
        return record, "missing or non-object annotation_v2.text_stats"

    add_special_tokens = bool(tokenizer_meta["add_special_tokens"])
    token_ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    token_count = len(token_ids)
    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    words = WORD_RE.findall(text)
    word_count = text_stats.get("word_count_rough")
    if not isinstance(word_count, (int, float)) or isinstance(word_count, bool) or word_count < 0:
        word_count = len(words)
    unique_words = {word.lower() for word in words}
    latin_count = len(LATIN_RE.findall(text))
    cyrillic_count = len(CYRILLIC_RE.findall(text))

    text_stats.update(
        {
            "token_count": token_count,
            "token_per_byte": safe_ratio(token_count, byte_count),
            "tokens_per_char": safe_ratio(token_count, char_count),
            "bytes_per_token": safe_ratio(byte_count, token_count),
            "chars_per_token": safe_ratio(char_count, token_count),
            "tokens_per_word_rough": safe_ratio(token_count, word_count),
            "unique_word_count_rough": len(unique_words),
            "avg_word_length_rough": safe_ratio(sum(len(word) for word in words), word_count),
            "latin_char_count": latin_count,
            "cyrillic_char_count": cyrillic_count,
            "latin_char_ratio": safe_ratio(latin_count, char_count),
            "cyrillic_char_ratio": safe_ratio(cyrillic_count, char_count),
        }
    )

    annotation["tokenizer"] = dict(tokenizer_meta)
    annotation["schema_version"] = TOKENIZED_SCHEMA_VERSION
    return record, None


def process_file(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    tokenizer_meta: dict[str, Any],
) -> dict[str, Any]:
    records_read = 0
    records_written = 0
    errors: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as in_fh, output_path.open("w", encoding="utf-8") as out_fh:
        for line_no, line in enumerate(in_fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            records_read += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_no}: record must be object")
                continue
            enriched, error = add_token_stats(record, tokenizer, tokenizer_meta)
            if error:
                chunk_id = record.get("chunk_id", "<missing>")
                errors.append(f"line {line_no} chunk_id={chunk_id}: {error}")
                continue
            out_fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            records_written += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "records_read": records_read,
        "records_written": records_written,
        "errors_count": len(errors),
        "errors_sample": errors[:20],
        "tokenizer": tokenizer_meta,
        "schema_version": TOKENIZED_SCHEMA_VERSION,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add tokenizer-aware annotation_v2 statistics.")
    parser.add_argument("--input", required=True, help="Input refined annotation_v2 JSONL.")
    parser.add_argument("--output", required=True, help="Output tokenized annotation_v2 JSONL.")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    parser.add_argument("--cache-dir", default=".hf_tokenizer_cache")
    parser.add_argument("--add-special-tokens", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    meta = tokenizer_metadata(
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer_name,
        revision=args.revision,
        add_special_tokens=args.add_special_tokens,
    )
    summary = process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        tokenizer=tokenizer,
        tokenizer_meta=meta,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["errors_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
