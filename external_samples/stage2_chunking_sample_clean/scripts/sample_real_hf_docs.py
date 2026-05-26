#!/usr/bin/env python3
"""Small safe Hugging Face streaming sampler for real stage2 benchmark docs."""

import argparse
import json
import random
import re
from pathlib import Path


MAX_SAFE_STREAM_LIMIT = 10_000


def safe_id_part(value):
    text = str(value or "sample").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "sample"


def extract_text(row, text_field):
    value = row.get(text_field)
    if isinstance(value, str) and value.strip():
        return value
    return ""


def reservoir_sample(stream, sample_size, stream_limit, seed, text_field, min_text_chars):
    rng = random.Random(seed)
    sample = []
    eligible_count = 0

    for stream_index, row in enumerate(stream):
        if stream_index >= stream_limit:
            break

        text = extract_text(row, text_field)
        if len(text.strip()) < min_text_chars:
            continue

        candidate = {
            "row": row,
            "text": text,
            "stream_index": stream_index,
        }
        eligible_count += 1

        if len(sample) < sample_size:
            sample.append(candidate)
            continue

        replace_index = rng.randrange(eligible_count)
        if replace_index < sample_size:
            sample[replace_index] = candidate

    sample.sort(key=lambda item: item["stream_index"])
    return sample, eligible_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", "--config", dest="config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--source-type", default="unknown")
    parser.add_argument("--stream-limit", type=int, default=500)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-chars", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.stream_limit <= 0:
        print("--stream-limit must be > 0")
        raise SystemExit(2)
    if args.sample_size <= 0:
        print("--sample-size must be > 0")
        raise SystemExit(2)
    if args.stream_limit > MAX_SAFE_STREAM_LIMIT:
        print(f"--stream-limit {args.stream_limit} is above safe cap {MAX_SAFE_STREAM_LIMIT}")
        raise SystemExit(2)

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets is not installed. No install was attempted.")
        raise SystemExit(2)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "streaming": True,
                "stream_limit": args.stream_limit,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "min_text_chars": args.min_text_chars,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if args.config:
        stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    else:
        stream = load_dataset(args.dataset, split=args.split, streaming=True)

    sample, eligible_count = reservoir_sample(
        stream=stream,
        sample_size=args.sample_size,
        stream_limit=args.stream_limit,
        seed=args.seed,
        text_field=args.text_field,
        min_text_chars=args.min_text_chars,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_prefix = safe_id_part(args.dataset_label)

    with output_path.open("w", encoding="utf-8") as f:
        for sample_index, item in enumerate(sample):
            record = {
                "id": f"{source_prefix}_{sample_index:06d}",
                "dataset": args.dataset_label,
                "source_type": args.source_type,
                "text": item["text"],
                "source_dataset": args.dataset,
                "source_split": args.split,
                "sample_index": sample_index,
                "stream_index": item["stream_index"],
            }
            if args.config:
                record["source_config"] = args.config
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "stream_limit": args.stream_limit,
                "eligible_records_seen": eligible_count,
                "records_written": len(sample),
                "output": str(output_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
