#!/usr/bin/env python3
"""Prepare a manifest for future observed-token NLL scoring.

This script does not load models, import transformers, use the network, or
compute logits. It only writes planned scoring records.
"""

import argparse
import json
from pathlib import Path


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_window_sizes(value):
    sizes = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        sizes.append(int(raw))
    if not sizes:
        raise ValueError("at least one window size is required")
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--window-sizes", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    window_sizes = parse_window_sizes(args.window_sizes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in iter_jsonl(input_path):
            manifest_record = {
                "chunk_id": record.get("chunk_id"),
                "dataset": record.get("dataset"),
                "source_type": record.get("source_type"),
                "domain": record.get("domain"),
                "field": record.get("field"),
                "subfield": record.get("subfield"),
                "token_count": record.get("token_count"),
                "text_chars": len(record.get("text") or ""),
                "model_name": args.model_name,
                "planned_window_sizes": window_sizes,
                "planned_scoring_method": "observed_token_nll",
                "status": "pending",
            }
            f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
            count += 1

    print(f"manifest_records_written: {count}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
