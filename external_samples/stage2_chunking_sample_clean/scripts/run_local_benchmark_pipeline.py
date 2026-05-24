#!/usr/bin/env python3
"""Run the local classifier benchmark pipeline."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print("+ " + " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"Command failed with exit code {completed.returncode}")
        raise SystemExit(completed.returncode)


def has_expected_labels(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            return any(key in record for key in ("expected_source_type", "expected_domain", "expected_field", "expected_subfield"))
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-input", default="examples/local_docs_classifier_benchmark.jsonl")
    parser.add_argument("--chunks-out", default="data_samples/classifier_benchmark_chunks.jsonl")
    parser.add_argument("--labeled-out", default="data_samples/classifier_benchmark_labeled.jsonl")
    parser.add_argument("--stats-out", default="data_samples/classifier_benchmark_run_stats.json")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-docs", type=int, default=100)
    args = parser.parse_args()

    python = sys.executable

    run_command(
        [
            python,
            "scripts/sample_fineweb_chunks.py",
            "--local-input",
            args.benchmark_input,
            "--max-docs",
            str(args.max_docs),
            "--out",
            args.chunks_out,
            "--stats-out",
            args.stats_out,
        ]
    )
    run_command([python, "scripts/validate_chunks.py", "--input", args.chunks_out])
    run_command([python, "scripts/classify_chunks_rule_based.py", "--input", args.chunks_out, "--output", args.labeled_out])
    run_command([python, "scripts/validate_chunks.py", "--input", args.labeled_out, "--require-labels"])
    run_command([python, "scripts/inspect_chunks.py", "--input", args.labeled_out, "--limit", "13"])

    if not args.skip_eval and has_expected_labels(Path(args.labeled_out)):
        run_command([python, "scripts/evaluate_chunk_labels.py", "--input", args.labeled_out])


if __name__ == "__main__":
    main()
