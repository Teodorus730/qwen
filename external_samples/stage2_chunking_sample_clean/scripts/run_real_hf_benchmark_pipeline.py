#!/usr/bin/env python3
"""Run local real-HF benchmark stages after docs have been sampled."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print("+ " + " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def output_prefix_for_docs(output_prefix, docs_path):
    stem = Path(docs_path).stem
    if stem.endswith("_docs"):
        stem = stem[:-5]
    return f"{output_prefix}_{stem}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", required=True)
    parser.add_argument("--output-prefix", default="real_hf_benchmark")
    parser.add_argument("--output-dir", default="data_samples/real_samples")
    parser.add_argument("--labels", default="taxonomy/simple_domain_labels.json")
    parser.add_argument("--embedding-python", required=True)
    parser.add_argument("--minilm-model", required=True)
    parser.add_argument("--min-embedding-confidence", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=100000)
    args = parser.parse_args()

    core_python = sys.executable
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for docs in args.docs:
        docs_path = Path(docs)
        if not docs_path.exists():
            print(f"Missing docs input: {docs_path}")
            raise SystemExit(2)

        prefix = output_prefix_for_docs(args.output_prefix, docs_path)
        chunks = output_dir / f"{prefix}_chunks.jsonl"
        stats = output_dir / f"{prefix}_run_stats.json"
        rule = output_dir / f"{prefix}_labeled_rule_based.jsonl"
        lexical = output_dir / f"{prefix}_labeled_lexical.jsonl"
        embedding = output_dir / f"{prefix}_labeled_embedding_minilm.jsonl"
        hybrid = output_dir / f"{prefix}_labeled_hybrid.jsonl"

        run_command(
            [
                core_python,
                "scripts/sample_fineweb_chunks.py",
                "--local-input",
                str(docs_path),
                "--max-docs",
                str(args.max_docs),
                "--out",
                str(chunks),
                "--stats-out",
                str(stats),
            ]
        )
        run_command([core_python, "scripts/validate_chunks.py", "--input", str(chunks)])
        run_command([core_python, "scripts/classify_chunks_rule_based.py", "--input", str(chunks), "--output", str(rule)])
        run_command([core_python, "scripts/validate_chunks.py", "--input", str(rule)])
        run_command(
            [
                core_python,
                "scripts/classify_chunks_lexical_baseline.py",
                "--input",
                str(chunks),
                "--labels",
                args.labels,
                "--output",
                str(lexical),
            ]
        )
        run_command([core_python, "scripts/validate_chunks.py", "--input", str(lexical)])
        run_command(
            [
                args.embedding_python,
                "scripts/classify_chunks_embedding_baseline.py",
                "--input",
                str(chunks),
                "--labels",
                args.labels,
                "--output",
                str(embedding),
                "--model",
                args.minilm_model,
                "--batch-size",
                str(args.batch_size),
                "--top-k",
                str(args.top_k),
            ]
        )
        run_command([core_python, "scripts/validate_chunks.py", "--input", str(embedding)])
        run_command(
            [
                core_python,
                "scripts/build_hybrid_labels.py",
                "--rule-based",
                str(rule),
                "--embedding",
                str(embedding),
                "--output",
                str(hybrid),
                "--min-embedding-confidence",
                str(args.min_embedding_confidence),
            ]
        )
        run_command([core_python, "scripts/validate_chunks.py", "--input", str(hybrid)])
        run_command([core_python, "scripts/inspect_chunks.py", "--input", str(hybrid), "--limit", "10"])


if __name__ == "__main__":
    main()
