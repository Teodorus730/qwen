#!/usr/bin/env python3
"""Print full future pipeline commands for one source without executing them."""

import argparse
import json
from pathlib import Path


def load_sources(path):
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Missing registry file: {path}")
        raise SystemExit(2)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {path}: {exc}")
        raise SystemExit(2)
    return registry.get("sources", [])


def kind(source):
    if source.get("source_kind"):
        return source.get("source_kind")
    if source.get("kind"):
        return source.get("kind")
    if source.get("local_path"):
        return "local"
    if source.get("hf_dataset"):
        return "hf"
    return "unknown"


def paths(source_id, out_dir):
    root = Path(out_dir)
    return {
        "chunks": root / f"{source_id}_chunks.jsonl",
        "stats": root / f"{source_id}_run_stats.json",
        "rule": root / f"{source_id}_labeled_rule_based.jsonl",
        "lexical": root / f"{source_id}_labeled_lexical.jsonl",
        "embedding": root / f"{source_id}_labeled_embedding.jsonl",
    }


def sample_command(source, max_docs, out_dir):
    p = paths(source.get("planned_output_prefix") or source.get("source_id"), out_dir)
    command = ["python", "scripts\\sample_fineweb_chunks.py"]
    if kind(source) == "local":
        command += ["--local-input", str(source.get("local_path"))]
    elif kind(source) == "hf":
        command += ["--use-hf-streaming"]
        if source.get("hf_dataset"):
            command += ["--dataset", str(source.get("hf_dataset"))]
        if source.get("hf_config"):
            command += ["--config", str(source.get("hf_config"))]
        if source.get("split"):
            command += ["--split", str(source.get("split"))]
    command += [
        "--dataset-label", str(source.get("dataset_label")),
        "--source-type", str(source.get("source_type")),
    ]
    if source.get("text_field"):
        command += ["--text-field", str(source.get("text_field"))]
    if source.get("id_field"):
        command += ["--id-field", str(source.get("id_field"))]
    command += [
        "--max-docs", str(max_docs),
        "--out", str(p["chunks"]),
        "--stats-out", str(p["stats"]),
    ]
    return command


def print_command(command, commented=False):
    prefix = "# " if commented else ""
    print(prefix + " ".join(command))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="config/dataset_sources.json")
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--out-dir", default="data_samples\\real_samples")
    args = parser.parse_args()

    sources = load_sources(Path(args.registry))
    source = next((item for item in sources if item.get("source_id") == args.source), None)
    if source is None:
        print(f"Unknown source: {args.source}")
        raise SystemExit(2)

    source_id = source.get("source_id")
    output_prefix = source.get("planned_output_prefix") or source_id
    p = paths(output_prefix, args.out_dir)
    if source.get("needs_verification") or (kind(source) == "hf" and not source.get("hf_dataset")):
        print("WARNING: Do not run until dataset id/config is verified.")
    if kind(source) == "hf" and source.get("text_field") is None:
        print("WARNING: HF text_field is null; verify row schema before running.")
    print(f"source_id: {source_id}")
    print(f"source_kind: {kind(source)}")
    print(f"text_field: {source.get('text_field')}")
    print(f"id_field: {source.get('id_field')}")
    print(f"planned_output_prefix: {output_prefix}")
    print("planned_commands:")
    print_command(sample_command(source, args.max_docs, args.out_dir))
    print_command(["python", "scripts\\validate_chunks.py", "--input", str(p["chunks"])])
    print_command(["python", "scripts\\classify_chunks_rule_based.py", "--input", str(p["chunks"]), "--output", str(p["rule"])])
    print_command(["python", "scripts\\classify_chunks_lexical_baseline.py", "--input", str(p["chunks"]), "--labels", "taxonomy\\simple_domain_labels.json", "--output", str(p["lexical"])])
    print_command(["python", "scripts\\validate_chunks.py", "--input", str(p["rule"]), "--require-labels"])
    print_command(["python", "scripts\\validate_chunks.py", "--input", str(p["lexical"]), "--require-labels"])
    print_command(["python", "scripts\\compare_label_runs.py", "--left", str(p["rule"]), "--right", str(p["lexical"]), "--left-name", "rule_based", "--right-name", "lexical"])
    print_command(["python", "scripts\\inspect_chunks.py", "--input", str(p["rule"]), "--limit", "30", "--show-text"])
    print_command(["python", "scripts\\classify_chunks_embedding_baseline.py", "--input", str(p["chunks"]), "--labels", "taxonomy\\simple_domain_labels.json", "--output", str(p["embedding"]), "--model", "sentence-transformers/all-MiniLM-L6-v2"], commented=True)


if __name__ == "__main__":
    main()
