#!/usr/bin/env python3
"""Print future real sample commands without executing them."""

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


def quote(value):
    return str(value)


def output_paths(source_id, out_dir):
    root = Path(out_dir)
    return root / f"{source_id}_chunks.jsonl", root / f"{source_id}_run_stats.json"


def sample_command(source, max_docs, out_dir):
    source_id = source.get("planned_output_prefix") or source.get("source_id")
    out_path, stats_path = output_paths(source_id, out_dir)
    command = [
        "python",
        "scripts\\sample_fineweb_chunks.py",
    ]

    if kind(source) == "local":
        command += ["--local-input", quote(source.get("local_path"))]
    elif kind(source) == "hf":
        command += ["--use-hf-streaming"]
        if source.get("hf_dataset"):
            command += ["--dataset", quote(source.get("hf_dataset"))]
        if source.get("hf_config"):
            command += ["--config", quote(source.get("hf_config"))]
        if source.get("split"):
            command += ["--split", quote(source.get("split"))]

    command += [
        "--dataset-label",
        quote(source.get("dataset_label")),
        "--source-type",
        quote(source.get("source_type")),
    ]
    if source.get("text_field"):
        command += ["--text-field", quote(source.get("text_field"))]
    if source.get("id_field"):
        command += ["--id-field", quote(source.get("id_field"))]
    command += [
        "--max-docs",
        str(max_docs),
        "--out",
        quote(out_path),
        "--stats-out",
        quote(stats_path),
    ]
    return command


def print_plan(source, max_docs, out_dir):
    print(f"source_id: {source.get('source_id')}")
    print(f"source_kind: {kind(source)}")
    print(f"dataset_label: {source.get('dataset_label')}")
    print(f"source_type: {source.get('source_type')}")
    print(f"text_field: {source.get('text_field')}")
    print(f"id_field: {source.get('id_field')}")
    print(f"planned_output_prefix: {source.get('planned_output_prefix')}")
    if source.get("needs_verification") or (kind(source) == "hf" and not source.get("hf_dataset")):
        print("WARNING: Do not run until dataset id/config is verified.")
    if kind(source) == "hf" and source.get("text_field") is None:
        print("WARNING: HF text_field is null; verify row schema before running.")
    print("suggested_command:")
    print(" ".join(sample_command(source, max_docs, out_dir)))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="config/dataset_sources.json")
    parser.add_argument("--source")
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--out-dir", default="data_samples\\real_samples")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    sources = load_sources(Path(args.registry))
    selected = []
    if args.all:
        selected = sources
    elif args.source:
        selected = [source for source in sources if source.get("source_id") == args.source]
        if not selected:
            print(f"Unknown source: {args.source}")
            raise SystemExit(2)
    else:
        selected = [source for source in sources if source.get("enabled_by_default")]

    for source in selected:
        if not args.include_disabled and not args.all and not args.source and not source.get("enabled_by_default"):
            print(f"Skipping disabled source: {source.get('source_id')}")
            continue
        print_plan(source, args.max_docs, args.out_dir)


if __name__ == "__main__":
    main()
