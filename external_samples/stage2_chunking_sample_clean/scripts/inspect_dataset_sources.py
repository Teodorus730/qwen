#!/usr/bin/env python3
"""Inspect dataset source registry without running any data loading."""

import argparse
import json
from pathlib import Path


def load_registry(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Missing registry file: {path}")
        raise SystemExit(2)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {path}: {exc}")
        raise SystemExit(2)


def source_kind(source):
    if source.get("kind"):
        return source.get("kind")
    if source.get("local_path"):
        return "local"
    if source.get("hf_dataset"):
        return "hf"
    return "unknown"


def validate_source(source):
    warnings = []
    kind = source_kind(source)
    if not source.get("source_id"):
        warnings.append("missing source_id")
    if not source.get("dataset_label"):
        warnings.append(f"{source.get('source_id', '<missing>')}: missing dataset_label")
    if not source.get("source_type"):
        warnings.append(f"{source.get('source_id', '<missing>')}: missing source_type")
    if kind == "hf" and not source.get("hf_dataset"):
        warnings.append(f"{source.get('source_id')}: hf source with null hf_dataset")
    if kind == "local" and not source.get("local_path"):
        warnings.append(f"{source.get('source_id')}: local source with missing local_path")
    return warnings


def bool_text(value):
    return "true" if bool(value) else "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="config/dataset_sources.json")
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    sources = registry.get("sources", [])
    if not isinstance(sources, list):
        print("Registry field 'sources' must be a list")
        raise SystemExit(2)

    warnings = []
    print("source_id | kind | dataset_label | source_type | enabled_by_default | recommended_max_docs_first_run | needs_verification")
    for source in sources:
        warnings.extend(validate_source(source))
        print(
            " | ".join(
                [
                    str(source.get("source_id")),
                    source_kind(source),
                    str(source.get("dataset_label")),
                    str(source.get("source_type")),
                    bool_text(source.get("enabled_by_default")),
                    str(source.get("recommended_max_docs_first_run")),
                    bool_text(source.get("needs_verification")),
                ]
            )
        )

    print(f"sources_count: {len(sources)}")
    print(f"warnings_count: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
