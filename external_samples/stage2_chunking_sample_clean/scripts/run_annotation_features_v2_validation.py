#!/usr/bin/env python
"""Run annotation_v2 feature validation over multiple files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate annotation_v2 validation summaries.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input feature JSONL files.")
    parser.add_argument("--summary-json", required=True, help="Aggregate summary JSON output.")
    parser.add_argument("--max-errors", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = Path(__file__).with_name("validate_annotation_features_v2.py")
    summaries: list[dict[str, Any]] = []
    exit_code = 0

    for input_arg in args.inputs:
        command = [
            sys.executable,
            str(validator),
            "--input",
            input_arg,
            "--max-errors",
            str(args.max_errors),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.stdout.strip():
            summary = json.loads(completed.stdout)
            summaries.append(summary)
        if completed.returncode != 0:
            exit_code = completed.returncode
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)

    aggregate = {
        "inputs": args.inputs,
        "files_checked": len(summaries),
        "records_checked": sum(item["records_checked"] for item in summaries),
        "errors_count": sum(item["errors_count"] for item in summaries),
        "warnings_count": sum(item["warnings_count"] for item in summaries),
        "files": summaries,
    }
    output_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
