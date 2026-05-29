"""Run semantic-topic embedding prediction/evaluation configs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def slug_model(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("-", "_")
        .lower()
    )


def run_command(command: list[str], dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--domain-descriptions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_embedding_cache"))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    output_plan = []
    for model in args.models:
        slug = slug_model(model)
        predictions = args.output_dir / f"real_hf_benchmark_v1_semantic_topic_embedding_{slug}.jsonl"
        evaluation = args.output_dir / f"real_hf_benchmark_v1_semantic_topic_embedding_{slug}_eval.json"

        classify_cmd = [
            sys.executable,
            "scripts/classify_semantic_topic_embedding.py",
            "--input",
            str(args.gold),
            "--domain-descriptions",
            str(args.domain_descriptions),
            "--output",
            str(predictions),
            "--model",
            model,
            "--cache-dir",
            str(args.cache_dir),
            "--threshold",
            str(args.threshold),
            "--margin-threshold",
            str(args.margin_threshold),
            "--top-k",
            str(args.top_k),
            "--text-field",
            args.text_field,
        ]
        if args.allow_download:
            classify_cmd.append("--allow-download")
        if args.dry_run:
            classify_cmd.append("--dry-run")

        evaluate_cmd = [
            sys.executable,
            "scripts/evaluate_semantic_topic_predictions.py",
            "--predictions",
            str(predictions),
            "--gold",
            str(args.gold),
            "--output-json",
            str(evaluation),
        ]
        if args.dry_run:
            evaluate_cmd.append("--dry-run")

        output_plan.append(
            {
                "model": model,
                "predictions": str(predictions),
                "evaluation": str(evaluation),
                "classify_command": classify_cmd,
                "evaluate_command": evaluate_cmd,
            }
        )

    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": output_plan}, ensure_ascii=False, indent=2))
        for item in output_plan:
            code = run_command(item["classify_command"], dry_run=False)
            if code != 0:
                raise SystemExit(code)
            code = run_command(item["evaluate_command"], dry_run=False)
            if code != 0:
                raise SystemExit(code)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_plan:
        code = run_command(item["classify_command"], dry_run=False)
        if code != 0:
            raise SystemExit(code)
        code = run_command(item["evaluate_command"], dry_run=False)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
