"""Reproducible lm-evaluation-harness baseline runs for Qwen base checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"
SUITES = {
    "core": ("wikitext", "lambada_openai", "hellaswag", "arc_easy", "piqa"),
    "extended_loglikelihood": ("ceval-valid", "mmmlu"),
    "extended_generation": ("mmlu_redux_generative", "mmlu_pro"),
    "instruction_control": ("ifeval",),
}
REPRODUCIBILITY_PACKAGES = (
    "torch",
    "transformers",
    "lm-eval",
    "datasets",
    "huggingface_hub",
    "tokenizers",
    "accelerate",
    "safetensors",
)


def package_version(name: str) -> str:
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def select_backend(requested: str) -> tuple[str, str, str, float | None]:
    """Return lm-eval device, backend label, dtype, and accelerator memory."""
    cuda_available = torch.cuda.is_available()
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

    if requested == "auto":
        requested = "cuda" if cuda_available else "xpu" if xpu_available else "cpu"

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but is unavailable.")
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return "cuda:0", "cuda", dtype, memory

    if requested == "xpu":
        if not xpu_available:
            raise RuntimeError("XPU was requested but is unavailable.")
        # Arc supports bf16, but retain an explicit capability fallback.
        supports_bf16 = getattr(torch.xpu, "is_bf16_supported", lambda: True)()
        dtype = "bfloat16" if supports_bf16 else "float16"
        properties = torch.xpu.get_device_properties(0)
        memory = getattr(properties, "total_memory", 0) / 1024**3 or None
        return "xpu:0", "xpu", dtype, memory

    if requested == "cpu":
        return "cpu", "cpu", "float32", None

    raise ValueError(f"Unsupported backend: {requested}")


def accelerator_name(backend: str) -> str:
    if backend == "cuda":
        return torch.cuda.get_device_name(0)
    if backend == "xpu":
        return torch.xpu.get_device_name(0)
    return platform.processor() or "CPU"


def model_revision(model_id: str) -> str | None:
    if Path(model_id).exists():
        return None
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception as exc:  # Network metadata must not prevent evaluation.
        print(f"Warning: unable to resolve model revision: {exc}", file=sys.stderr)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suite", choices=SUITES, help="Named benchmark suite; defaults to core.")
    parser.add_argument("--tasks", nargs="+", help="Exact lm-eval task or group IDs; overrides suite selection.")
    parser.add_argument("--backend", choices=("auto", "cuda", "xpu", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-fewshot", type=int, help="Explicit few-shot override; omit to preserve task defaults.")
    parser.add_argument("--max-length", type=int, help="Model context limit; core defaults to 2048.")
    parser.add_argument("--limit", type=float, help="Limit examples per task; use only for smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("pretrain_benchmarks/results"))
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument(
        "--write-baseline-summary",
        action="store_true",
        help="Write a compact, versionable JSON summary for a completed full suite.",
    )
    return parser.parse_args()


def resolve_tasks_and_protocol(args: argparse.Namespace) -> tuple[list[str], str | None, int | None, str, int | None, str]:
    if args.suite and args.tasks:
        raise ValueError("Use either --suite or --tasks, not both.")
    suite = args.suite or (None if args.tasks else "core")
    tasks = list(args.tasks) if args.tasks else list(SUITES[suite])

    if args.num_fewshot is not None:
        num_fewshot, fewshot_source = args.num_fewshot, "explicit_override"
    elif suite == "core":
        num_fewshot, fewshot_source = 0, "core_suite_default_zero_shot"
    else:
        num_fewshot, fewshot_source = None, "native_task_default"

    if args.max_length is not None:
        max_length, max_length_source = args.max_length, "explicit_override"
    elif suite == "core":
        max_length, max_length_source = 2048, "core_suite_default"
    else:
        max_length, max_length_source = None, "model_or_task_default"
    return tasks, suite, num_fewshot, fewshot_source, max_length, max_length_source


def write_baseline_summary(run_dir: Path, run_metadata: dict[str, Any]) -> Path:
    result_files = sorted(run_dir.rglob("results_*.json"), key=lambda path: path.stat().st_mtime)
    if not result_files:
        raise FileNotFoundError(f"No lm-eval results JSON found under {run_dir}")
    raw_results = json.loads(result_files[-1].read_text(encoding="utf-8"))
    task_configs = raw_results.get("configs", {})
    datasets: dict[str, dict[str, Any]] = {}
    for task, config in task_configs.items():
        if not isinstance(config, dict):
            continue
        dataset_kwargs = config.get("dataset_kwargs") or {}
        datasets[task] = {
            "dataset_path": config.get("dataset_path"),
            "dataset_name": config.get("dataset_name"),
            "dataset_kwargs": dataset_kwargs,
            "dataset_revision": dataset_kwargs.get("revision"),
        }
    summary = {
        "run_id": run_metadata["run_id"],
        "timestamp_utc": run_metadata["timestamp_utc"],
        "model_id": run_metadata["model_id"],
        "model_revision": run_metadata["model_revision"],
        "backend": run_metadata["backend"],
        "device": run_metadata["device"],
        "accelerator_name": run_metadata["accelerator_name"],
        "dtype": run_metadata["dtype"],
        "torch": run_metadata["torch"],
        "transformers": run_metadata["transformers"],
        "lm_eval": run_metadata["lm_eval"],
        "tasks": run_metadata["tasks"],
        "num_fewshot": run_metadata["num_fewshot"],
        "seed": run_metadata["seed"],
        "batch_size_policy": run_metadata["batch_size"],
        "model_args": run_metadata["model_args"],
        "command": run_metadata["command"],
        "task_versions": raw_results.get("versions", {}),
        "dataset_identifiers": datasets,
        # This is the complete small lm-eval aggregate/config JSON, never samples.
        "lm_eval_aggregate": raw_results,
    }
    destination = Path("pretrain_benchmarks/baseline_results") / run_metadata["model_id"].replace("/", "__")
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / f"{run_metadata['run_id']}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    environment_path = summary_path.with_suffix(".environment.txt")
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    environment_path.write_text(frozen.stdout, encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    tasks, suite, num_fewshot, fewshot_source, max_length, max_length_source = resolve_tasks_and_protocol(args)
    device, backend, selected_dtype, vram_gb = select_backend(args.backend)
    dtype = selected_dtype if args.dtype == "auto" else args.dtype
    if backend == "cpu" and dtype != "float32":
        raise ValueError("CPU runs require float32; use CPU only for diagnostics.")
    if args.write_baseline_summary and args.limit is not None:
        raise ValueError("--write-baseline-summary is for an unbounded full suite; do not combine it with --limit.")

    revision = model_revision(args.model)
    if args.write_baseline_summary and revision is None:
        raise RuntimeError(
            "A versioned full baseline requires an exact model revision SHA; retry when it can be resolved."
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / args.model.replace("/", "__") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_arg_parts = [f"pretrained={args.model}"]
    if revision is not None:
        model_arg_parts.append(f"revision={revision}")
    model_arg_parts.extend((f"dtype={dtype}", "backend=causal"))
    if max_length is not None:
        model_arg_parts.append(f"max_length={max_length}")
    model_args = ",".join(model_arg_parts)
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--device",
        device,
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--output_path",
        str(run_dir),
    ]
    if num_fewshot is not None:
        command.extend(("--num_fewshot", str(num_fewshot)))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.log_samples:
        command.append("--log_samples")

    run_metadata: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "model_revision": revision,
        "device": device,
        "backend": backend,
        "accelerator_name": accelerator_name(backend),
        "vram_gb": vram_gb,
        "dtype": dtype,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "lm_eval": package_version("lm-eval"),
        "package_versions": {
            package: package_version(package) for package in REPRODUCIBILITY_PACKAGES
        },
        "suite": suite,
        "tasks": tasks,
        "batch_size": args.batch_size,
        "num_fewshot": num_fewshot,
        "fewshot_source": fewshot_source,
        "max_length": max_length,
        "max_length_source": max_length_source,
        "limit": args.limit,
        "seed": args.seed,
        "model_args": model_args,
        "command": command,
    }
    (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print("Running:", subprocess.list2cmdline(command))
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(command, cwd=Path.cwd(), env=child_env)
    run_metadata["return_code"] = completed.returncode
    run_metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    if completed.returncode == 0 and args.write_baseline_summary:
        print(f"Versionable summary: {write_baseline_summary(run_dir, run_metadata)}")
    print(f"Artifacts: {run_dir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
