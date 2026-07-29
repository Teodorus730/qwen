from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.config import ExperimentConfig, load_config, resolve
from src.data import cyclic_batches, load_or_build_blocks, materialize_slice
from src.runtime import (
    build_optimizer,
    config_as_dict,
    configure_project_environment,
    environment_report,
    load_model,
    load_tokenizer,
    project_data_paths,
    set_seed,
    write_json,
)

MIB = 1024**2


def training_step(
    model,
    optimizer,
    batch,
    cfg: ExperimentConfig,
    *,
    probe_device_memory: bool = False,
) -> tuple[float, list[int]]:
    free_samples: list[int] = []

    def sample_free() -> None:
        if probe_device_memory:
            free_samples.append(torch.cuda.mem_get_info()[0])

    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=batch, labels=batch, use_cache=False)
    sample_free()
    output.loss.backward()
    sample_free()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg.training.gradient_clip
    )
    sample_free()
    optimizer.step()
    sample_free()
    return float(output.loss.detach()), free_samples


def worker(args) -> int:
    cfg = load_config(args.config)
    configure_project_environment(cfg.root)
    set_seed(cfg.training.seed)
    result_path = Path(args.result_json).resolve()
    common = {
        "batch_size": args.worker_batch_size,
        "sequence_length": cfg.data.sequence_length,
        "optimizer": cfg.training.optimizer,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "warmup_steps": cfg.benchmark.warmup_steps,
        "measured_steps": cfg.benchmark.measured_steps,
    }
    if not torch.cuda.is_available():
        write_json(result_path, {**common, "status": "error", "error": "no CUDA"})
        return 1

    try:
        torch.cuda.set_per_process_memory_fraction(
            cfg.training.cuda_memory_fraction
        )
        device = torch.device("cuda")
        free_before, total = torch.cuda.mem_get_info()
        tokenizer = load_tokenizer(cfg)
        slice_path, blocks_path = project_data_paths(cfg)
        train_blocks, _, metadata = load_or_build_blocks(
            cfg.data, slice_path, tokenizer, blocks_path
        )
        batch_iter = cyclic_batches(
            train_blocks,
            args.worker_batch_size,
            seed=cfg.training.seed,
        )

        model = load_model(cfg, device)
        model.train()
        model_allocated = torch.cuda.memory_allocated()
        optimizer = build_optimizer(cfg, model)

        probe_free_samples: list[int] = []
        for warmup_index in range(cfg.benchmark.warmup_steps):
            batch = next(batch_iter).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            _, samples = training_step(
                model,
                optimizer,
                batch,
                cfg,
                probe_device_memory=(
                    warmup_index == cfg.benchmark.warmup_steps - 1
                ),
            )
            probe_free_samples.extend(samples)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        static_allocated = torch.cuda.memory_allocated()
        static_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        durations: list[float] = []
        losses: list[float] = []
        for _ in range(cfg.benchmark.measured_steps):
            batch = next(batch_iter).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, _ = training_step(model, optimizer, batch, cfg)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            losses.append(loss)

        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        device_peak_used = total - min(probe_free_samples)
        baseline_device_used = total - free_before
        tokens_per_step = args.worker_batch_size * cfg.data.sequence_length
        mean_step_seconds = statistics.fmean(durations)
        result = {
            **common,
            "status": "ok",
            "gpu_name": torch.cuda.get_device_name(0),
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "packed_blocks_available": metadata["train_blocks"],
            "tokens_per_step": tokens_per_step,
            "mean_loss": statistics.fmean(losses),
            "step_seconds": durations,
            "mean_step_seconds": mean_step_seconds,
            "median_step_seconds": statistics.median(durations),
            "p90_step_seconds": float(np.percentile(durations, 90)),
            "tokens_per_second": tokens_per_step / mean_step_seconds,
            "model_allocated_mib": model_allocated / MIB,
            "static_allocated_mib": static_allocated / MIB,
            "static_reserved_mib": static_reserved / MIB,
            "peak_allocated_mib": peak_allocated / MIB,
            "peak_reserved_mib": peak_reserved / MIB,
            "dynamic_peak_mib": (peak_allocated - static_allocated) / MIB,
            "baseline_device_used_mib": baseline_device_used / MIB,
            "device_peak_used_mib": device_peak_used / MIB,
            "incremental_device_peak_mib": (
                device_peak_used - baseline_device_used
            )
            / MIB,
            "free_vram_before_process_mib": free_before / MIB,
            "total_vram_mib": total / MIB,
        }
        write_json(result_path, result)
        print(f"[worker result] {json.dumps(result)}", flush=True)
        return 0
    except torch.OutOfMemoryError as error:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        result = {**common, "status": "oom", "error": str(error)}
        write_json(result_path, result)
        print(f"[worker OOM] batch={args.worker_batch_size}: {error}", flush=True)
        return 0
    except Exception as error:
        result = {
            **common,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(result_path, result)
        print(f"[worker error] {result}", flush=True)
        return 1


def linear_fit(rows: list[dict], key: str) -> dict | None:
    ok = [row for row in rows if row["status"] == "ok"]
    if len(ok) < 2:
        return None
    x = np.array([row["batch_size"] for row in ok], dtype=float)
    y = np.array([row[key] for row in ok], dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    residual = float(np.square(y - prediction).sum())
    total = float(np.square(y - y.mean()).sum())
    return {
        "metric": key,
        "slope_mib_per_sample": float(slope),
        "intercept_mib": float(intercept),
        "r_squared": 1.0 - residual / total if total else 1.0,
        "fit_batch_sizes": x.astype(int).tolist(),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "batch_size",
        "status",
        "gpu_name",
        "sequence_length",
        "tokens_per_step",
        "mean_step_seconds",
        "median_step_seconds",
        "p90_step_seconds",
        "tokens_per_second",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "baseline_device_used_mib",
        "device_peak_used_mib",
        "incremental_device_peak_mib",
        "mean_loss",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: list[dict], fit: dict | None) -> None:
    import matplotlib.pyplot as plt

    ok = [row for row in rows if row["status"] == "ok"]
    if not ok:
        return
    batches = np.array([row["batch_size"] for row in ok])
    peaks = np.array([row["incremental_device_peak_mib"] / 1024 for row in ok])
    throughput = np.array([row["tokens_per_second"] for row in ok])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(batches, peaks, "o-", label="measured peak")
    if fit:
        fit_y = (
            fit["slope_mib_per_sample"] * batches + fit["intercept_mib"]
        ) / 1024
        axes[0].plot(
            batches,
            fit_y,
            "--",
            label=f"linear fit, R²={fit['r_squared']:.4f}",
        )
    axes[0].set(
        xlabel="Micro-batch size",
        ylabel="Incremental device peak, GiB",
        title="Training memory",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(batches, throughput, "o-", color="#d95f02")
    axes[1].set(
        xlabel="Micro-batch size",
        ylabel="Packed tokens/s",
        title="End-to-end training throughput",
    )
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        f"{ok[0]['gpu_name']} — StellaAthena Qwen3-0.6B, "
        f"seq={ok[0]['sequence_length']}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def orchestrator(args) -> int:
    cfg = load_config(args.config)
    configure_project_environment(cfg.root)
    results_dir = resolve(cfg.root, cfg.benchmark.results_dir)
    workers_dir = results_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)

    slice_path, blocks_path = project_data_paths(cfg)
    slice_report = materialize_slice(cfg.data, slice_path)
    tokenizer = load_tokenizer(cfg)
    _, _, metadata = load_or_build_blocks(
        cfg.data, slice_path, tokenizer, blocks_path
    )
    print(f"[prepared] {slice_report}", flush=True)
    print(f"[prepared] {metadata}", flush=True)

    batch_sizes = (
        tuple(args.batch_sizes)
        if args.batch_sizes
        else cfg.benchmark.batch_sizes
    )
    rows: list[dict] = []
    script = Path(__file__).resolve()
    for batch_size in batch_sizes:
        result_path = workers_dir / f"batch_{batch_size}.json"
        # Never accept a stale success left by an earlier worker that now
        # crashes before producing its own result.
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(script),
            "--config",
            str(Path(args.config).resolve()),
            "--worker-batch-size",
            str(batch_size),
            "--result-json",
            str(result_path),
        ]
        print(f"\n[benchmark] batch={batch_size}", flush=True)
        completed = subprocess.run(command, cwd=cfg.root, env=os.environ.copy())
        if not result_path.exists():
            raise RuntimeError(
                f"Worker batch={batch_size} exited {completed.returncode} "
                "without a result file"
            )
        row = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(row)
        if row["status"] == "error":
            raise RuntimeError(f"Worker batch={batch_size} failed: {row['error']}")
        if row["status"] == "oom" and cfg.benchmark.stop_after_oom:
            break

    fit = linear_fit(rows, "incremental_device_peak_mib")
    summary = {
        "model": cfg.model.id,
        "model_revision": cfg.model.revision,
        "dataset": f"{cfg.data.dataset_id}/{cfg.data.subset}",
        "config": config_as_dict(cfg),
        "data": metadata,
        "environment": environment_report(cfg.root),
        "linear_fit": fit,
        "rows": rows,
    }
    write_json(results_dir / "benchmark.json", summary)
    write_json(results_dir / "environment.json", summary["environment"])
    save_csv(results_dir / "benchmark.csv", rows)
    save_plot(results_dir / "benchmark.png", rows, fit)
    print(f"\n[done] {results_dir / 'benchmark.json'}", flush=True)
    print(f"[fit] {fit}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolated micro-batch VRAM/throughput benchmark."
    )
    parser.add_argument("--config", default="configs/vast_16gb.yaml")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--worker-batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_batch_size is not None:
        if not args.result_json:
            parser.error("--result-json is required in worker mode")
        return worker(args)
    return orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
