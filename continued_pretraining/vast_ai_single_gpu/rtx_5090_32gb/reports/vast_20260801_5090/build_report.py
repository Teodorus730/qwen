"""Rebuild the RTX 5090 experiment figures and derived metrics.

The script reads the original RTX 5090 metrics export, the ignored full RTX
5070 Ti archive (or the already extracted analysis cache), and the local RTX
3090 Ti benchmark.  It does not read values back from REPORT.md.
"""

from __future__ import annotations

import io
import json
import math
import re
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
CONTINUED = REPO_ROOT / "continued_pretraining"
VAST_ROOT = CONTINUED / "vast_ai_single_gpu"
RTX_5090_ROOT = VAST_ROOT / "rtx_5090_32gb"
RTX_5070_ROOT = VAST_ROOT / "rtx_5070_ti_16gb"
RTX_3090_ROOT = CONTINUED / "local_rtx3090ti"

COLORS = {
    "RTX 3090 Ti": "#667085",
    "RTX 5070 Ti": "#2E90FA",
    "RTX 5090": "#F04438",
    "good": "#12B76A",
    "warning": "#F79009",
    "ink": "#101828",
    "muted": "#667085",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#D0D5DD",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#EAECF0",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_directory(pattern: str) -> Path:
    matches = sorted(path for path in RTX_5090_ROOT.glob(pattern) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"No directory matches: {RTX_5090_ROOT / pattern}")
    return matches[-1]


def load_5090() -> dict[str, Any]:
    metrics = latest_directory("exports/qwen_vast_5090_metrics_*")
    run = metrics / "outputs/vast_5090_32gb_10m"
    full = latest_directory("exports/qwen_vast_5090_full_*")
    log_files = sorted((full / "logs").glob("run_*.log"))
    pipeline_seconds = parse_pipeline_seconds(log_files[-1].read_text(encoding="utf-8", errors="replace"))
    return {
        "benchmark": read_json(metrics / "results/benchmark.json"),
        "environment": read_json(metrics / "results/environment.json"),
        "summary": read_json(run / "summary.json"),
        "config": read_json(run / "config.json"),
        "train": read_jsonl(run / "train_log.jsonl"),
        "eval": read_jsonl(run / "eval_log.jsonl"),
        "pipeline_seconds": pipeline_seconds,
        "metrics_source": str(metrics.relative_to(REPO_ROOT)).replace("\\", "/"),
        "full_source": str(full.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def load_5070_from_tar(archive: Path) -> dict[str, Any]:
    wanted = {
        "benchmark": "results/benchmark.json",
        "environment": "results/environment.json",
        "summary": "outputs/vast_16gb_10m/summary.json",
        "config": "outputs/vast_16gb_10m/config.json",
        "train": "outputs/vast_16gb_10m/train_log.jsonl",
        "eval": "outputs/vast_16gb_10m/eval_log.jsonl",
    }
    loaded: dict[str, Any] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        for key, name in wanted.items():
            stream = bundle.extractfile(members[name])
            if stream is None:
                raise FileNotFoundError(f"Missing {name} in {archive}")
            text = io.TextIOWrapper(stream, encoding="utf-8").read()
            loaded[key] = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if name.endswith(".jsonl")
                else json.loads(text)
            )
    return loaded


def load_5070() -> dict[str, Any]:
    cache = REPO_ROOT / ".dist/report_5090_sources/rtx5070"
    archive = RTX_5070_ROOT / "qwen_vast_full_20260729T161440Z.tar.gz"
    if (cache / "results/benchmark.json").exists():
        run = cache / "outputs/vast_16gb_10m"
        loaded = {
            "benchmark": read_json(cache / "results/benchmark.json"),
            "environment": read_json(cache / "results/environment.json"),
            "summary": read_json(run / "summary.json"),
            "config": read_json(run / "config.json"),
            "train": read_jsonl(run / "train_log.jsonl"),
            "eval": read_jsonl(run / "eval_log.jsonl"),
        }
    else:
        if not archive.exists():
            raise FileNotFoundError(
                "The ignored RTX 5070 Ti full archive is required to rebuild comparative plots: "
                f"{archive}"
            )
        loaded = load_5070_from_tar(archive)

    previous_derived = read_json(RTX_5070_ROOT / "reports/vast_20260729_5070ti/derived_metrics.json")
    loaded["pipeline_seconds"] = previous_derived["pipeline_seconds"]
    loaded["source"] = str(archive.relative_to(REPO_ROOT)).replace("\\", "/")
    return loaded


def load_3090() -> dict[str, Any]:
    return {
        "benchmark": read_json(RTX_3090_ROOT / "results/benchmark.json"),
        "environment": read_json(RTX_3090_ROOT / "results/environment.json"),
        "smoke": read_json(RTX_3090_ROOT / "results/training_smoke.json"),
        "source": str((RTX_3090_ROOT / "results").relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def parse_pipeline_seconds(text: str) -> int:
    start_match = re.search(r"\[run\] UTC start: ([^\s]+)", text)
    finish_match = re.search(r"\[run\] UTC finish: ([^\s]+)", text)
    if not start_match or not finish_match:
        raise ValueError("Could not find UTC start/finish markers in run log")
    start = datetime.fromisoformat(start_match.group(1))
    finish = datetime.fromisoformat(finish_match.group(1))
    return int((finish - start).total_seconds())


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def training_metrics(run: dict[str, Any]) -> dict[str, Any]:
    train = run["train"]
    evaluations = run["eval"]
    summary = run["summary"]
    losses = array(train, "loss")
    grad_norm = array(train, "grad_norm")
    elapsed = array(train, "session_elapsed_seconds")
    tokens = array(train, "tokens_processed")
    update_seconds = np.diff(np.r_[0.0, elapsed])
    tokens_per_update = int(tokens[0])
    update_tps = tokens_per_update / update_seconds
    stable_tps = update_tps[20:]

    baseline = evaluations[0]
    final = evaluations[-1]
    best = min(evaluations, key=lambda row: row["validation_loss"])
    periodic_eval_seconds = np.asarray([row["evaluation_seconds"] for row in evaluations[1:]], dtype=float)
    total_gain = baseline["validation_loss"] - final["validation_loss"]
    by_step = {row["update_step"]: row for row in evaluations}

    config = run["config"]
    train_data_tokens = run["benchmark"]["data"]["train_tokens"]
    validation_data_tokens = run["benchmark"]["data"]["validation_tokens"]
    final_tokens = int(tokens[-1])
    return {
        "train_rows": len(train),
        "eval_rows": len(evaluations),
        "periodic_eval_seconds_median": float(np.median(periodic_eval_seconds)),
        "all_eval_seconds_total": float(sum(row["evaluation_seconds"] for row in evaluations)),
        "pipeline_seconds": run["pipeline_seconds"],
        "training_summary_seconds": summary["session_seconds"],
        "training_log_seconds": float(elapsed[-1]),
        "training_tps_log_final": float(train[-1]["session_tokens_per_second"]),
        "training_tps_including_final_eval_save": final_tokens / summary["session_seconds"],
        "pipeline_effective_tps": final_tokens / run["pipeline_seconds"],
        "median_update_tps": float(np.median(stable_tps)),
        "p10_update_tps": float(np.percentile(stable_tps, 10)),
        "p90_update_tps": float(np.percentile(stable_tps, 90)),
        "first_100_train_loss_mean": float(np.mean(losses[:100])),
        "last_100_train_loss_mean": float(np.mean(losses[-100:])),
        "train_loss_mean_delta_pct": float((np.mean(losses[-100:]) / np.mean(losses[:100]) - 1) * 100),
        "train_loss_min": float(np.min(losses)),
        "train_loss_min_step": int(np.argmin(losses) + 1),
        "grad_norm_max": float(np.max(grad_norm)),
        "grad_norm_median": float(np.median(grad_norm)),
        "grad_clipped_fraction": float(np.mean(grad_norm > config["training"]["gradient_clip"])),
        "grad_norm_first100_mean": float(np.mean(grad_norm[:100])),
        "grad_norm_last100_mean": float(np.mean(grad_norm[-100:])),
        "lr_peak": float(max(row["learning_rate"] for row in train)),
        "lr_peak_step": int(max(train, key=lambda row: row["learning_rate"])["update_step"]),
        "baseline_loss": baseline["validation_loss"],
        "final_loss": final["validation_loss"],
        "loss_abs_change": final["validation_loss"] - baseline["validation_loss"],
        "loss_rel_change_pct": (final["validation_loss"] / baseline["validation_loss"] - 1) * 100,
        "baseline_ppl": baseline["validation_ppl"],
        "final_ppl": final["validation_ppl"],
        "ppl_abs_change": final["validation_ppl"] - baseline["validation_ppl"],
        "ppl_rel_change_pct": (final["validation_ppl"] / baseline["validation_ppl"] - 1) * 100,
        "best_eval_step": int(best["update_step"]),
        "best_eval_tokens": int(best["tokens_processed"]),
        "best_eval_loss": best["validation_loss"],
        "best_eval_ppl": best["validation_ppl"],
        "final_minus_best_loss": final["validation_loss"] - best["validation_loss"],
        "gain_by_step100_fraction": (baseline["validation_loss"] - by_step[100]["validation_loss"]) / total_gain,
        "gain_by_step200_fraction": (baseline["validation_loss"] - by_step[200]["validation_loss"]) / total_gain,
        "unique_train_tokens": train_data_tokens,
        "consumed_unique_fraction": final_tokens / train_data_tokens,
        "validation_evaluated_fraction": baseline["validation_tokens"] / validation_data_tokens,
        "tokens_processed": final_tokens,
        "target_tokens": summary["target_tokens"],
        "overshoot_tokens": final_tokens - summary["target_tokens"],
        "overshoot_pct": (final_tokens / summary["target_tokens"] - 1) * 100,
        "train_peak_allocated_mib": max(row["peak_allocated_mib"] for row in train),
        "train_peak_reserved_mib": max(row["peak_reserved_mib"] for row in train),
        "micro_batch_size": config["training"]["micro_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "tokens_per_update": config["derived"]["tokens_per_update"],
    }


def benchmark_metrics(benchmark: dict[str, Any]) -> dict[str, Any]:
    ok = [row for row in benchmark["rows"] if row["status"] == "ok"]
    failed = [row for row in benchmark["rows"] if row["status"] != "ok"]
    best = max(ok, key=lambda row: row["tokens_per_second"])
    last = max(ok, key=lambda row: row["batch_size"])
    return {
        "fit_slope_mib_per_sample": benchmark["linear_fit"]["slope_mib_per_sample"],
        "fit_intercept_mib": benchmark["linear_fit"]["intercept_mib"],
        "fit_r_squared": benchmark["linear_fit"]["r_squared"],
        "successful_batches": [row["batch_size"] for row in ok],
        "first_oom_batch": failed[0]["batch_size"] if failed else None,
        "max_successful_batch": last["batch_size"],
        "max_successful_headroom_mib": last["total_vram_mib"] - last["device_peak_used_mib"],
        "best_throughput_batch": best["batch_size"],
        "best_throughput_tps": best["tokens_per_second"],
        "rows": [
            {
                "batch_size": row["batch_size"],
                "status": row["status"],
                **(
                    {
                        "tokens_per_second": row["tokens_per_second"],
                        "mean_step_seconds": row["mean_step_seconds"],
                        "incremental_device_peak_mib": row["incremental_device_peak_mib"],
                        "device_peak_used_mib": row["device_peak_used_mib"],
                        "total_vram_mib": row["total_vram_mib"],
                        "physical_headroom_mib": row["total_vram_mib"] - row["device_peak_used_mib"],
                    }
                    if row["status"] == "ok"
                    else {}
                ),
            }
            for row in benchmark["rows"]
        ],
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(SCRIPT_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def series(run: dict[str, Any]) -> dict[str, np.ndarray]:
    train = run["train"]
    elapsed = array(train, "session_elapsed_seconds")
    tokens = array(train, "tokens_processed")
    update_seconds = np.diff(np.r_[0.0, elapsed])
    return {
        "step": array(train, "update_step"),
        "tokens_m": tokens / 1e6,
        "loss": array(train, "loss"),
        "lr": array(train, "learning_rate"),
        "grad": array(train, "grad_norm"),
        "elapsed": elapsed,
        "update_seconds": update_seconds,
        "update_tps": tokens[0] / update_seconds,
        "cumulative_tps": array(train, "session_tokens_per_second"),
    }


def plot_training_dynamics(run: dict[str, Any], metrics: dict[str, Any]) -> None:
    s = series(run)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("RTX 5090: динамика 10M-token continued pretraining", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(s["tokens_m"], s["loss"], color="#FDA29B", alpha=0.45, linewidth=0.8, label="Loss на update")
    ax.plot(s["tokens_m"], rolling(s["loss"], 50), color=COLORS["RTX 5090"], linewidth=2.2, label="Скользящее среднее, 50")
    ax.set(title="Training loss", xlabel="Обработано токенов, млн", ylabel="Cross-entropy")
    ax.legend()
    style_axis(ax)

    ax = axes[0, 1]
    ax.plot(s["tokens_m"], s["lr"] * 1e5, color="#7F56D9", linewidth=2)
    ax.axvline(metrics["lr_peak_step"] * metrics["tokens_per_update"] / 1e6, color="#B692F6", linestyle="--", linewidth=1)
    ax.set(title="Learning-rate schedule", xlabel="Обработано токенов, млн", ylabel="Learning rate, ×10⁻⁵")
    style_axis(ax)

    ax = axes[1, 0]
    ax.scatter(s["tokens_m"], s["grad"], s=9, color="#84CAFF", alpha=0.45, label="Grad norm")
    ax.plot(s["tokens_m"], rolling(s["grad"], 50), color=COLORS["RTX 5070 Ti"], linewidth=2, label="Среднее, 50")
    ax.axhline(1.0, color=COLORS["warning"], linestyle="--", linewidth=1.5, label="Порог clipping = 1.0")
    ax.set(title=f"Норма градиента · выше порога {metrics['grad_clipped_fraction']:.1%} updates", xlabel="Обработано токенов, млн", ylabel="L2 norm")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2)
    style_axis(ax)

    ax = axes[1, 1]
    ax.plot(s["tokens_m"], rolling(s["update_tps"], 50), color=COLORS["good"], linewidth=2, label="Update throughput, среднее 50")
    ax.plot(s["tokens_m"], s["cumulative_tps"], color=COLORS["ink"], linewidth=1.6, label="Накопительный throughput")
    ax.set(title="Скорость обучения после стартового разгона", xlabel="Обработано токенов, млн", ylabel="Токенов/с")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend()
    style_axis(ax)
    save(fig, "01_training_dynamics_5090.png")


def plot_validation(runs: dict[str, dict[str, Any]], metrics: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Валидация: RTX 5070 Ti и RTX 5090", fontsize=17, fontweight="bold")
    for label in ("RTX 5070 Ti", "RTX 5090"):
        evaluations = runs[label]["eval"]
        x = array(evaluations, "tokens_processed") / 1e6
        loss = array(evaluations, "validation_loss")
        ppl = array(evaluations, "validation_ppl")
        color = COLORS[label]
        axes[0, 0].plot(x, loss, marker="o", markersize=4, color=color, linewidth=2, label=label)
        axes[0, 1].plot(x, ppl, marker="o", markersize=4, color=color, linewidth=2, label=label)
        gain = (loss[0] - loss) / (loss[0] - loss[-1]) * 100
        axes[1, 0].plot(x, gain, marker="o", markersize=4, color=color, linewidth=2, label=label)
        axes[1, 1].plot(x, array(evaluations, "evaluation_seconds"), marker="o", markersize=4, color=color, linewidth=2, label=label)
        best = metrics[label]
        axes[0, 0].scatter(best["best_eval_tokens"] / 1e6, best["best_eval_loss"], s=95, facecolors="white", edgecolors=color, linewidths=2.2, zorder=5)

    axes[0, 0].set(title="Validation loss (лучшие точки обведены)", xlabel="Обработано токенов, млн", ylabel="Cross-entropy")
    axes[0, 1].set(title="Validation perplexity", xlabel="Обработано токенов, млн", ylabel="Perplexity")
    axes[1, 0].set(title="Доля итогового снижения loss", xlabel="Обработано токенов, млн", ylabel="Доля, %")
    axes[1, 0].axhline(100, color="#98A2B3", linestyle="--", linewidth=1)
    axes[1, 1].set(title="Время одного eval на 64 блоках", xlabel="Обработано токенов, млн", ylabel="Секунд")
    for ax in axes.flat:
        ax.legend()
        style_axis(ax)
    save(fig, "02_validation_convergence.png")


def ok_rows(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in benchmark["rows"] if row["status"] == "ok"]


def oom_batch(benchmark: dict[str, Any]) -> int | None:
    failed = [row for row in benchmark["rows"] if row["status"] != "ok"]
    return failed[0]["batch_size"] if failed else None


def plot_batch_throughput(benchmarks: dict[str, dict[str, Any]]) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.65, 1]}, constrained_layout=True)
    fig.suptitle("Batch sweep: производительность трёх GPU", fontsize=17, fontweight="bold")
    for label, benchmark in benchmarks.items():
        rows = ok_rows(benchmark)
        batches = np.asarray([row["batch_size"] for row in rows])
        tps = np.asarray([row["tokens_per_second"] for row in rows])
        low, high = [], []
        for row in rows:
            step_tps = row["tokens_per_step"] / np.asarray(row["step_seconds"])
            low.append(row["tokens_per_second"] - np.min(step_tps))
            high.append(np.max(step_tps) - row["tokens_per_second"])
        ax.errorbar(batches, tps, yerr=np.asarray([low, high]), marker="o", markersize=6, capsize=3, linewidth=2, color=COLORS[label], label=label)
        oom = oom_batch(benchmark)
        if oom is not None:
            ax.scatter([oom], [0], marker="X", s=90, color=COLORS[label], clip_on=False)
            ax.annotate(f"OOM b={oom}", (oom, 0), xytext=(0, 13), textcoords="offset points", ha="center", color=COLORS[label], fontsize=9)

    ax.set(title="Throughput и разброс 5 измеренных шагов", xlabel="Micro-batch", ylabel="Токенов/с")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend()
    style_axis(ax)

    categories = ["Batch 4\n(общая точка)", "Лучший batch\nкаждой GPU"]
    width = 0.24
    x = np.arange(len(categories))
    for idx, (label, benchmark) in enumerate(benchmarks.items()):
        rows = ok_rows(benchmark)
        b4 = next(row["tokens_per_second"] for row in rows if row["batch_size"] == 4)
        best = max(row["tokens_per_second"] for row in rows)
        bars = ax2.bar(x + (idx - 1) * width, [b4, best], width, color=COLORS[label], label=label)
        ax2.bar_label(bars, labels=[f"{b4/1000:.1f}k", f"{best/1000:.1f}k"], padding=3, fontsize=8)
    ax2.set(title="Сопоставимая и максимальная скорость", ylabel="Токенов/с", xticks=x, xticklabels=categories)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    ax2.legend(loc="upper left")
    style_axis(ax2)
    save(fig, "03_batch_throughput_comparison.png")


def plot_memory(benchmarks: dict[str, dict[str, Any]]) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    fig.suptitle("VRAM масштабируется почти линейно с micro-batch", fontsize=17, fontweight="bold")
    for label, benchmark in benchmarks.items():
        rows = ok_rows(benchmark)
        batches = np.asarray([row["batch_size"] for row in rows], dtype=float)
        memory = np.asarray([row["incremental_device_peak_mib"] for row in rows], dtype=float)
        fit = benchmark["linear_fit"]
        max_x = max(batches) + 0.3
        grid = np.linspace(0.8, max_x, 100)
        ax.scatter(batches, memory / 1024, s=50, color=COLORS[label], label=f"{label}: измерения")
        ax.plot(grid, (fit["intercept_mib"] + fit["slope_mib_per_sample"] * grid) / 1024, color=COLORS[label], linestyle="--", linewidth=1.6, label=f"{label}: {fit['slope_mib_per_sample']/1024:.2f} GiB/sample")

        headroom = np.asarray([row["total_vram_mib"] - row["device_peak_used_mib"] for row in rows]) / 1024
        ax2.plot(batches, headroom, marker="o", linewidth=2, color=COLORS[label], label=label)
        oom = oom_batch(benchmark)
        if oom is not None:
            ax2.scatter([oom], [0], marker="X", s=85, color=COLORS[label])

    ax.set(title="Incremental device peak и линейные аппроксимации", xlabel="Micro-batch", ylabel="Дополнительная VRAM, GiB")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, ncol=2)
    style_axis(ax)
    ax2.axhspan(0, 0.5, color="#FEF0C7", alpha=0.8, label="< 0.5 GiB: пограничная зона")
    ax2.set(title="Физический запас VRAM после успешного шага", xlabel="Micro-batch", ylabel="Свободно, GiB")
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(fontsize=8)
    style_axis(ax2)
    save(fig, "04_memory_scaling_comparison.png")


def plot_training_comparison(runs: dict[str, dict[str, Any]]) -> None:
    s5070 = series(runs["RTX 5070 Ti"])
    s5090 = series(runs["RTX 5090"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Одинаковый 10M-token рецепт: RTX 5070 Ti против RTX 5090", fontsize=17, fontweight="bold")

    for label, s in (("RTX 5070 Ti", s5070), ("RTX 5090", s5090)):
        axes[0, 0].plot(s["tokens_m"], rolling(s["loss"], 50), color=COLORS[label], linewidth=2, label=label)
        axes[1, 0].plot(s["tokens_m"], rolling(s["update_tps"], 50), color=COLORS[label], linewidth=2, label=label)
        axes[1, 1].plot(s["tokens_m"], s["cumulative_tps"], color=COLORS[label], linewidth=2, label=label)

    loss_delta = rolling(s5090["loss"], 50) - rolling(s5070["loss"], 50)
    axes[0, 1].plot(s5090["tokens_m"], loss_delta, color="#7F56D9", linewidth=1.8)
    axes[0, 1].axhline(0, color="#98A2B3", linestyle="--", linewidth=1)

    axes[0, 0].set(title="Training loss, скользящее среднее 50", xlabel="Токенов, млн", ylabel="Cross-entropy")
    axes[0, 1].set(title="Разность training loss: 5090 − 5070 Ti", xlabel="Токенов, млн", ylabel="Δ loss")
    axes[1, 0].set(title="Локальный throughput updates, среднее 50", xlabel="Токенов, млн", ylabel="Токенов/с")
    axes[1, 1].set(title="Накопительный throughput с начала train-session", xlabel="Токенов, млн", ylabel="Токенов/с")
    for ax in axes.flat:
        style_axis(ax)
    axes[0, 0].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    for ax in axes[1, :]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    save(fig, "05_training_5070_vs_5090.png")


def plot_runtime(metrics: dict[str, dict[str, Any]]) -> None:
    labels = ["RTX 5070 Ti", "RTX 5090"]
    colors = [COLORS[label] for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), constrained_layout=True)
    fig.suptitle("Время, throughput и рабочая память двух полных прогонов", fontsize=17, fontweight="bold")

    def bars(ax: plt.Axes, values: list[float], title: str, ylabel: str, fmt: str) -> None:
        rects = ax.bar(labels, values, color=colors, width=0.58)
        ax.bar_label(rects, labels=[fmt.format(value) for value in values], padding=4, fontweight="bold")
        ax.set(title=title, ylabel=ylabel)
        ax.set_ylim(0, max(values) * 1.22)
        style_axis(ax)

    bars(axes[0, 0], [metrics[label]["training_summary_seconds"] / 60 for label in labels], "Train-session, включая финальные eval/save", "Минут", "{:.2f}")
    bars(axes[0, 1], [metrics[label]["training_tps_including_final_eval_save"] for label in labels], "Эффективный throughput train-session", "Токенов/с", "{:.0f}")
    bars(axes[1, 0], [metrics[label]["pipeline_seconds"] / 60 for label in labels], "Полный pipeline wall-clock (нестрогое сравнение)", "Минут", "{:.2f}")

    x = np.arange(len(labels))
    width = 0.34
    allocated = [metrics[label]["train_peak_allocated_mib"] / 1024 for label in labels]
    reserved = [metrics[label]["train_peak_reserved_mib"] / 1024 for label in labels]
    r1 = axes[1, 1].bar(x - width / 2, allocated, width, color="#84CAFF", label="PyTorch allocated")
    r2 = axes[1, 1].bar(x + width / 2, reserved, width, color="#175CD3", label="PyTorch reserved")
    axes[1, 1].bar_label(r1, labels=[f"{v:.1f}" for v in allocated], padding=3, fontsize=9)
    axes[1, 1].bar_label(r2, labels=[f"{v:.1f}" for v in reserved], padding=3, fontsize=9)
    axes[1, 1].set(title="Пиковая память в полном обучении", ylabel="GiB", xticks=x, xticklabels=labels)
    axes[1, 1].legend()
    style_axis(axes[1, 1])
    save(fig, "06_runtime_and_efficiency.png")


def plot_frontier(benchmarks: dict[str, dict[str, Any]], selected: dict[str, int]) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    fig.suptitle("Производительность на единицу VRAM и выбор рабочего batch", fontsize=17, fontweight="bold")
    for label, benchmark in benchmarks.items():
        rows = ok_rows(benchmark)
        memory = np.asarray([row["incremental_device_peak_mib"] for row in rows]) / 1024
        tps = np.asarray([row["tokens_per_second"] for row in rows])
        batches = np.asarray([row["batch_size"] for row in rows])
        ax.plot(memory, tps, marker="o", linewidth=2, color=COLORS[label], label=label)
        for mem, throughput, batch in zip(memory, tps, batches):
            ax.annotate(f"b{batch}", (mem, throughput), xytext=(4, 4), textcoords="offset points", fontsize=8, color=COLORS[label])
        efficiency = tps / memory
        ax2.plot(batches, efficiency, marker="o", linewidth=2, color=COLORS[label], label=label)
        if label in selected:
            chosen = next(row for row in rows if row["batch_size"] == selected[label])
            ax.scatter(chosen["incremental_device_peak_mib"] / 1024, chosen["tokens_per_second"], s=180, facecolors="none", edgecolors=COLORS[label], linewidths=2.5)

    ax.set(title="Throughput–VRAM frontier (обведены training batches)", xlabel="Incremental device peak, GiB", ylabel="Токенов/с")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    ax.legend()
    style_axis(ax)
    ax2.set(title="Throughput на 1 GiB incremental VRAM", xlabel="Micro-batch", ylabel="Токенов/с/GiB")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()
    style_axis(ax2)
    save(fig, "07_capacity_efficiency_frontier.png")


def plot_stability(runs: dict[str, dict[str, Any]], benchmarks: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Устойчивость обучения и коротких benchmark-замеров", fontsize=17, fontweight="bold")
    labels = ["RTX 5070 Ti", "RTX 5090"]
    x = np.arange(len(labels))
    width = 0.34

    first = [np.mean(array(runs[label]["train"], "loss")[:100]) for label in labels]
    last = [np.mean(array(runs[label]["train"], "loss")[-100:]) for label in labels]
    b1 = axes[0, 0].bar(x - width / 2, first, width, color="#98A2B3", label="Первые 100")
    b2 = axes[0, 0].bar(x + width / 2, last, width, color=COLORS["good"], label="Последние 100")
    axes[0, 0].bar_label(b1, labels=[f"{v:.3f}" for v in first], padding=3)
    axes[0, 0].bar_label(b2, labels=[f"{v:.3f}" for v in last], padding=3)
    axes[0, 0].set(title="Средний training loss в начале и конце", ylabel="Cross-entropy", xticks=x, xticklabels=labels)
    axes[0, 0].legend()

    grad_data = [array(runs[label]["train"], "grad_norm") for label in labels]
    box = axes[0, 1].boxplot(grad_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], [COLORS[label] for label in labels]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0, 1].axhline(1.0, color=COLORS["warning"], linestyle="--", label="Порог clipping")
    axes[0, 1].set(title="Распределение grad norm (без выбросов на boxplot)", ylabel="L2 norm")
    axes[0, 1].legend()

    for label in labels:
        evaluations = [row for row in runs[label]["eval"] if row["update_step"] >= 700]
        axes[1, 0].plot(array(evaluations, "update_step"), array(evaluations, "validation_loss"), marker="o", linewidth=2, color=COLORS[label], label=label)
    axes[1, 0].set(title="Плато validation loss после update 700", xlabel="Optimizer update", ylabel="Validation loss")
    axes[1, 0].legend()

    for label, benchmark in benchmarks.items():
        rows = ok_rows(benchmark)
        batches = [row["batch_size"] for row in rows]
        cv = [np.std(row["step_seconds"], ddof=1) / np.mean(row["step_seconds"]) * 100 for row in rows]
        axes[1, 1].plot(batches, cv, marker="o", linewidth=2, color=COLORS[label], label=label)
    axes[1, 1].set(title="Разброс времени шага в batch sweep", xlabel="Micro-batch", ylabel="Coefficient of variation, %")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].legend()

    for ax in axes.flat:
        style_axis(ax)
    save(fig, "08_stability_and_distributions.png")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    run5090 = load_5090()
    run5070 = load_5070()
    run3090 = load_3090()
    runs = {"RTX 5070 Ti": run5070, "RTX 5090": run5090}
    benchmarks = {
        "RTX 3090 Ti": run3090["benchmark"],
        "RTX 5070 Ti": run5070["benchmark"],
        "RTX 5090": run5090["benchmark"],
    }

    training = {label: training_metrics(run) for label, run in runs.items()}
    batch = {label: benchmark_metrics(benchmark) for label, benchmark in benchmarks.items()}
    b4 = {
        label: next(row for row in ok_rows(benchmark) if row["batch_size"] == 4)["tokens_per_second"]
        for label, benchmark in benchmarks.items()
    }
    comparisons = {
        "training_5090_vs_5070_speedup": training["RTX 5070 Ti"]["training_summary_seconds"] / training["RTX 5090"]["training_summary_seconds"],
        "training_5090_vs_5070_runtime_reduction_pct": (1 - training["RTX 5090"]["training_summary_seconds"] / training["RTX 5070 Ti"]["training_summary_seconds"]) * 100,
        "training_effective_tps_5090_vs_5070": training["RTX 5090"]["training_tps_including_final_eval_save"] / training["RTX 5070 Ti"]["training_tps_including_final_eval_save"],
        "periodic_eval_5090_vs_5070_speedup": training["RTX 5070 Ti"]["periodic_eval_seconds_median"] / training["RTX 5090"]["periodic_eval_seconds_median"],
        "pipeline_5090_vs_5070_change_pct": (run5090["pipeline_seconds"] / run5070["pipeline_seconds"] - 1) * 100,
        "batch4_speedups": {
            "5070_vs_3090": b4["RTX 5070 Ti"] / b4["RTX 3090 Ti"],
            "5090_vs_5070": b4["RTX 5090"] / b4["RTX 5070 Ti"],
            "5090_vs_3090": b4["RTX 5090"] / b4["RTX 3090 Ti"],
        },
        "best_raw_speedups": {
            "5070_vs_3090": batch["RTX 5070 Ti"]["best_throughput_tps"] / batch["RTX 3090 Ti"]["best_throughput_tps"],
            "5090_vs_5070": batch["RTX 5090"]["best_throughput_tps"] / batch["RTX 5070 Ti"]["best_throughput_tps"],
            "5090_vs_3090": batch["RTX 5090"]["best_throughput_tps"] / batch["RTX 3090 Ti"]["best_throughput_tps"],
        },
        "final_eval_loss_5090_minus_5070": training["RTX 5090"]["final_loss"] - training["RTX 5070 Ti"]["final_loss"],
        "best_eval_loss_5090_minus_5070": training["RTX 5090"]["best_eval_loss"] - training["RTX 5070 Ti"]["best_eval_loss"],
        "memory_slope_spread_pct_of_min": (max(item["fit_slope_mib_per_sample"] for item in batch.values()) / min(item["fit_slope_mib_per_sample"] for item in batch.values()) - 1) * 100,
    }

    derived = {
        "report_generated_from_raw_metrics": True,
        "sources": {
            "rtx_5090_metrics": run5090["metrics_source"],
            "rtx_5090_full": run5090["full_source"],
            "rtx_5070": run5070["source"],
            "rtx_3090": run3090["source"],
        },
        "experiment_identity": {
            "model": run5090["benchmark"]["model"],
            "model_revision": run5090["benchmark"]["model_revision"],
            "dataset": run5090["benchmark"]["dataset"],
            "dataset_revision": run5090["benchmark"]["config"]["data"]["revision"],
            "slice_sha256_5070_5090": run5090["benchmark"]["data"]["slice_sha256"],
            "sequence_length": run5090["benchmark"]["data"]["sequence_length"],
            "parameters": ok_rows(run5090["benchmark"])[0]["model_parameters"],
        },
        "training": training,
        "benchmark": batch,
        "comparisons": comparisons,
        "rtx_3090_smoke": run3090["smoke"],
    }
    (SCRIPT_DIR / "derived_metrics.json").write_text(
        json.dumps(json_safe(derived), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plot_training_dynamics(run5090, training["RTX 5090"])
    plot_validation(runs, training)
    plot_batch_throughput(benchmarks)
    plot_memory(benchmarks)
    plot_training_comparison(runs)
    plot_runtime(training)
    plot_frontier(benchmarks, {"RTX 3090 Ti": 4, "RTX 5070 Ti": 4, "RTX 5090": 8})
    plot_stability(runs, benchmarks)
    print(f"Wrote {SCRIPT_DIR / 'derived_metrics.json'} and 8 PNG figures")


if __name__ == "__main__":
    main()
