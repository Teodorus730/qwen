"""Rebuild final sweep CSV and plots from the eight verified-adapter runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path



RUNS = ((0.00, "adapter_check_a0p00_seed0", "0p00"),
        (0.05, "saved_adapter_a0p05_seed0", "0p05"),
        (0.10, "saved_adapter_a0p10_seed0", "0p10"),
        (0.20, "saved_adapter_a0p20_seed0", "0p20"),
        (0.25, "saved_adapter_a0p25_seed0", "0p25"),
        (0.30, "saved_adapter_a0p30_seed0", "0p30"),
        (0.35, "saved_adapter_a0p35_seed0", "0p35"),
        (0.50, "saved_adapter_a0p50_seed0", "0p50"))


def phase_records(run_dir: Path) -> dict:
    records = [json.loads(line) for line in (run_dir / "log.jsonl").read_text(
        encoding="utf-8").splitlines()]
    return {record.get("phase"): record for record in records}


def collect(root: Path) -> list[dict]:
    rows = []
    for alpha, run_name, suffix in RUNS:
        run_dir = root / "results" / run_name
        adapter_dir = root / "artifacts" / "lora_adapters" / f"alpha_{suffix}"
        phases = phase_records(run_dir)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metadata = json.loads((adapter_dir / "run_metadata.json").read_text(
            encoding="utf-8"))
        verified = json.loads((adapter_dir / "verify_report.json").read_text(
            encoding="utf-8"))
        teacher, noised, final, complete = (phases["teacher_baseline"],
                                             phases["post_noise"], phases["eval"],
                                             phases["data_complete"])
        if complete["teacher_train_forwards"] != 0:
            raise RuntimeError(f"teacher was used in training: {run_name}")
        if metadata["final_metrics"] != {key: metadata["final_metrics"][key]
                                         for key in metadata["final_metrics"]}:
            raise RuntimeError(f"invalid metadata: {run_name}")
        for key in ("ppl", "kl_teacher_student", "kl_student_teacher", "cka_mean"):
            if not math.isclose(float(final[key]), float(verified["verified_metrics"][key]),
                                rel_tol=1e-3, abs_tol=1e-3):
                raise RuntimeError(f"verification mismatch {run_name}: {key}")
        nll_noise, nll_final, nll_teacher = map(math.log, (noised["ppl"], final["ppl"], teacher["ppl"]))
        recovery = "" if alpha == 0 else 100 * (nll_noise - nll_final) / (nll_noise - nll_teacher)
        rows.append({
            "alpha": alpha, "noise_seed": config["noise_seed"],
            "training_seed": config["training_seed"], "run_name": run_name,
            "adapter_path": f"artifacts/lora_adapters/alpha_{suffix}",
            "teacher_ppl": teacher["ppl"], "post_noise_ppl": noised["ppl"],
            "post_lora_ppl": final["ppl"], "post_noise_nll": nll_noise,
            "post_lora_nll": nll_final, "nll_recovery_pct": recovery,
            "post_noise_kl_teacher_student": noised["kl_teacher_student"],
            "post_lora_kl_teacher_student": final["kl_teacher_student"],
            "post_noise_kl_student_teacher": noised["kl_student_teacher"],
            "post_lora_kl_student_teacher": final["kl_student_teacher"],
            "post_noise_cka": noised["cka_mean"], "post_lora_cka": final["cka_mean"],
            "elapsed_s": final["elapsed"], "peak_memory_mb": final["peak_memory_mb"],
            "optimizer_steps": complete["optimizer_steps_completed"],
            "processed_examples": complete["training_source_rows"],
            "teacher_train_forwards": complete["teacher_train_forwards"],
            "verification_logit_delta": verified["adapter_effect_max_abs_logit_delta"],
        })
    return rows


def write_csv(root: Path, rows: list[dict]) -> None:
    with (root / "SYNTHETIC_CE_SWEEP.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(root: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    alphas = [row["alpha"] for row in rows]
    series = (("PPL", "Perplexity", "post_noise_ppl", "post_lora_ppl", True, "PPL"),
              ("KL", "KL divergence", "post_noise_kl_teacher_student", "post_lora_kl_teacher_student", False, "KL"),
              ("CKA", "Mean layer CKA", "post_noise_cka", "post_lora_cka", False, "CKA"))
    for title, ylabel, noise_key, final_key, logy, suffix in series:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(alphas, [row[noise_key] for row in rows], marker="o", label="post-noise")
        ax.plot(alphas, [row[final_key] for row in rows], marker="o", label="post-LoRA")
        if title == "KL":
            ax.plot(alphas, [row["post_lora_kl_student_teacher"] for row in rows],
                    marker="o", linestyle="--", label="post-LoRA KL S→T")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("noise alpha")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Synthetic CE LoRA recovery: {title}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / f"SYNTHETIC_CE_SWEEP_{suffix}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = collect(root)
    write_csv(root, rows)
    write_plots(root, rows)
    print(f"wrote final sweep summary for {len(rows)} adapters")


if __name__ == "__main__":
    main()
