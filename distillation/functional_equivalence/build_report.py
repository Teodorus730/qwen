"""Build the Russian Markdown report and the five requested key figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RAW_PATH = OUTPUT_DIR / "raw_results.json"
PAIR_PATH = OUTPUT_DIR / "soft_vs_hard.json"
AUDIT_PATH = OUTPUT_DIR / "representation_hypothesis_audit.json"
PROBE_PATH = OUTPUT_DIR / "linear_probe_results.json"
ATTENTION_PATH = OUTPUT_DIR / "attention_alignment_results.json"
REPORT_PATH = SCRIPT_DIR / "REPORT.md"

COLORS = {
    "soft_kd": "#2563EB",
    "hard_teacher_ce": "#DC2626",
    "teacher": "#111827",
}
LABELS = {
    "soft_kd": "Soft KD",
    "hard_teacher_ce": "Hard teacher CE",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(model: dict[str, Any], key: str) -> float:
    return float(model["baseline"][key]["estimate"])


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.23,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


def ordered_models(raw: dict[str, Any], objective: str) -> list[dict[str, Any]]:
    return sorted(
        (
            model for model in raw["models"].values()
            if model["objective"] == objective
        ),
        key=lambda model: model["alpha"],
    )


def line_with_ci(
    axis,
    models: list[dict[str, Any]],
    key: str,
    objective: str,
    scale: float = 1.0,
) -> None:
    x = np.asarray([model["alpha"] for model in models])
    y = np.asarray([metric(model, key) * scale for model in models])
    lo = np.asarray([
        model["baseline"][key]["ci95"][0] * scale for model in models
    ])
    hi = np.asarray([
        model["baseline"][key]["ci95"][1] * scale for model in models
    ])
    axis.plot(
        x, y, marker="o", linewidth=2, color=COLORS[objective],
        label=LABELS[objective],
    )
    axis.fill_between(x, lo, hi, color=COLORS[objective], alpha=0.13)


def save_output_fidelity(
    raw: dict[str, Any], pairs: dict[str, Any]
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for objective in ("soft_kd", "hard_teacher_ce"):
        models = ordered_models(raw, objective)
        line_with_ci(axes[0, 0], models, "top1_match", objective, 100)
        line_with_ci(
            axes[0, 1], models, "kl_teacher_student_t1", objective
        )
        axes[0, 2].plot(
            [model["alpha"] for model in models],
            [model["baseline"]["student_ppl"] for model in models],
            marker="o", linewidth=2, color=COLORS[objective],
            label=LABELS[objective],
        )
        line_with_ci(axes[1, 0], models, "topk_overlap", objective, 100)

    axes[0, 0].axhline(95, color="#6B7280", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Совпадение top-1 с Teacher", ylabel="% токенов")
    axes[0, 1].axhline(0.1, color="#6B7280", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="KL(Teacher || Student), T=1", ylabel="нат/токен")
    teacher_ppl = next(iter(raw["models"].values()))["baseline"]["teacher_ppl"]
    axes[0, 2].axhline(
        teacher_ppl, color=COLORS["teacher"], linestyle="--",
        label=f"Teacher ({teacher_ppl:.2f})",
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set(title="Perplexity на реальном next token", ylabel="PPL")
    axes[1, 0].set(title="Пересечение top-10", ylabel="% токенов")

    pair_rows = sorted(
        pairs["pairs"].values(), key=lambda item: item["alpha"]
    )
    pair_alpha = [item["alpha"] for item in pair_rows]
    axes[1, 1].plot(
        pair_alpha,
        [item["summary"]["top1_match"]["estimate"] * 100 for item in pair_rows],
        marker="o", color="#7C3AED", linewidth=2,
    )
    axes[1, 1].set(
        title="Soft vs Hard напрямую: top-1", ylabel="% совпадений"
    )
    axes[1, 2].plot(
        pair_alpha,
        [
            item["summary"]["kl_teacher_student_t1"]["estimate"]
            for item in pair_rows
        ],
        marker="o", color="#7C3AED", linewidth=2,
    )
    axes[1, 2].set(
        title="Soft vs Hard напрямую: KL(Soft || Hard)",
        ylabel="нат/токен",
    )
    for axis in axes.flat:
        axis.set_xlabel("α исходного шума")
    axes[0, 0].legend()
    axes[0, 2].legend()
    figure.suptitle(
        "Выходная эквивалентность: точечные оценки и 95% paired bootstrap CI",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "01_output_fidelity.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def save_error_analysis(raw: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for objective in ("soft_kd", "hard_teacher_ce"):
        models = ordered_models(raw, objective)
        x = [model["alpha"] for model in models]
        axes[0].plot(
            x,
            [
                model["baseline"]["error_jaccard"]["estimate"] * 100
                for model in models
            ],
            marker="o", linewidth=2, color=COLORS[objective],
            label=f"{LABELS[objective]}: Jaccard позиций",
        )
        axes[0].plot(
            x,
            [
                model["baseline"][
                    "same_wrong_prediction_given_teacher_error"
                ]["estimate"] * 100
                for model in models
            ],
            marker="s", linewidth=1.5, linestyle="--",
            color=COLORS[objective],
            label=f"{LABELS[objective]}: та же ошибка",
        )
    axes[0].set(
        title="Совпадение ошибок",
        xlabel="α исходного шума",
        ylabel="%",
    )
    axes[0].legend(fontsize=8)

    for axis, model_id, title in (
        (axes[1], "soft_a0.5", "Soft KD, α=0.5"),
        (axes[2], "hard_a0.5", "Hard CE, α=0.5"),
    ):
        scatter = raw["models"][model_id]["error_confidence_scatter"]
        x = np.asarray(scatter["teacher_confidence_on_teacher_errors"])
        y = np.asarray(scatter["student_confidence_on_teacher_errors"])
        axis.scatter(x, y, s=9, alpha=0.27, color="#4F46E5")
        limit = max(float(x.max(initial=0.1)), float(y.max(initial=0.1)))
        axis.plot([0, limit], [0, limit], color="#111827", linestyle="--")
        axis.set(
            title=title,
            xlabel="Confidence Teacher",
            ylabel="Confidence Student",
            xlim=(0, limit),
            ylim=(0, limit),
        )
    figure.suptitle(
        "Ошибки и уверенность на позициях, где Teacher ошибся", fontsize=15
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "02_error_analysis.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def robustness_values(
    model: dict[str, Any], family: str
) -> list[dict[str, Any]]:
    return sorted(
        (
            item for item in model["robustness"].values()
            if item["family"] == family
        ),
        key=lambda item: item["level"],
    )


def save_embedding_robustness(raw: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    palette = plt.cm.viridis(np.linspace(0.12, 0.9, 5))
    for column, objective in enumerate(("soft_kd", "hard_teacher_ce")):
        models = ordered_models(raw, objective)
        teacher_drawn = False
        for color, model in zip(palette, models):
            items = robustness_values(model, "embedding_noise_sigma")
            x = [item["level"] for item in items]
            if not teacher_drawn:
                axes[0, column].plot(
                    x,
                    [
                        item["teacher_top1_stability"]["estimate"] * 100
                        for item in items
                    ],
                    color=COLORS["teacher"],
                    linestyle="--",
                    linewidth=2,
                    label="Teacher",
                )
                teacher_drawn = True
            axes[0, column].plot(
                x,
                [
                    item["student_top1_stability"]["estimate"] * 100
                    for item in items
                ],
                marker="o",
                color=color,
                label=f"Student α={model['alpha']:g}",
            )
            axes[1, column].plot(
                x,
                [
                    item["stability_gap_student_minus_teacher"]["estimate"]
                    * 100
                    for item in items
                ],
                marker="o",
                color=color,
                label=f"α={model['alpha']:g}",
            )
        axes[0, column].set(
            title=f"{LABELS[objective]}: top-1 stability",
            ylabel="% совпадений с чистым выходом",
        )
        axes[1, column].axhspan(-2, 2, color="#10B981", alpha=0.12)
        axes[1, column].axhline(0, color="#111827", linewidth=1)
        axes[1, column].set(
            title=f"{LABELS[objective]}: Student − Teacher",
            xlabel="σ абсолютного шума embedding",
            ylabel="разница, п.п.",
        )
        axes[0, column].legend(fontsize=7, ncol=2)
    figure.suptitle(
        "Устойчивость к Gaussian noise в embeddings (исходная гипотеза Qwen)",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "03_embedding_robustness.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_token_robustness(raw: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    palette = plt.cm.plasma(np.linspace(0.12, 0.9, 5))
    families = (
        ("token_dropout_rate", "Zero-embedding token dropout"),
        ("token_replacement_rate", "Внешняя замена токенов"),
    )
    for row, (family, family_title) in enumerate(families):
        for column, objective in enumerate(("soft_kd", "hard_teacher_ce")):
            axis = axes[row, column]
            axis.axhspan(-2, 2, color="#10B981", alpha=0.12)
            axis.axhline(0, color="#111827", linewidth=1)
            for color, model in zip(
                palette, ordered_models(raw, objective)
            ):
                items = robustness_values(model, family)
                axis.plot(
                    [item["level"] * 100 for item in items],
                    [
                        item[
                            "stability_gap_student_minus_teacher"
                        ]["estimate"] * 100
                        for item in items
                    ],
                    marker="o",
                    color=color,
                    label=f"α={model['alpha']:g}",
                )
            axis.set(
                title=f"{family_title}: {LABELS[objective]}",
                xlabel="доля повреждённых токенов, %",
                ylabel="Student − Teacher stability, п.п.",
            )
            axis.legend(fontsize=7, ncol=2)
    figure.suptitle(
        "Парные кривые устойчивости; зелёная полоса — equivalence band ±2 п.п.",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "04_token_robustness.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_generation(raw: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for objective in ("soft_kd", "hard_teacher_ce"):
        models = ordered_models(raw, objective)
        x = [model["alpha"] for model in models]
        style = {
            "marker": "o",
            "linewidth": 2,
            "color": COLORS[objective],
            "label": LABELS[objective],
        }
        axes[0, 0].plot(
            x,
            [model["generation"]["token_match"]["estimate"] * 100
             for model in models],
            **style,
        )
        axes[0, 1].plot(
            x,
            [model["generation"]["common_prefix_tokens"]["estimate"]
             for model in models],
            **style,
        )
        axes[1, 0].plot(
            x,
            [model["generation"]["teacher_path_kl"]["estimate"]
             for model in models],
            **style,
        )
        axes[1, 1].plot(
            x,
            [model["generation"]["student_path_kl"]["estimate"]
             for model in models],
            **style,
        )
    axes[0, 0].set(title="Совпадение токенов greedy continuation", ylabel="%")
    axes[0, 1].set(
        title="Общий префикс до первого расхождения", ylabel="токенов из 32"
    )
    axes[1, 0].set(
        title="KL на траектории Teacher", ylabel="нат/токен"
    )
    axes[1, 1].set(
        title="KL на траектории Student", ylabel="нат/токен"
    )
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xlabel("α исходного шума")
    axes[0, 0].legend()
    figure.suptitle(
        "Свободная greedy-генерация: совпадение последовательностей и on-policy KL",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "05_free_generation.png", dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def baseline_table(raw: dict[str, Any]) -> str:
    header = (
        "| Модель | Top-1 match, 95% CI | KL(T‖S) | PPL Student | "
        "Top-10 overlap | Та же ошибка при ошибке T | Greedy token match |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for objective in ("soft_kd", "hard_teacher_ce"):
        for model in ordered_models(raw, objective):
            baseline = model["baseline"]
            top1 = baseline["top1_match"]
            rows.append(
                "| {label}, α={alpha:g} | {estimate} "
                "[{lo}; {hi}] | {kl:.4f} | {ppl:.2f} | {top10} | "
                "{same_wrong} | {generation} |".format(
                    label=LABELS[objective],
                    alpha=model["alpha"],
                    estimate=pct(top1["estimate"], 2),
                    lo=pct(top1["ci95"][0], 2),
                    hi=pct(top1["ci95"][1], 2),
                    kl=baseline["kl_teacher_student_t1"]["estimate"],
                    ppl=baseline["student_ppl"],
                    top10=pct(baseline["topk_overlap"]["estimate"], 2),
                    same_wrong=pct(
                        baseline[
                            "same_wrong_prediction_given_teacher_error"
                        ]["estimate"],
                        2,
                    ),
                    generation=pct(
                        model["generation"]["token_match"]["estimate"], 2
                    ),
                )
            )
    return "\n".join(rows)


def robustness_table(raw: dict[str, Any]) -> str:
    lines = [
        "| Модель | Условий внутри ±2 п.п. (95% CI) | "
        "Макс. абсолютный gap | Gap при σ=0.01 | Gap при dropout 10% |",
        "|---|---:|---:|---:|---:|",
    ]
    for objective in ("soft_kd", "hard_teacher_ce"):
        for model in ordered_models(raw, objective):
            items = model["robustness"]
            passed = sum(
                item["equivalent_within_top1_band"]
                for item in items.values()
            )
            max_gap = max(
                abs(item[
                    "stability_gap_student_minus_teacher"
                ]["estimate"])
                for item in items.values()
            )
            noise = items["embedding_noise_sigma_0.01"][
                "stability_gap_student_minus_teacher"
            ]["estimate"]
            dropout = items["token_dropout_rate_0.1"][
                "stability_gap_student_minus_teacher"
            ]["estimate"]
            lines.append(
                f"| {LABELS[objective]}, α={model['alpha']:g} | "
                f"{passed}/10 | {max_gap * 100:.2f} п.п. | "
                f"{noise * 100:+.2f} п.п. | {dropout * 100:+.2f} п.п. |"
            )
    return "\n".join(lines)


def pair_table(pairs: dict[str, Any]) -> str:
    lines = [
        "| α | Top-1 Soft vs Hard | KL(Soft‖Hard) | "
        "Для сравнения: KL(T‖Soft) / KL(T‖Hard) |",
        "|---:|---:|---:|---:|",
    ]
    raw = load_json(RAW_PATH)
    for item in sorted(
        pairs["pairs"].values(), key=lambda row: row["alpha"]
    ):
        alpha = item["alpha"]
        soft = raw["models"][f"soft_a{alpha:g}"]
        hard = raw["models"][f"hard_a{alpha:g}"]
        lines.append(
            f"| {alpha:g} | "
            f"{pct(item['summary']['top1_match']['estimate'], 2)} | "
            f"{item['summary']['kl_teacher_student_t1']['estimate']:.4f} | "
            f"{metric(soft, 'kl_teacher_student_t1'):.4f} / "
            f"{metric(hard, 'kl_teacher_student_t1'):.4f} |"
        )
    return "\n".join(lines)


def audit_table(audit: dict[str, Any]) -> str:
    lines = [
        "| Модель | Слой | CKA | Effective-rank S/T | "
        "Spectrum cosine | Held-out Procrustes R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, model in audit["models"].items():
        for layer, values in model["layers"].items():
            lines.append(
                f"| {model_id} | {layer} | {values['linear_cka']:.4f} | "
                f"{values['effective_rank_ratio_student_teacher']:.3f} | "
                f"{values['singular_spectrum_cosine']:.4f} | "
                f"{values['heldout_scaled_orthogonal_procrustes_r2']:.3f} |"
            )
    return "\n".join(lines)


def linear_probe_table(probe: dict[str, Any]) -> str:
    lines = [
        "| Модель | Teacher probe на raw Student | Отдельный Student probe | "
        "Teacher probe после Procrustes | Средний / max raw drop | ≤5 п.п. на всех слоях |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, model in probe["models"].items():
        layer_rows = list(model["layers"].values())
        raw = np.mean([
            row["qwen_frozen_teacher_probe_on_raw_student"]["accuracy"]
            for row in layer_rows
        ])
        own = np.mean([
            row["separate_student_probe"]["accuracy"] for row in layer_rows
        ])
        aligned = np.mean([
            row["teacher_probe_on_procrustes_aligned_student"]["accuracy"]
            for row in layer_rows
        ])
        drop = np.mean([row["accuracy_drop_qwen_raw"] for row in layer_rows])
        max_drop = max(row["accuracy_drop_qwen_raw"] for row in layer_rows)
        lines.append(
            f"| {model_id} | {raw * 100:.2f}% | {own * 100:.2f}% | "
            f"{aligned * 100:.2f}% | {drop * 100:+.2f} / {max_drop * 100:+.2f} п.п. | "
            f"{'да' if max_drop <= 0.05 else 'нет'} |"
        )
    return "\n".join(lines)


def attention_table(attention: dict[str, Any]) -> str:
    lines = [
        "| Модель | Same-index mean | Same-index min | Hungarian mean | "
        "Голов < 0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id, model in attention["models"].items():
        summary = model["summary"]
        lines.append(
            f"| {model_id} | {summary['same_index_mean']:.4f} | "
            f"{summary['same_index_min']:.4f} | {summary['matched_mean']:.4f} | "
            f"{summary['same_index_low_count']}/{summary['total_heads']} |"
        )
    return "\n".join(lines)


def save_linear_probing(probe: dict[str, Any]) -> None:
    layers = [str(layer) for layer in probe["definition"]["layers"]]
    figure, axes = plt.subplots(2, len(layers), figsize=(15, 8), sharey=True)
    styles = (
        ("qwen_frozen_teacher_probe_on_raw_student", "Raw cross-probe", "o", "-"),
        ("separate_student_probe", "Separate probe", "s", "--"),
        ("teacher_probe_on_procrustes_aligned_student", "Aligned cross-probe", "^", ":"),
    )
    for row_index, objective in enumerate(("soft_kd", "hard_teacher_ce")):
        models = sorted(
            (model for model in probe["models"].values() if model["objective"] == objective),
            key=lambda model: model["alpha"],
        )
        for column, layer in enumerate(layers):
            axis = axes[row_index, column]
            teacher_accuracy = probe["teacher"]["layers"][layer]["accuracy"] * 100
            axis.axhline(teacher_accuracy, color="#111827", linewidth=1.5,
                         label=f"Teacher ({teacher_accuracy:.1f}%)")
            for field, label, marker, linestyle in styles:
                axis.plot(
                    [model["alpha"] for model in models],
                    [model["layers"][layer][field]["accuracy"] * 100 for model in models],
                    marker=marker, linestyle=linestyle, linewidth=1.8, label=label,
                )
            axis.set(title=f"{LABELS[objective]}, слой {layer}", xlabel="α")
            if column == 0:
                axis.set_ylabel("UPOS accuracy, %")
            if row_index == 0 and column == len(layers) - 1:
                axis.legend(fontsize=8)
    figure.suptitle("Linear probing: переносимость и информационная ёмкость", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "06_linear_probing.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_attention_alignment(attention: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fields = (
        ("same_index_mean", "Средний same-index cosine", False),
        ("same_index_min", "Минимальный same-index cosine", False),
        ("same_index_low_count", "Число голов с cosine < 0.5", False),
    )
    for objective in ("soft_kd", "hard_teacher_ce"):
        models = sorted(
            (model for model in attention["models"].values() if model["objective"] == objective),
            key=lambda model: model["alpha"],
        )
        x = [model["alpha"] for model in models]
        for axis, (field, title, _) in zip(axes.flat[:3], fields):
            axis.plot(x, [model["summary"][field] for model in models], marker="o",
                      linewidth=2, color=COLORS[objective], label=LABELS[objective])
            axis.set(title=title, xlabel="α")
        axes[1, 1].plot(
            x,
            [(model["summary"]["matched_mean"] - model["summary"]["same_index_mean"])
             for model in models],
            marker="o", linewidth=2, color=COLORS[objective], label=LABELS[objective],
        )
    axes[0, 0].axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
    axes[0, 1].axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
    axes[1, 1].set(title="Выигрыш Hungarian matching", xlabel="α", ylabel="Δ cosine")
    for axis in axes.flat:
        axis.legend()
    figure.suptitle("Attention alignment на 100 одинаковых входах", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "07_attention_alignment.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary_csv(
    raw: dict[str, Any],
    probe: dict[str, Any],
    attention: dict[str, Any],
) -> None:
    rows = []
    for model in raw["models"].values():
        baseline = model["baseline"]
        model_id = model["id"]
        probe_layers = list(probe["models"][model_id]["layers"].values())
        attention_summary = attention["models"][model_id]["summary"]
        rows.append({
            "model_id": model["id"],
            "objective": model["objective"],
            "alpha": model["alpha"],
            "top1_match": baseline["top1_match"]["estimate"],
            "top1_ci_low": baseline["top1_match"]["ci95"][0],
            "top1_ci_high": baseline["top1_match"]["ci95"][1],
            "kl_teacher_student_t1": baseline[
                "kl_teacher_student_t1"
            ]["estimate"],
            "kl_teacher_student_t2": baseline[
                "kl_teacher_student_t2"
            ]["estimate"],
            "kl_student_teacher_t1": baseline[
                "kl_student_teacher_t1"
            ]["estimate"],
            "teacher_ppl": baseline["teacher_ppl"],
            "student_ppl": baseline["student_ppl"],
            "top10_overlap": baseline["topk_overlap"]["estimate"],
            "same_wrong_given_teacher_error": baseline[
                "same_wrong_prediction_given_teacher_error"
            ]["estimate"],
            "greedy_token_match": model["generation"][
                "token_match"
            ]["estimate"],
            "teacher_path_kl": model["generation"][
                "teacher_path_kl"
            ]["estimate"],
            "student_path_kl": model["generation"][
                "student_path_kl"
            ]["estimate"],
            "qwen_project_criteria_pass": model["verdict"][
                "qwen_project_criteria_pass"
            ],
            "linear_probe_raw_cross_accuracy_mean": np.mean([
                layer["qwen_frozen_teacher_probe_on_raw_student"]["accuracy"]
                for layer in probe_layers
            ]),
            "linear_probe_separate_accuracy_mean": np.mean([
                layer["separate_student_probe"]["accuracy"]
                for layer in probe_layers
            ]),
            "linear_probe_aligned_cross_accuracy_mean": np.mean([
                layer["teacher_probe_on_procrustes_aligned_student"]["accuracy"]
                for layer in probe_layers
            ]),
            "attention_same_index_mean": attention_summary["same_index_mean"],
            "attention_same_index_min": attention_summary["same_index_min"],
            "attention_same_index_low_count": attention_summary["same_index_low_count"],
            "attention_hungarian_mean": attention_summary["matched_mean"],
        })
    pd.DataFrame(rows).sort_values(
        ["objective", "alpha"]
    ).to_csv(OUTPUT_DIR / "summary.csv", index=False)


def write_report(
    raw: dict[str, Any],
    pairs: dict[str, Any],
    audit: dict[str, Any],
    probe: dict[str, Any],
    attention: dict[str, Any],
) -> None:
    teacher_ppl = next(iter(raw["models"].values()))["baseline"]["teacher_ppl"]
    synthetic = audit["synthetic_invariance"]
    teacher_probe_mean = np.mean([
        layer["accuracy"] for layer in probe["teacher"]["layers"].values()
    ])
    raw_probe_drops = {
        model_id: float(np.mean([
            layer["accuracy_drop_qwen_raw"] for layer in model["layers"].values()
        ]))
        for model_id, model in probe["models"].items()
    }
    separate_probe_drops = {
        model_id: float(np.mean([
            layer["accuracy_drop_separate_probe"] for layer in model["layers"].values()
        ]))
        for model_id, model in probe["models"].items()
    }
    worst_raw_probe = max(raw_probe_drops, key=raw_probe_drops.get)
    worst_separate_probe = max(separate_probe_drops, key=separate_probe_drops.get)
    attention_means = {
        model_id: model["summary"]["same_index_mean"]
        for model_id, model in attention["models"].items()
    }
    attention_mins = {
        model_id: model["summary"]["same_index_min"]
        for model_id, model in attention["models"].items()
    }
    lowest_attention_model = min(attention_means, key=attention_means.get)
    lowest_attention_head_model = min(attention_mins, key=attention_mins.get)
    attention_top1_corr = attention["correlations"]["metrics"]["same_index_mean"]["top1_match"]
    attention_kl_corr = attention["correlations"]["metrics"]["same_index_mean"]["kl_teacher_student_t1"]
    attention_error_corr = attention["correlations"]["metrics"]["same_index_mean"]["same_wrong_given_teacher_error"]
    attention_robust_corr = attention["correlations"]["metrics"]["same_index_mean"]["mean_abs_robustness_stability_gap"]
    low_top1_corr = attention["correlations"]["metrics"]["same_index_low_count"]["top1_match"]
    low_robust_corr = attention["correlations"]["metrics"]["same_index_low_count"]["mean_abs_robustness_stability_gap"]
    report = f"""# Проверка функциональной эквивалентности Qwen Teacher и Student

Дата итогового прогона: 21 июля 2026 года.

## Итог

**Строгая функциональная эквивалентность не подтверждена ни для одного из десяти
чекпойнтов.** Ни одна модель не проходит одновременно исходные пороги
`Top-1 Match > 95%`, `KL(Teacher||Student) < 0.1` и equivalence band
устойчивости `±2 п.п.`. Лучший top-1 результат — **89.07% у Soft KD α=0.05**;
его KL равен **0.0368**, а PPL **14.84** против **{teacher_ppl:.2f}** у Teacher.
Это близкое качество и близкое распределение, но не идентичное поведение.

Главное различие целей обучения видно особенно чисто на `α=0.05`:

- Soft KD: top-1 89.07%, KL 0.0368, PPL 14.84.
- Hard teacher CE: top-1 86.91%, KL 2.4899, PPL 158.99.

Hard CE действительно копирует много argmax-решений, но не восстанавливает
форму распределения вероятностей и реальную языковую likelihood. Поэтому
высокий top-1 agreement нельзя трактовать как полную функциональную
эквивалентность.

![Выходная эквивалентность](outputs/01_output_fidelity.png)

## Что именно проверено

- Teacher: `Qwen/Qwen3-0.6B-Base`.
- Students: по пять финальных чекпойнтов Soft KD и Hard teacher CE для
  `α = 0.05, 0.1, 0.2, 0.35, 0.5`.
- Данные: не использованный в обучающем старте хвост локального FineWeb-Edu,
  начиная после 45 000 прошедших score-фильтр документов.
- Baseline: 16 блоков × 255 предсказаний = **4 080 next-token примеров**.
- Robustness: 8 блоков × 255 = **2 040 парных примеров** на каждый уровень.
- Free generation: 16 одинаковых префиксов, по 32 greedy-токена.
- Linear probing: 12 000 train и 4 000 test целей UPOS из UD English EWT,
  слои 7/14/21, три варианта переноса probe для каждого Student.
- Attention alignment: 100 одинаковых входов длиной 128 токенов, все 28 слоёв
  и 16 attention-голов, то есть 448 пар голов на каждый Student.
- Интервалы: paired bootstrap по блокам, 2 000 повторов. Токены внутри одного
  блока намеренно не объявлялись независимыми.
- Все модели заморожены; tokenizer, входы, порядок и случайные возмущения
  совпадают.

Вывод относится к этому held-out распределению и набору возмущений. Конечный
набор тестов не может математически доказать равенство функций на всех входах.

## Перепроверка плана Qwen

1. **Сравнение выходных распределений — оставлено и усилено.** Помимо top-1,
   KL при `T=1/2` и Pearson посчитаны reverse-KL, Jensen-Shannon, total
   variation и top-10 overlap. Pearson оставлен ради исходной гипотезы, но он
   сильно зависит от длинного хвоста логитов и не является главным критерием.
2. **Error overlap — оставлен, но Jaccard дополнен точным совпадением неверного
   токена.** Для base LM большинство exact next-token позиций сложные, поэтому
   простой Jaccard множеств ошибок искусственно высокий. Метрика «тот же
   неправильный top-1 при ошибке Teacher» информативнее.
3. **Robustness — оставлен, статистический критерий исправлен.** «Не нашли
   различий» не означает эквивалентность. Здесь эквивалентность принимается
   только если весь 95% CI парной разницы лежит внутри `[-2; +2]` п.п.
4. **Embedding noise и token dropout Qwen не удалены.** Gaussian noise использует
   исходные абсолютные `σ`; token dropout реализован как обнуление embedding при
   сохранении позиции. Дополнительно введена внешняя замена токенов из того же
   блока, потому что у Qwen3 нет отдельного mask-token, а внутренняя интервенция
   не равна обычному повреждению текста.
5. **Linear probing выполнен полностью, но не включён в output-level вердикт.** Исходная схема
   «обучить probe на Teacher и без адаптации применить к Student» чувствительна
   к координатам и может провалиться даже при сохранённой информации. Для
   вопроса об информационной ёмкости дополнительно обучены отдельные Student
   probes на том же split, а также проверен перенос после Procrustes alignment.
6. **Attention alignment выполнен полностью, но имеет отдельный механистический
   статус.** Исходное same-index сравнение всех голов сохранено. Дополнительно
   выполнен Hungarian matching, поскольку перестановка голов не должна
   ошибочно считаться потерей механистического сходства.

## Baseline: распределения, реальные токены и ошибки

{baseline_table(raw)}

Порог top-1 95% не проходит никто. По KL точечный порог 0.1 проходят только
Soft KD `α=0.05` и `α=0.1`, но второй находится у самой границы; совместный
критерий всё равно не выполнен.

![Ошибки и confidence](outputs/02_error_analysis.png)

Высокий Jaccard ошибок сам по себе вводит в заблуждение: например, модели часто
обе ошибаются потому, что exact next-token accuracy у base LM невысока. Столбец
«та же ошибка при ошибке T» требует ещё и совпадения неверного argmax и
монотонно падает с ростом α: у Soft KD с 84.66% до 29.00%, у Hard CE с 82.10%
до 32.25%.

## Robustness

{robustness_table(raw)}

![Robustness к embedding noise](outputs/03_embedding_robustness.png)

Абсолютные `σ=0.05/0.1/0.2` слишком велики для содержательной локальной
проверки: уже при `σ=0.05` Teacher сохраняет свой clean top-1 примерно лишь на
6.5% позиций, а при `σ=0.1` — примерно на 1.1%. Это saturation/floor regime,
где близость двух почти разрушенных кривых не доказывает одинаковую
устойчивость. Самый диагностичный уровень — `σ=0.01`: разрыв Student−Teacher
растёт по модулю с α и достигает −11.32 п.п. у Soft `α=0.5` и −9.12 п.п. у
Hard `α=0.5`.

![Robustness к token corruption](outputs/04_token_robustness.png)

Ни одна модель не проходит все десять robustness-условий с 95% CI внутри
`±2 п.п.`. У малых α точечные разницы часто небольшие, но часть интервалов
шире equivalence band; это означает **недостаточно доказательств
эквивалентности**, а не доказанное большое различие. У `α=0.5` появляются уже
и крупные систематические разрывы.

Важно: высокая top-1 stability у плохой модели не всегда означает полезную
robustness — вырожденный или плохо различающий входы predictor тоже может мало
менять решение. Поэтому кривые интерпретируются только вместе с clean PPL/KL.

## Свободная генерация

![Свободная генерация](outputs/05_free_generation.png)

Даже лучший Soft `α=0.05` не дал ни одного полностью совпавшего continuation из
16; среднее совпадение позиций — 30.08%, общий префикс — 7.63 из 32 токенов.
При этом KL на обеих траекториях около 0.03. Это ожидаемая нелинейность greedy:
небольшое изменение вероятностей может поменять argmax, после чего контексты
расходятся каскадно.

Hard `α=0.05` дал 2/16 точных продолжений и 28.32% совпадения токенов, но его
trajectory KL около 2.2. То есть редкое точное совпадение greedy-пути не
компенсирует сильное расхождение распределений.

## Этап 4. Linear Probing

Задача — предсказать UPOS **следующего слова** по состоянию непосредственно
перед его первым субтокеном, поэтому target-слово не видно causal-модели.
Teacher probe обучен на 12 000 целях и проверен на независимых 4 000 целях из
UD English EWT. Средняя accuracy Teacher по слоям 7/14/21 —
**{teacher_probe_mean * 100:.2f}%**.

{linear_probe_table(probe)}

![Linear probing](outputs/06_linear_probing.png)

По верхней границе исходного критерия Qwen (`drop ≤ 5 п.п.` на каждом из трёх
слоёв) этап проходят **6 из 10** моделей; по строгой нижней границе `≤3 п.п.` —
**3 из 10**.

Исходный Qwen cross-probe почти переносится при малом шуме, но разрыв растёт с
α: максимальное среднее падение — **{raw_probe_drops[worst_raw_probe] * 100:.2f}
п.п. у `{worst_raw_probe}`**. Отдельно обученные Student probes отделяют потерю
информации от смены координат: вплоть до α=0.35 они в среднем не хуже Teacher,
а при α=0.5 уже падают; худшее среднее падение —
**{separate_probe_drops[worst_separate_probe] * 100:.2f} п.п. у
`{worst_separate_probe}`**. Procrustes не устраняет разрыв при больших α, то
есть простой общий поворот/масштаб недостаточен.

## Этап 5. Attention Alignment

Для каждой модели cosine посчитан по полным causal attention-матрицам каждой из
448 пар «тот же слой, тот же номер головы», накопленным на 100 одинаковых
входах. Ниже исходная метрика Qwen и дополнительный permutation-aware контроль.

{attention_table(attention)}

![Attention alignment](outputs/07_attention_alignment.png)

Самый низкий средний same-index cosine —
**{attention_means[lowest_attention_model]:.4f} у `{lowest_attention_model}`**;
минимальная отдельная голова — **{attention_mins[lowest_attention_head_model]:.4f}
у `{lowest_attention_head_model}`**. Hungarian matching заметно меняет результат
только у наиболее повреждённых моделей; значит, основное расхождение нельзя
объяснить одной лишь перестановкой голов.

Связь проверена со всеми этапами 1–3. Для среднего attention cosine Pearson
равен `r={attention_top1_corr['pearson_r']:.3f}` с top-1 (этап 1),
`r={attention_kl_corr['pearson_r']:.3f}` с KL (этап 1),
`r={attention_error_corr['pearson_r']:.3f}` с точным совпадением ошибок
(этап 2) и `r={attention_robust_corr['pearson_r']:.3f}` со средним абсолютным
robustness-gap (этап 3). Число критически низких голов ненулевое только у двух
моделей α=0.5, поэтому его корреляции особенно нестабильны: `r={low_top1_corr['pearson_r']:.3f}`
с top-1 и `r={low_robust_corr['pearson_r']:.3f}` с robustness-gap. Это
исследовательские корреляции при `n=10`, без поправки на множественные проверки:
они показывают связь, но не причинность.

## Soft KD против Hard CE напрямую

{pair_table(pairs)}

Неожиданный выходной эффект: с ростом α обе модели удаляются от Teacher, но
`KL(Soft||Hard)` падает с 2.5456 до 0.4582. Это обосновывает новую гипотезу:
**при большом исходном повреждении общий noisy initialization, данные и режим
обучения сильнее определяют финальный attractor, чем различие soft/hard
objective.** Сейчас это корреляционное наблюдение на одном noise seed. Для
проверки нужны минимум 3–5 независимых noise/data seeds.

## Проверка объяснения про «поворот базиса»

Эта часть не использовалась для решения о функциональной эквивалентности; она
нужна только потому, что в исходной истории именно поворот был предложен как
объяснение низкого CKA.

- Локальная реализация linear CKA прямо документирует инвариантность к
  ортогональным преобразованиям. Синтетическая проверка дала
  `CKA(X, XQ) = {synthetic['cka_original_vs_orthogonal_rotation']:.6f}` для
  ортогонального `Q`, тогда как анизотропное преобразование дало
  `{synthetic['cka_original_vs_anisotropic_transform']:.4f}`.
- У Teacher и всех Students `tie_word_embeddings=true`, а input embedding и
  LM head реально делят одно storage. Следовательно, объяснение «свободный
  независимый LM head повернулся обратно» к этой архитектуре неприменимо.
- На верхнем слое при `α=0.5` held-out scaled orthogonal Procrustes даёт
  `R²=0.029` для Soft и `R²=-0.038` для Hard. Простой общий поворот+масштаб не
  переносится на held-out половину.
- Одновременно явного rank collapse нет: effective-rank ratio Student/Teacher
  на верхнем слое равен 1.006 для Soft и 1.423 для Hard. Значит, наблюдение
  больше похоже на неортогональную и sample-dependent перестройку геометрии,
  а не на чистый поворот или потерю ранга.

{audit_table(audit)}

## Проверяемые гипотезы после этого прогона

1. **H1 — исходная функциональная эквивалентность:** опровергнута в заданных
   строгих порогах для всех десяти checkpoints.
2. **H2 — Hard CE восстанавливает решения, но не вероятностное поведение:**
   подтверждается сочетанием сравнительно высокого top-1 и KL/PPL на порядок
   хуже Soft KD при малых α.
3. **H3 — низкий CKA объясняется чистым ортогональным поворотом:**
   опровергается инвариантностью CKA и held-out Procrustes.
4. **H4 — низкий CKA вызван rank collapse:** не поддерживается выбранными
   контрольными моделями; effective rank не падает.
5. **H5 — общий attractor Soft/Hard при больших α:** выдвинута из монотонного
   падения KL между двумя Students; требует независимых seeds.
6. **H6 — малые распределительные расхождения достаточно без free-running
   теста:** опровергается каскадным расхождением greedy continuations даже у
   Soft `α=0.05`.
7. **H7 — провал Teacher→Student linear probe обязательно означает потерю
   информации:** опровергается для α≤0.35 отдельными Student probes; при α=0.5
   появляется уже и собственная потеря probe-accuracy.
8. **H8 — attention alignment связан с выходной близостью:** поддерживается
   корреляцией по этим десяти checkpoints, но требует независимых seeds для
   проверки воспроизводимости и не доказывает причинную связь.

## Ограничения и следующий строгий шаг

- 4 080 next-token позиций достаточно для первичной проверки, но всего 16
  блоков ограничивают точность block-bootstrap, особенно для equivalence band
  `±2 п.п.`.
- Корпус один и англоязычный; нет out-of-domain, русского текста, кода,
  математики и длинных контекстов.
- Noise/data seed один. Нельзя отделить общую закономерность от конкретной
  траектории оптимизации.
- POS probe проверяет одну англоязычную морфосинтаксическую задачу; вывод нельзя
  автоматически переносить на семантику или другие языки.
- Attention cosine усредняет полные матрицы, в которых общая causal-структура
  может повышать сходство; дополнительно полезны сравнения attention output и
  функциональные head-ablation тесты.
- Absolute embedding noise не нормирован на RMS embeddings и быстро попадает в
  saturation. Следующий прогон должен добавить относительный шум
  `σ_rel × RMS(embedding)` при сохранении исходных уровней как контрольных.
- Для финального утверждения: 5 seeds × минимум 64 независимых блоков,
  русский/английский/code/math strata, длинные контексты и предварительно
  зарегистрированные equivalence margins.

## Артефакты

- `outputs/raw_results.json` — сырые block-level результаты и bootstrap.
- `outputs/summary.csv` — компактная сводка.
- `outputs/soft_vs_hard.json` — прямое сравнение Students.
- `outputs/representation_hypothesis_audit.json` — вторичная проверка
  rotation/rank гипотез.
- `outputs/linear_probe_results.json` — Stage 4: все probe-метрики, per-class
  accuracy и confusion matrices.
- `outputs/attention_alignment_results.json` — Stage 5: все 448 сравнений голов
  на модель, Hungarian mappings и корреляции с выходными метриками.
- `evaluate_outputs.py`, `compare_soft_hard.py`,
  `audit_representation_hypotheses.py`, `run_linear_probing.py`,
  `run_attention_alignment.py` — воспроизводимый код.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    configure_plotting()
    raw = load_json(RAW_PATH)
    pairs = load_json(PAIR_PATH)
    audit = load_json(AUDIT_PATH)
    probe = load_json(PROBE_PATH)
    attention = load_json(ATTENTION_PATH)
    save_output_fidelity(raw, pairs)
    save_error_analysis(raw)
    save_embedding_robustness(raw)
    save_token_robustness(raw)
    save_generation(raw)
    save_linear_probing(probe)
    save_attention_alignment(attention)
    write_summary_csv(raw, probe, attention)
    write_report(raw, pairs, audit, probe, attention)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
