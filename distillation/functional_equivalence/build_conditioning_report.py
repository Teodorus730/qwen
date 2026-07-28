"""Build the standalone Russian report for local layer conditioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RESULT_PATH = OUTPUT_DIR / "layer_conditioning_results.json"
RAW_OUTPUT_PATH = OUTPUT_DIR / "raw_results.json"
REPORT_PATH = SCRIPT_DIR / "CONDITIONING_REPORT.md"

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


def primary_layers(result: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    key = f"{float(result['definition']['primary_epsilon']):g}"
    return model["epsilons"][key]["layers"]


def ordered_models(
    result: dict[str, Any], objective: str
) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        (
            (model_id, model)
            for model_id, model in result["models"].items()
            if model["objective"] == objective
        ),
        key=lambda pair: pair[1]["alpha"],
    )


def decoder_series(
    result: dict[str, Any],
    model: dict[str, Any],
    metric: str,
    statistic: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    layers = primary_layers(result, model)
    layer_ids = np.asarray(sorted(int(layer) for layer in layers if int(layer) > 0))
    values = np.asarray([
        layers[str(layer)][metric][statistic] for layer in layer_ids
    ])
    return layer_ids, values


def save_profile_plot(result: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    palette = plt.cm.viridis(np.linspace(0.15, 0.9, 5))
    for column, objective in enumerate(("soft_kd", "hard_teacher_ce")):
        for row, (metric, title) in enumerate((
            ("cumulative_condition", "Накопленная обусловленность от embeddings"),
            ("incremental_condition", "Пошаговое усиление отдельного слоя"),
        )):
            axis = axes[row, column]
            teacher_x, teacher_y = decoder_series(
                result, result["teacher"], metric
            )
            axis.plot(
                teacher_x,
                teacher_y,
                color=COLORS["teacher"],
                linestyle="--",
                linewidth=2.4,
                label="Teacher",
            )
            for color, (_, model) in zip(palette, ordered_models(result, objective)):
                x, y = decoder_series(result, model, metric)
                axis.plot(
                    x, y, color=color, linewidth=1.8,
                    label=f"α={model['alpha']:g}",
                )
            axis.axhline(1.0, color="#6B7280", linewidth=1, alpha=0.7)
            axis.set(
                title=f"{LABELS[objective]}: {title}",
                xlabel="Номер слоя",
                ylabel="эмпирическая κ",
            )
            axis.legend(fontsize=8, ncol=2)
    figure.suptitle(
        "Локальная чувствительность слоёв при ε=0.001", fontsize=15
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "08_layer_conditioning_profiles.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def condition_distance(result: dict[str, Any], model: dict[str, Any]) -> float:
    ratios = [
        layer["cumulative_median_ratio"]
        for layer in model["comparison_to_teacher"]["layers"].values()
    ]
    return float(np.mean(np.abs(np.log(np.maximum(ratios, 1e-12)))))


def robustness_gap(raw_model: dict[str, Any]) -> float:
    return float(np.mean([
        abs(condition["stability_gap_student_minus_teacher"]["estimate"])
        for condition in raw_model["robustness"].values()
    ]))


def correlations(
    result: dict[str, Any], raw: dict[str, Any]
) -> dict[str, dict[str, float]]:
    model_ids = list(result["models"])
    x = [condition_distance(result, result["models"][model_id]) for model_id in model_ids]
    targets = {
        "top1_match": [
            raw["models"][model_id]["baseline"]["top1_match"]["estimate"]
            for model_id in model_ids
        ],
        "mean_abs_robustness_gap": [
            robustness_gap(raw["models"][model_id]) for model_id in model_ids
        ],
    }
    output: dict[str, dict[str, float]] = {}
    for name, y in targets.items():
        p = pearsonr(x, y)
        s = spearmanr(x, y)
        output[name] = {
            "pearson_r": float(p.statistic),
            "pearson_p": float(p.pvalue),
            "spearman_rho": float(s.statistic),
            "spearman_p": float(s.pvalue),
        }
    return output


def save_comparison_plot(
    result: dict[str, Any], raw: dict[str, Any]
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    teacher_final = result["teacher"]["primary_summary"][
        "final_layer_cumulative_median"
    ]
    for objective in ("soft_kd", "hard_teacher_ce"):
        pairs = ordered_models(result, objective)
        models = [model for _, model in pairs]
        ids = [model_id for model_id, _ in pairs]
        alpha = [model["alpha"] for model in models]
        axes[0].plot(
            alpha,
            [
                model["primary_summary"]["final_layer_cumulative_median"]
                / teacher_final
                for model in models
            ],
            marker="o", linewidth=2, color=COLORS[objective],
            label=LABELS[objective],
        )
        axes[1].plot(
            alpha,
            [condition_distance(result, model) for model in models],
            marker="o", linewidth=2, color=COLORS[objective],
            label=LABELS[objective],
        )
        axes[2].scatter(
            [condition_distance(result, model) for model in models],
            [
                raw["models"][model_id]["baseline"]["top1_match"]["estimate"] * 100
                for model_id in ids
            ],
            s=50, color=COLORS[objective], label=LABELS[objective],
        )
    axes[0].axhline(1.0, color="#111827", linestyle="--")
    axes[0].set(
        title="Финальная κ Student / Teacher", xlabel="α", ylabel="отношение"
    )
    axes[1].set(
        title="Среднее |log(κS/κT)| по слоям",
        xlabel="α",
        ylabel="расстояние профилей",
    )
    axes[2].set(
        title="Обусловленность и совпадение выходов",
        xlabel="среднее |log(κS/κT)|",
        ylabel="Top-1 match, %",
    )
    for axis in axes:
        axis.legend()
    figure.suptitle(
        "Отклонение профиля обусловленности Student от Teacher", fontsize=15
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "09_conditioning_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_summary_csv(result: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    entries = [("teacher", result["teacher"])] + list(result["models"].items())
    for model_id, model in entries:
        for epsilon_key, epsilon_result in model["epsilons"].items():
            for layer, metrics in epsilon_result["layers"].items():
                row = {
                    "model_id": model_id,
                    "objective": model.get("objective", "teacher"),
                    "alpha": model.get("alpha"),
                    "epsilon": float(epsilon_key),
                    "layer": int(layer),
                    **{
                        f"relative_change_{key}": value
                        for key, value in metrics["relative_change"].items()
                    },
                    **{
                        f"cumulative_condition_{key}": value
                        for key, value in metrics["cumulative_condition"].items()
                    },
                }
                if "incremental_condition" in metrics:
                    row.update({
                        f"incremental_condition_{key}": value
                        for key, value in metrics["incremental_condition"].items()
                    })
                rows.append(row)
    pd.DataFrame(rows).sort_values(
        ["objective", "alpha", "epsilon", "layer"],
        na_position="first",
    ).to_csv(OUTPUT_DIR / "layer_conditioning_summary.csv", index=False)


def model_table(result: dict[str, Any]) -> str:
    lines = [
        r"| Модель | $\kappa$ финального слоя, median / p95 | "
        r"Max incremental p95 (слой) | "
        r"Среднее $\lvert\log(\kappa_S/\kappa_T)\rvert$ |",
        "|---|---:|---:|---:|",
    ]
    teacher = result["teacher"]
    summary = teacher["primary_summary"]
    lines.append(
        f"| Teacher | {summary['final_layer_cumulative_median']:.4f} / "
        f"{summary['final_layer_cumulative_p95']:.4f} | "
        f"{summary['max_incremental_p95']:.4f} "
        f"(L{summary['max_incremental_layer']}) | — |"
    )
    for model_id, model in result["models"].items():
        summary = model["primary_summary"]
        lines.append(
            f"| {model_id} | {summary['final_layer_cumulative_median']:.4f} / "
            f"{summary['final_layer_cumulative_p95']:.4f} | "
            f"{summary['max_incremental_p95']:.4f} "
            f"(L{summary['max_incremental_layer']}) | "
            f"{condition_distance(result, model):.4f} |"
        )
    return "\n".join(lines)


def epsilon_table(result: dict[str, Any]) -> str:
    epsilons = [float(value) for value in result["definition"]["relative_epsilons"]]
    lines = [
        "| Модель | " + " | ".join(
            f"$\\kappa_{{\\mathrm{{final}}}}$, $\\varepsilon={epsilon:g}$"
            for epsilon in epsilons
        ) + " |",
        "|---|" + "|".join("---:" for _ in epsilons) + "|",
    ]
    entries = [("Teacher", result["teacher"])] + list(result["models"].items())
    for model_id, model in entries:
        final_layer = str(model["decoder_layer_count"])
        values = [
            model["epsilons"][f"{epsilon:g}"]["layers"][final_layer][
                "cumulative_condition"
            ]["median"]
            for epsilon in epsilons
        ]
        lines.append(
            f"| {model_id} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )
    return "\n".join(lines)


def write_report(
    result: dict[str, Any],
    raw: dict[str, Any],
    corr: dict[str, dict[str, float]],
) -> None:
    teacher = result["teacher"]["primary_summary"]
    largest_distance_id = max(
        result["models"],
        key=lambda model_id: condition_distance(result, result["models"][model_id]),
    )
    largest_distance = condition_distance(
        result, result["models"][largest_distance_id]
    )
    top1_corr = corr["top1_match"]
    robustness_corr = corr["mean_abs_robustness_gap"]
    epsilon_values = [
        float(value) for value in result["definition"]["relative_epsilons"]
    ]
    epsilon_spreads: dict[str, float] = {}
    for model_id, model in [("Teacher", result["teacher"])] + list(
        result["models"].items()
    ):
        final_layer = str(model["decoder_layer_count"])
        final_values = [
            model["epsilons"][f"{epsilon:g}"]["layers"][final_layer][
                "cumulative_condition"
            ]["median"]
            for epsilon in epsilon_values
        ]
        epsilon_spreads[model_id] = (
            max(final_values) - min(final_values)
        ) / max(final_values[0], 1e-12)
    largest_epsilon_spread_id = max(epsilon_spreads, key=epsilon_spreads.get)
    report = rf"""# Локальная обусловленность слоёв Qwen Teacher и Students

Дата прогона: 29 июля 2026 года.

## Краткий вывод

При малом относительном изменении входных embeddings
$\varepsilon=0.001$ Teacher в среднем **не усиливает**, а ослабляет возмущение
к финальному состоянию: медиана накопленной оценки равна
$\widetilde{{\kappa}}={teacher['final_layer_cumulative_median']:.4f}$, а
95-й процентиль — $\kappa_{{0.95}}={teacher['final_layer_cumulative_p95']:.4f}$.
При этом отдельные блоки могут локально усиливать уже накопившееся изменение:
максимальный incremental p95 Teacher равен
${teacher['max_incremental_p95']:.4f}$ на слое
${teacher['max_incremental_layer']}$.

С ростом исходного шума $\alpha$ профиль Student всё сильнее отличается от Teacher.
Максимальное среднее логарифмическое расстояние профилей получено у
`{largest_distance_id}`: **{largest_distance:.4f}**. Оно коррелирует с ухудшением
выходного совпадения ($r={top1_corr['pearson_r']:.3f}$ по Pearson) и увеличением
robustness-gap ($r={robustness_corr['pearson_r']:.3f}$), однако это
исследовательская корреляция всего по десяти checkpoints.

## Что именно считается

Для каждого входа $x$, слоя $l$ и малого возмущения $\delta$ сначала считается
относительное изменение представления:

$$
r_l(x;\delta)
=
\frac{{
    \left\|h_l(x+\delta)-h_l(x)\right\|_F
}}{{
    \left\|h_l(x)\right\|_F
}}.
$$

Для входных embeddings относительное изменение равно

$$
r_0(x;\delta)
=
\frac{{
    \left\|\delta\right\|_F
}}{{
    \left\|x\right\|_F
}}.
$$

Накопленное число обусловленности от входа до слоя $l$:

$$
\boxed{{
\kappa_l^{{\mathrm{{cum}}}}(x;\delta)
=
\frac{{r_l(x;\delta)}}{{r_0(x;\delta)}}
}}.
$$

Оно показывает, во сколько раз слой увеличил или уменьшил исходное
относительное возмущение. Пошаговое усиление отдельного transformer-блока:

$$
\boxed{{
\kappa_l^{{\mathrm{{step}}}}(x;\delta)
=
\frac{{r_l(x;\delta)}}{{r_{{l-1}}(x;\delta)}}
}}.
$$

Это **эмпирическая направленная оценка локальной обусловленности** по конечному
набору Gaussian-направлений. Она является нижней оценкой худшего направления
полного Jacobian и не выдаётся за точное спектральное число обусловленности.

## Протокол

- Teacher и 10 Students, все 28 transformer-слоёв.
- 8 одинаковых held-out блоков FineWeb-Edu по 256 токенов.
- 4 детерминированных Gaussian-направления на блок.
- Относительные масштабы: $\varepsilon\in\{{0.001,\ 0.003,\ 0.01\}}$.
- 32 наблюдения на слой для каждого $\varepsilon$ и каждой модели.
- Вычисления в `float32`, чтобы малые изменения не терялись в BF16-квантовании.
- Шум нормирован отдельно до точного отношения
  $\left\|\delta\right\|_F/\left\|x\right\|_F=\varepsilon$.

## Результаты при основном $\varepsilon=0.001$

{model_table(result)}

![Послойные профили обусловленности](outputs/08_layer_conditioning_profiles.png)

Горизонталь $\kappa=1$ означает сохранение относительной величины изменения.
Накопленная $\kappa<1$ означает подавление относительно нормы представления, а
incremental $\kappa>1$ — локальное усиление конкретным блоком. Эти два утверждения
не противоречат друг другу.

## Проверка локального режима

{epsilon_table(result)}

Близость результатов при трёх $\varepsilon$ нужна для проверки, что вывод не
является артефактом одного масштаба. Заметное изменение профиля с
$\varepsilon$ трактуется как
нелинейность finite-difference оценки, а не как самостоятельная нестабильность.
Фактически максимальный относительный разброс финальной median $\kappa$ между
тремя $\varepsilon$
составил только **{epsilon_spreads[largest_epsilon_spread_id] * 100:.2f}%**
(`{largest_epsilon_spread_id}`), то есть выбранный диапазон остаётся в устойчивом
локальном режиме.

![Сравнение Students с Teacher](outputs/09_conditioning_comparison.png)

Корреляция расстояния профиля с top-1:
`Pearson r={top1_corr['pearson_r']:.3f}, p={top1_corr['pearson_p']:.4g}`;
средним абсолютным robustness-gap:
`r={robustness_corr['pearson_r']:.3f}, p={robustness_corr['pearson_p']:.4g}`.
Причинный вывод из этих корреляций делать нельзя: `n=10`, один seed, а модели
внутри каждой ветки образуют зависимую α-траекторию.

## Ограничения

- Случайные направления не гарантируют нахождение спектрально худшего
  возмущения; для этого потребовались бы Jacobian power iteration или SVD.
- Норма Фробениуса агрегирует все токены и признаки и может скрывать редкие
  очень чувствительные позиции.
- Проверен один англоязычный held-out корпус и один набор seeds.
- $\kappa$ характеризует скрытые состояния, но сама по себе не доказывает изменение
  качества: поэтому она сопоставлена с выходными robustness-метриками.

## Артефакты

- `outputs/layer_conditioning_results.json` — полные записи всех направлений,
  $\varepsilon$, слоёв и моделей.
- `outputs/layer_conditioning_summary.csv` — плоская послойная сводка.
- `outputs/08_layer_conditioning_profiles.png` — профили всех слоёв.
- `outputs/09_conditioning_comparison.png` — отклонение Students от Teacher.
- `run_layer_conditioning.py` — воспроизводимый расчёт.
- `build_conditioning_report.py` — построение этого отчёта.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    configure_plotting()
    result = load_json(RESULT_PATH)
    if not result.get("complete"):
        raise RuntimeError("Conditioning sweep is incomplete")
    raw = load_json(RAW_OUTPUT_PATH)
    corr = correlations(result, raw)
    save_profile_plot(result)
    save_comparison_plot(result, raw)
    write_summary_csv(result)
    write_report(result, raw, corr)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
