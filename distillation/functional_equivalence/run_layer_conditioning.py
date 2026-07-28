"""Estimate local layer conditioning under small input perturbations.

For a hidden state h_l(x), the cumulative directional condition estimate is

    kappa_l(x, delta) =
        (||h_l(x + delta) - h_l(x)|| / ||h_l(x)||)
        -------------------------------------------------
                   (||delta|| / ||x||).

The incremental estimate for decoder block l divides the relative change after
the block by the relative change before it.  Multiple deterministic Gaussian
directions and perturbation magnitudes are evaluated.  Their maxima are still
finite-direction lower estimates of the worst-case Jacobian norm, not exact
spectral condition numbers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_outputs import (
    atomic_json_dump,
    load_text_blocks,
    resolve_relative,
    seed_everything,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "outputs" / "layer_conditioning_results.json"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one block, one direction and the primary epsilon.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_model(
    source: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    cache_dir: str | None = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=Path(str(source)).exists(),
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def epsilon_key(value: float) -> str:
    return f"{value:g}"


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "min": float(array.min()),
        "n": int(array.size),
    }


def condition_profile(
    clean_states: tuple[torch.Tensor, ...],
    noisy_states: tuple[torch.Tensor, ...],
) -> tuple[list[float], list[float], list[float | None]]:
    """Return relative changes, cumulative kappa and incremental kappa."""
    if len(clean_states) != len(noisy_states):
        raise ValueError("Clean and perturbed hidden-state counts differ")
    relative_changes: list[float] = []
    for clean, noisy in zip(clean_states, noisy_states):
        numerator = torch.linalg.vector_norm(noisy.float() - clean.float())
        denominator = torch.linalg.vector_norm(clean.float()).clamp_min(1e-12)
        relative_changes.append(float((numerator / denominator).item()))
    input_relative_change = max(relative_changes[0], 1e-12)
    cumulative = [
        float(value / input_relative_change) for value in relative_changes
    ]
    incremental: list[float | None] = [None]
    for layer in range(1, len(relative_changes)):
        incremental.append(
            float(relative_changes[layer] / max(relative_changes[layer - 1], 1e-12))
        )
    return relative_changes, cumulative, incremental


def summarize_records(
    records: list[dict[str, Any]],
    epsilons: list[float],
    n_states: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for epsilon in epsilons:
        selected = [
            record for record in records
            if abs(float(record["epsilon"]) - epsilon) < 1e-15
        ]
        layers: dict[str, Any] = {}
        for layer in range(n_states):
            relative = [
                float(record["relative_changes"][layer]) for record in selected
            ]
            cumulative = [
                float(record["cumulative_condition"][layer]) for record in selected
            ]
            layer_result: dict[str, Any] = {
                "relative_change": summarize(relative),
                "cumulative_condition": summarize(cumulative),
            }
            if layer > 0:
                incremental = [
                    float(record["incremental_condition"][layer])
                    for record in selected
                ]
                layer_result["incremental_condition"] = summarize(incremental)
            layers[str(layer)] = layer_result
        result[epsilon_key(epsilon)] = {
            "epsilon": epsilon,
            "samples": len(selected),
            "layers": layers,
        }
    return result


def primary_summary(
    epsilon_results: dict[str, Any],
    primary_epsilon: float,
) -> dict[str, Any]:
    primary = epsilon_results[epsilon_key(primary_epsilon)]
    decoder_layers = {
        layer: metrics
        for layer, metrics in primary["layers"].items()
        if int(layer) > 0
    }
    max_cumulative_layer = max(
        decoder_layers,
        key=lambda layer: decoder_layers[layer]["cumulative_condition"]["p95"],
    )
    max_incremental_layer = max(
        decoder_layers,
        key=lambda layer: decoder_layers[layer]["incremental_condition"]["p95"],
    )
    final_layer = str(max(map(int, decoder_layers)))
    return {
        "epsilon": primary_epsilon,
        "max_cumulative_layer": int(max_cumulative_layer),
        "max_cumulative_p95": decoder_layers[max_cumulative_layer][
            "cumulative_condition"
        ]["p95"],
        "max_incremental_layer": int(max_incremental_layer),
        "max_incremental_p95": decoder_layers[max_incremental_layer][
            "incremental_condition"
        ]["p95"],
        "final_layer_cumulative_median": decoder_layers[final_layer][
            "cumulative_condition"
        ]["median"],
        "final_layer_cumulative_p95": decoder_layers[final_layer][
            "cumulative_condition"
        ]["p95"],
    }


@torch.inference_mode()
def evaluate_model(
    model,
    blocks: torch.Tensor,
    epsilons: list[float],
    directions_per_block: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    n_states: int | None = None
    for block_index, block in enumerate(blocks):
        input_ids = block.unsqueeze(0).to(device)
        embeddings = model.get_input_embeddings()(input_ids).float()
        clean_output = model(
            inputs_embeds=embeddings,
            output_hidden_states=True,
            use_cache=False,
        )
        clean_states = tuple(clean_output.hidden_states)
        if n_states is None:
            n_states = len(clean_states)
        elif n_states != len(clean_states):
            raise RuntimeError("Hidden-state count changed between blocks")
        embedding_norm = torch.linalg.vector_norm(embeddings).clamp_min(1e-12)

        noisy_embeddings_rows: list[torch.Tensor] = []
        variant_metadata: list[dict[str, Any]] = []
        for direction_index in range(directions_per_block):
            direction_seed = seed + block_index * 1009 + direction_index * 9176
            generator = torch.Generator(device="cpu").manual_seed(direction_seed)
            base_direction = torch.randn(
                embeddings.shape, generator=generator, dtype=torch.float32
            ).to(device)
            base_direction /= torch.linalg.vector_norm(base_direction).clamp_min(1e-12)
            for epsilon in epsilons:
                delta = base_direction * (epsilon * embedding_norm)
                achieved_epsilon = float(
                    (
                        torch.linalg.vector_norm(delta)
                        / embedding_norm
                    ).item()
                )
                noisy_embeddings_rows.append(embeddings + delta)
                variant_metadata.append({
                    "block": block_index,
                    "direction": direction_index,
                    "direction_seed": direction_seed,
                    "epsilon": epsilon,
                    "achieved_relative_input_change": achieved_epsilon,
                })
                del delta
        noisy_embeddings_batch = torch.cat(noisy_embeddings_rows, dim=0)
        noisy_output = model(
            inputs_embeds=noisy_embeddings_batch,
            output_hidden_states=True,
            use_cache=False,
        )
        for row, metadata in enumerate(variant_metadata):
            noisy_states = tuple(
                state[row:row + 1] for state in noisy_output.hidden_states
            )
            relative, cumulative, incremental = condition_profile(
                clean_states, noisy_states
            )
            records.append({
                **metadata,
                "relative_changes": relative,
                "cumulative_condition": cumulative,
                "incremental_condition": incremental,
            })
        del noisy_output, noisy_embeddings_batch, noisy_embeddings_rows
        del clean_output, clean_states, embeddings, input_ids
        print(f"    blocks: {block_index + 1}/{len(blocks)}", flush=True)
    assert n_states is not None
    epsilon_results = summarize_records(records, epsilons, n_states)
    return {
        "hidden_state_count": n_states,
        "decoder_layer_count": n_states - 1,
        "records": records,
        "epsilons": epsilon_results,
    }


def compare_to_teacher(
    student: dict[str, Any],
    teacher: dict[str, Any],
    primary_epsilon: float,
) -> dict[str, Any]:
    key = epsilon_key(primary_epsilon)
    student_layers = student["epsilons"][key]["layers"]
    teacher_layers = teacher["epsilons"][key]["layers"]
    layers: dict[str, Any] = {}
    for layer in student_layers:
        if int(layer) == 0:
            continue
        student_cumulative = student_layers[layer]["cumulative_condition"]["median"]
        teacher_cumulative = teacher_layers[layer]["cumulative_condition"]["median"]
        student_incremental = student_layers[layer]["incremental_condition"]["median"]
        teacher_incremental = teacher_layers[layer]["incremental_condition"]["median"]
        layers[layer] = {
            "cumulative_median_difference": student_cumulative - teacher_cumulative,
            "cumulative_median_ratio": student_cumulative / max(
                teacher_cumulative, 1e-12
            ),
            "incremental_median_difference": student_incremental - teacher_incremental,
            "incremental_median_ratio": student_incremental / max(
                teacher_incremental, 1e-12
            ),
        }
    worst_cumulative = max(
        layers,
        key=lambda layer: abs(np.log(max(layers[layer]["cumulative_median_ratio"], 1e-12))),
    )
    worst_incremental = max(
        layers,
        key=lambda layer: abs(np.log(max(layers[layer]["incremental_median_ratio"], 1e-12))),
    )
    return {
        "epsilon": primary_epsilon,
        "layers": layers,
        "largest_cumulative_ratio_deviation_layer": int(worst_cumulative),
        "largest_cumulative_ratio_deviation": layers[worst_cumulative][
            "cumulative_median_ratio"
        ],
        "largest_incremental_ratio_deviation_layer": int(worst_incremental),
        "largest_incremental_ratio_deviation": layers[worst_incremental][
            "incremental_median_ratio"
        ],
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    condition_cfg = cfg["layer_conditioning"]
    runtime = cfg["runtime"]
    seed = int(runtime["seed"])
    seed_everything(seed)
    device = torch.device(runtime.get("device", "cuda"))
    dtype = dtype_from_name(condition_cfg.get("dtype", "float32"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(runtime.get("cuda_memory_fraction", 1.0))
        )

    epsilons = [float(value) for value in condition_cfg["relative_epsilons"]]
    primary_epsilon = float(condition_cfg["primary_epsilon"])
    blocks_count = int(condition_cfg["blocks"])
    directions = int(condition_cfg["directions_per_block"])
    if args.smoke:
        epsilons = [primary_epsilon]
        blocks_count = 1
        directions = 1

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["teacher_model"], cache_dir=args.cache_dir
    )
    data_cfg = cfg["data"]
    blocks = load_text_blocks(
        tokenizer=tokenizer,
        jsonl_path=resolve_relative(data_cfg["jsonl"], config_dir),
        text_field=data_cfg["text_field"],
        score_field=data_cfg["score_field"],
        min_score=float(data_cfg["min_score"]),
        skip_docs=int(data_cfg["skip_docs"]),
        seq_len=int(condition_cfg["seq_len"]),
        n_blocks=blocks_count,
    )
    input_hash = hashlib.sha256(blocks.numpy().tobytes()).hexdigest()

    output_path = Path(args.output).resolve()
    if args.resume and output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        result = {
            "definition": {
                "name": "Empirical local directional layer condition estimate",
                "cumulative_formula": "(||delta h_l|| / ||h_l||) / (||delta x|| / ||x||)",
                "incremental_formula": "(||delta h_l|| / ||h_l||) / (||delta h_(l-1)|| / ||h_(l-1)||)",
                "interpretation": (
                    "Finite-direction lower estimate of worst-case local "
                    "sensitivity; not the exact spectral condition number of "
                    "the full Jacobian."
                ),
                "blocks": blocks_count,
                "seq_len": int(condition_cfg["seq_len"]),
                "directions_per_block": directions,
                "relative_epsilons": epsilons,
                "primary_epsilon": primary_epsilon,
                "dtype": str(dtype).replace("torch.", ""),
                "input_token_ids_sha256": input_hash,
                "noise": (
                    "Deterministic Gaussian directions normalized to exact "
                    "relative Frobenius perturbation per model and block."
                ),
            },
            "teacher": None,
            "models": {},
        }

    if result.get("teacher") is None:
        print("[teacher] local layer conditioning", flush=True)
        teacher_model = load_model(
            cfg["teacher_model"], device, dtype, cache_dir=args.cache_dir
        )
        teacher_result = evaluate_model(
            teacher_model, blocks, epsilons, directions, seed, device
        )
        teacher_result["primary_summary"] = primary_summary(
            teacher_result["epsilons"], primary_epsilon
        )
        result["teacher"] = teacher_result
        atomic_json_dump(result, output_path)
        del teacher_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = set(args.only or [])
    all_specs = cfg["models"]
    model_specs = [
        spec for spec in all_specs if not selected or spec["id"] in selected
    ]
    for model_index, spec in enumerate(model_specs):
        model_id = spec["id"]
        if args.resume and model_id in result["models"]:
            print(f"[resume] {model_id}", flush=True)
            continue
        print(f"[student {model_index + 1}/{len(model_specs)}] {model_id}", flush=True)
        model = load_model(resolve_relative(spec["path"], config_dir), device, dtype)
        model_result = evaluate_model(
            model, blocks, epsilons, directions, seed, device
        )
        model_result["objective"] = spec["objective"]
        model_result["alpha"] = float(spec["alpha"])
        model_result["primary_summary"] = primary_summary(
            model_result["epsilons"], primary_epsilon
        )
        model_result["comparison_to_teacher"] = compare_to_teacher(
            model_result, result["teacher"], primary_epsilon
        )
        result["models"][model_id] = model_result
        atomic_json_dump(result, output_path)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_ids = {spec["id"] for spec in all_specs}
    result["complete"] = expected_ids.issubset(result["models"])
    atomic_json_dump(result, output_path)
    print(f"[complete={result['complete']}] {output_path}", flush=True)


if __name__ == "__main__":
    main()
