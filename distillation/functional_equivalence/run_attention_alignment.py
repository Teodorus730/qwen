"""Measure Teacher/Student attention alignment for stage 5.

The primary metric follows the original Qwen proposal exactly: cosine
similarity between same-index attention heads at every layer.  Because head
indices are not identifiable under a permutation, the script also reports a
Hungarian-matched control without replacing the original metric.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
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
    parser.add_argument("--output", default=str(
        SCRIPT_DIR / "outputs" / "attention_alignment_results.json"
    ))
    parser.add_argument("--output-results", default=str(
        SCRIPT_DIR / "outputs" / "raw_results.json"
    ))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_eager_model(
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
        attn_implementation="eager",
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def cache_teacher_attentions(
    model,
    blocks: torch.Tensor,
    device: torch.device,
) -> list[tuple[torch.Tensor, ...]]:
    cache: list[tuple[torch.Tensor, ...]] = []
    for index, block in enumerate(blocks):
        output = model(
            input_ids=block.unsqueeze(0).to(device),
            output_attentions=True,
            use_cache=False,
        )
        if output.attentions is None:
            raise RuntimeError("The model did not return attention matrices in eager mode")
        cache.append(tuple(
            layer[0].detach().to(device="cpu", dtype=torch.float16)
            for layer in output.attentions
        ))
        del output
        if (index + 1) % 10 == 0 or index + 1 == len(blocks):
            print(f"    teacher examples: {index + 1}/{len(blocks)}", flush=True)
    return cache


@torch.inference_mode()
def compare_student(
    model,
    blocks: torch.Tensor,
    teacher_cache: list[tuple[torch.Tensor, ...]],
    device: torch.device,
    low_threshold: float,
) -> dict[str, Any]:
    first_teacher = teacher_cache[0]
    n_layers = len(first_teacher)
    n_heads = first_teacher[0].shape[0]
    dot = torch.zeros((n_layers, n_heads, n_heads), dtype=torch.float64, device=device)
    teacher_norm = torch.zeros((n_layers, n_heads), dtype=torch.float64, device=device)
    student_norm = torch.zeros((n_layers, n_heads), dtype=torch.float64, device=device)

    for example_index, block in enumerate(blocks):
        output = model(
            input_ids=block.unsqueeze(0).to(device),
            output_attentions=True,
            use_cache=False,
        )
        if output.attentions is None or len(output.attentions) != n_layers:
            raise RuntimeError("Teacher and Student attention layer counts differ")
        for layer_index, (teacher_layer, student_layer) in enumerate(
            zip(teacher_cache[example_index], output.attentions)
        ):
            teacher_flat = teacher_layer.to(device=device, dtype=torch.float32).flatten(1)
            student_flat = student_layer[0].float().flatten(1)
            if teacher_flat.shape != student_flat.shape:
                raise RuntimeError(
                    f"Attention shape mismatch at layer {layer_index}: "
                    f"{tuple(teacher_flat.shape)} vs {tuple(student_flat.shape)}"
                )
            dot[layer_index] += (teacher_flat @ student_flat.T).double()
            teacher_norm[layer_index] += teacher_flat.square().sum(1).double()
            student_norm[layer_index] += student_flat.square().sum(1).double()
        del output
        if (example_index + 1) % 10 == 0 or example_index + 1 == len(blocks):
            print(f"    student examples: {example_index + 1}/{len(blocks)}", flush=True)

    cosine = dot / (
        teacher_norm.sqrt().unsqueeze(2) * student_norm.sqrt().unsqueeze(1)
    ).clamp_min(1e-12)
    cosine_np = cosine.cpu().numpy()

    layers: dict[str, Any] = {}
    same_values: list[float] = []
    matched_values: list[float] = []
    low_same: list[dict[str, Any]] = []
    low_matched: list[dict[str, Any]] = []
    for layer_index in range(n_layers):
        matrix = cosine_np[layer_index]
        same = np.diag(matrix)
        teacher_indices, student_indices = linear_sum_assignment(-matrix)
        matched = matrix[teacher_indices, student_indices]
        mapping = [
            {
                "teacher_head": int(teacher_head),
                "student_head": int(student_head),
                "cosine": float(value),
            }
            for teacher_head, student_head, value in zip(
                teacher_indices, student_indices, matched
            )
        ]
        for head, value in enumerate(same):
            if value < low_threshold:
                low_same.append({
                    "layer": layer_index,
                    "teacher_head": head,
                    "student_head": head,
                    "cosine": float(value),
                })
        for item in mapping:
            if item["cosine"] < low_threshold:
                low_matched.append({"layer": layer_index, **item})
        layers[str(layer_index)] = {
            "same_index_cosine_by_head": same.tolist(),
            "same_index_mean": float(same.mean()),
            "same_index_min": float(same.min()),
            "matched_mean": float(matched.mean()),
            "matched_min": float(matched.min()),
            "hungarian_mapping": mapping,
            "all_pairs_cosine": matrix.tolist(),
        }
        same_values.extend(same.tolist())
        matched_values.extend(matched.tolist())

    return {
        "layers": layers,
        "summary": {
            "same_index_mean": float(np.mean(same_values)),
            "same_index_min": float(np.min(same_values)),
            "matched_mean": float(np.mean(matched_values)),
            "matched_min": float(np.min(matched_values)),
            "same_index_low_count": len(low_same),
            "matched_low_count": len(low_matched),
            "total_heads": len(same_values),
        },
        "same_index_heads_below_threshold": low_same,
        "matched_heads_below_threshold": low_matched,
    }


def estimate(value: Any) -> float:
    if isinstance(value, dict) and "estimate" in value:
        return float(value["estimate"])
    return float(value)


def output_metrics(model_result: dict[str, Any]) -> dict[str, float]:
    baseline = model_result["baseline"]
    stability_gaps = [
        abs(estimate(condition["stability_gap_student_minus_teacher"]))
        for condition in model_result["robustness"].values()
    ]
    return {
        "top1_match": estimate(baseline["top1_match"]),
        "kl_teacher_student_t1": estimate(baseline["kl_teacher_student_t1"]),
        "same_wrong_given_teacher_error": estimate(
            baseline["same_wrong_prediction_given_teacher_error"]
        ),
        "mean_abs_robustness_stability_gap": float(np.mean(stability_gaps)),
        "max_abs_robustness_stability_gap": float(np.max(stability_gaps)),
    }


def safe_correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson_r": None, "pearson_p": None,
                "spearman_rho": None, "spearman_p": None}
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def add_correlations(
    result: dict[str, Any],
    output_result_path: Path,
) -> None:
    if not output_result_path.exists():
        result["correlations"] = {"error": f"Missing {output_result_path}"}
        return
    output_results = json.loads(output_result_path.read_text(encoding="utf-8"))
    shared_ids = [
        model_id for model_id in result["models"]
        if model_id in output_results["models"]
    ]
    attention_fields = [
        "same_index_mean", "matched_mean", "same_index_low_count", "matched_low_count"
    ]
    output_by_id = {
        model_id: output_metrics(output_results["models"][model_id])
        for model_id in shared_ids
    }
    correlations: dict[str, Any] = {
        "model_ids": shared_ids,
        "note": "Exploratory correlations over model checkpoints (n=10); p-values are not multiplicity-corrected.",
        "metrics": {},
    }
    for attention_field in attention_fields:
        correlations["metrics"][attention_field] = {}
        x = [
            float(result["models"][model_id]["summary"][attention_field])
            for model_id in shared_ids
        ]
        for output_field in next(iter(output_by_id.values())).keys():
            y = [output_by_id[model_id][output_field] for model_id in shared_ids]
            correlations["metrics"][attention_field][output_field] = safe_correlation(x, y)
    result["correlations"] = correlations


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    attn_cfg = cfg["attention_alignment"]
    runtime = cfg["runtime"]
    seed_everything(int(runtime["seed"]))
    device = torch.device(runtime.get("device", "cuda"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[runtime.get("dtype", "bfloat16")]
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(runtime.get("cuda_memory_fraction", 1.0))
        )

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
        seq_len=int(attn_cfg["seq_len"]),
        n_blocks=int(attn_cfg["examples"]),
    )
    block_hash = hashlib.sha256(blocks.numpy().tobytes()).hexdigest()
    print("[teacher] caching attention matrices", flush=True)
    teacher = load_eager_model(
        cfg["teacher_model"], device, dtype, cache_dir=args.cache_dir
    )
    teacher_cache = cache_teacher_attentions(teacher, blocks, device)
    n_layers = len(teacher_cache[0])
    n_heads = teacher_cache[0][0].shape[0]
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_path = Path(args.output).resolve()
    if args.resume and output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        result = {
            "definition": {
                "primary": "Cosine similarity of flattened same-index attention matrices, accumulated over all examples.",
                "control": "Maximum-weight one-to-one head matching (Hungarian algorithm) within each layer.",
                "examples": len(blocks),
                "seq_len": int(attn_cfg["seq_len"]),
                "layers": n_layers,
                "heads_per_layer": n_heads,
                "total_head_comparisons_per_model": n_layers * n_heads,
                "low_similarity_threshold": float(attn_cfg["low_similarity_threshold"]),
                "input_token_ids_sha256": block_hash,
            },
            "models": {},
        }

    selected = set(args.only or [])
    model_specs = [spec for spec in cfg["models"] if not selected or spec["id"] in selected]
    for model_index, spec in enumerate(model_specs):
        model_id = spec["id"]
        if args.resume and model_id in result["models"]:
            print(f"[resume] {model_id}", flush=True)
            continue
        print(f"[student {model_index + 1}/{len(model_specs)}] {model_id}", flush=True)
        student = load_eager_model(
            resolve_relative(spec["path"], config_dir), device, dtype
        )
        model_result = compare_student(
            student, blocks, teacher_cache, device,
            float(attn_cfg["low_similarity_threshold"]),
        )
        model_result["objective"] = spec["objective"]
        model_result["alpha"] = float(spec["alpha"])
        result["models"][model_id] = model_result
        atomic_json_dump(result, output_path)
        del student
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["complete"] = len(result["models"]) == len(model_specs)
    add_correlations(result, Path(args.output_results).resolve())
    atomic_json_dump(result, output_path)
    del teacher_cache
    print(f"[complete] {output_path}", flush=True)


if __name__ == "__main__":
    main()
