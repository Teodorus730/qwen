"""Secondary diagnostics for the representation claims in the supplied plan.

These checks do not participate in the output-equivalence verdict. They only
test two proposed explanations for low CKA:

1. "It is just an orthogonal basis rotation."
2. "A freely rotating LM head compensates for that rotation."
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from evaluate_outputs import (
    atomic_json_dump,
    load_model,
    load_text_blocks,
    resolve_relative,
    seed_everything,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--output", default=str(
        SCRIPT_DIR / "outputs" / "representation_hypothesis_audit.json"
    ))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["soft_a0.05", "hard_a0.05", "soft_a0.5", "hard_a0.5"],
    )
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", nargs="*", type=int, default=[0, 14, 28])
    return parser.parse_args()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    cross = (y.T @ x).square().sum()
    xx = (x.T @ x).square().sum().sqrt()
    yy = (y.T @ y).square().sum().sqrt()
    return float((cross / (xx * yy).clamp_min(1e-12)).item())


def effective_rank(centered: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-30).log()
    ).sum()
    return float(entropy.exp().item())


def spectrum_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    sx = torch.linalg.svdvals(x - x.mean(0, keepdim=True)).square()
    sy = torch.linalg.svdvals(y - y.mean(0, keepdim=True)).square()
    sx = sx / sx.sum().clamp_min(1e-12)
    sy = sy / sy.sum().clamp_min(1e-12)
    return float(F_cosine(sx, sy).item())


def F_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x @ y) / (x.norm() * y.norm()).clamp_min(1e-12)


def scaled_orthogonal_procrustes_r2(
    student: torch.Tensor,
    teacher: torch.Tensor,
    train_rows: int,
) -> float:
    """Fit scale + orthogonal map on train rows; score held-out rows."""
    x_train = student[:train_rows]
    y_train = teacher[:train_rows]
    x_test = student[train_rows:]
    y_test = teacher[train_rows:]
    x_mean = x_train.mean(0, keepdim=True)
    y_mean = y_train.mean(0, keepdim=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    u, singular, vh = torch.linalg.svd(
        x_centered.T @ y_centered, full_matrices=False
    )
    rotation = u @ vh
    scale = singular.sum() / x_centered.square().sum().clamp_min(1e-12)
    prediction = (x_test - x_mean) @ rotation * scale + y_mean
    residual = (y_test - prediction).square().sum()
    total = (y_test - y_train.mean(0, keepdim=True)).square().sum()
    return float((1.0 - residual / total.clamp_min(1e-12)).item())


@torch.inference_mode()
def collect_layers(
    model,
    blocks: torch.Tensor,
    layers: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for block in blocks:
        output = model(
            input_ids=block.unsqueeze(0).to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = output.hidden_states
        for layer in layers:
            collected[layer].append(
                hidden_states[layer].reshape(
                    -1, hidden_states[layer].shape[-1]
                ).float().cpu()
            )
        del output, hidden_states
    return {
        layer: torch.cat(values, dim=0)
        for layer, values in collected.items()
    }


def synthetic_rotation_check(device: torch.device) -> dict[str, float]:
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(512, 128, generator=generator).to(device)
    q, _ = torch.linalg.qr(
        torch.randn(128, 128, generator=generator).to(device)
    )
    rotated = x @ q
    anisotropic = rotated * torch.linspace(
        0.2, 2.0, 128, device=device
    )
    return {
        "cka_original_vs_orthogonal_rotation": linear_cka(x, rotated),
        "cka_original_vs_anisotropic_transform": linear_cka(x, anisotropic),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
        seq_len=int(data_cfg["seq_len"]),
        n_blocks=args.blocks,
    )
    print("[load] teacher", flush=True)
    teacher = load_model(
        cfg["teacher_model"], device, dtype, cache_dir=args.cache_dir
    )
    teacher_tied = (
        teacher.get_input_embeddings().weight.data_ptr()
        == teacher.get_output_embeddings().weight.data_ptr()
    )
    teacher_features = collect_layers(
        teacher, blocks, args.layers, device
    )
    selected_specs = {
        spec["id"]: spec for spec in cfg["models"]
        if spec["id"] in set(args.models)
    }
    audit: dict[str, Any] = {
        "purpose": (
            "Secondary hypothesis audit only; these internal metrics do not "
            "decide functional equivalence."
        ),
        "synthetic_invariance": synthetic_rotation_check(device),
        "teacher_tie_word_embeddings_config": bool(
            teacher.config.tie_word_embeddings
        ),
        "teacher_input_output_weights_share_storage": teacher_tied,
        "blocks": args.blocks,
        "tokens": int(args.blocks * int(data_cfg["seq_len"])),
        "layers": args.layers,
        "models": {},
    }
    rows_per_block = int(data_cfg["seq_len"])
    train_rows = (args.blocks // 2) * rows_per_block

    for model_id in args.models:
        spec = selected_specs[model_id]
        print(f"[model] {model_id}", flush=True)
        student = load_model(
            resolve_relative(spec["path"], config_dir), device, dtype
        )
        student_tied = (
            student.get_input_embeddings().weight.data_ptr()
            == student.get_output_embeddings().weight.data_ptr()
        )
        student_features = collect_layers(
            student, blocks, args.layers, device
        )
        layer_results = {}
        for layer in args.layers:
            teacher_x = teacher_features[layer].to(device)
            student_x = student_features[layer].to(device)
            print(f"  [layer] {layer}", flush=True)
            t_rank = effective_rank(teacher_x - teacher_x.mean(0, keepdim=True))
            s_rank = effective_rank(student_x - student_x.mean(0, keepdim=True))
            layer_results[str(layer)] = {
                "linear_cka": linear_cka(student_x, teacher_x),
                "teacher_effective_rank": t_rank,
                "student_effective_rank": s_rank,
                "effective_rank_ratio_student_teacher": s_rank / t_rank,
                "singular_spectrum_cosine": spectrum_cosine(
                    student_x, teacher_x
                ),
                "heldout_scaled_orthogonal_procrustes_r2": (
                    scaled_orthogonal_procrustes_r2(
                        student_x, teacher_x, train_rows
                    )
                ),
            }
            del teacher_x, student_x
        audit["models"][model_id] = {
            "objective": spec["objective"],
            "alpha": float(spec["alpha"]),
            "tie_word_embeddings_config": bool(
                student.config.tie_word_embeddings
            ),
            "input_output_weights_share_storage": student_tied,
            "layers": layer_results,
        }
        del student, student_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_json_dump(audit, Path(args.output).resolve())
    print(f"[complete] {args.output}", flush=True)


if __name__ == "__main__":
    main()
