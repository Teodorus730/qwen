"""Directly compare soft-KD and hard-CE students at matching noise alpha."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from evaluate_outputs import (
    atomic_json_dump,
    baseline_block_metrics,
    load_model,
    load_text_blocks,
    resolve_relative,
    seed_everything,
    summarize_baseline,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--output", default=str(
        SCRIPT_DIR / "outputs" / "soft_vs_hard.json"
    ))
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = cfg["runtime"]
    seed = int(runtime["seed"])
    seed_everything(seed)
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_cfg = cfg["data"]
    blocks = load_text_blocks(
        tokenizer=tokenizer,
        jsonl_path=resolve_relative(data_cfg["jsonl"], config_dir),
        text_field=data_cfg["text_field"],
        score_field=data_cfg["score_field"],
        min_score=float(data_cfg["min_score"]),
        skip_docs=int(data_cfg["skip_docs"]),
        seq_len=int(data_cfg["seq_len"]),
        n_blocks=int(data_cfg["eval_blocks"]),
    )
    by_alpha: dict[float, dict[str, dict]] = {}
    for spec in cfg["models"]:
        by_alpha.setdefault(float(spec["alpha"]), {})[spec["objective"]] = spec

    output: dict[str, object] = {
        "definition": (
            "At each alpha, soft-KD is the reference distribution and "
            "hard-teacher-CE is the comparison distribution."
        ),
        "pairs": {},
    }
    metrics_cfg = cfg["metrics"]
    for pair_index, (alpha, specs) in enumerate(sorted(by_alpha.items())):
        soft_spec = specs["soft_kd"]
        hard_spec = specs["hard_teacher_ce"]
        print(f"[pair] alpha={alpha:g}", flush=True)
        soft_model = load_model(
            resolve_relative(soft_spec["path"], config_dir),
            device,
            dtype,
        )
        hard_model = load_model(
            resolve_relative(hard_spec["path"], config_dir),
            device,
            dtype,
        )
        records = []
        for block_index, block in enumerate(blocks):
            ids = block.unsqueeze(0).to(device)
            labels = ids[:, 1:]
            with torch.inference_mode():
                soft_logits = soft_model(ids).logits[:, :-1, :]
                hard_logits = hard_model(ids).logits[:, :-1, :]
            record, _, _, _ = baseline_block_metrics(
                soft_logits,
                hard_logits,
                labels,
                vocab_chunk=int(runtime["vocab_chunk"]),
                topk=int(metrics_cfg["topk"]),
                temperatures=[
                    float(value) for value in metrics_cfg["temperatures"]
                ],
            )
            records.append(record)
            del ids, labels, soft_logits, hard_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"  [blocks] {block_index + 1}/{len(blocks)}",
                end="\r",
                flush=True,
            )
        print("", flush=True)
        summary = summarize_baseline(
            records,
            seed=seed + pair_index,
            samples=int(runtime["bootstrap_samples"]),
        )
        output["pairs"][f"alpha_{alpha:g}"] = {
            "alpha": alpha,
            "soft_id": soft_spec["id"],
            "hard_id": hard_spec["id"],
            "summary": summary,
            "block_records": records,
        }
        print(
            f"  top1={summary['top1_match']['estimate']:.4f} "
            f"KL(soft||hard)={summary['kl_teacher_student_t1']['estimate']:.4f}",
            flush=True,
        )
        del soft_model, hard_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_json_dump(output, Path(args.output).resolve())


if __name__ == "__main__":
    main()
