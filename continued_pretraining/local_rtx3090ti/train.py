from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from src.config import ExperimentConfig, load_config, resolve
from src.data import cyclic_batches, load_or_build_blocks, materialize_slice
from src.runtime import (
    build_optimizer,
    config_as_dict,
    configure_project_caches,
    environment_report,
    learning_rate_at,
    load_model,
    load_tokenizer,
    project_data_paths,
    set_seed,
    write_json,
)


def apply_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    training = cfg.training
    for field, value in (
        ("max_steps", args.max_steps),
        ("micro_batch_size", args.batch_size),
        ("gradient_accumulation_steps", args.grad_accum),
    ):
        if value is not None:
            training = replace(training, **{field: value})
    return replace(cfg, training=training)


def save_checkpoint(
    output_dir: Path,
    step: int,
    model,
    tokenizer,
    optimizer,
    cfg: ExperimentConfig,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    write_json(
        checkpoint / "trainer_state.json",
        {
            "completed_update_steps": step,
            "config": config_as_dict(cfg),
        },
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean causal-LM continued pretraining.")
    parser.add_argument("--config", default="configs/rtx3090ti.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--resume-from")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    configure_project_caches(cfg.root)
    set_seed(cfg.training.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This training recipe requires CUDA.")
    torch.cuda.set_per_process_memory_fraction(cfg.training.cuda_memory_fraction)
    device = torch.device("cuda")

    resume_dir = Path(args.resume_from).resolve() if args.resume_from else None
    initial_step = 0
    if resume_dir:
        state = json.loads(
            (resume_dir / "trainer_state.json").read_text(encoding="utf-8")
        )
        initial_step = int(state["completed_update_steps"])

    output_dir = resolve(cfg.root, cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config_as_dict(cfg))
    write_json(output_dir / "environment.json", environment_report())

    slice_path, blocks_path = project_data_paths(cfg)
    slice_report = materialize_slice(cfg.data, slice_path)
    print(f"[data] {slice_report}", flush=True)

    tokenizer = load_tokenizer(cfg, resume_dir)
    blocks, blocks_report = load_or_build_blocks(
        cfg.data, slice_path, tokenizer, blocks_path
    )
    print(f"[data] {blocks_report}", flush=True)

    model = load_model(cfg, device, resume_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[model] {cfg.model.id} params={parameter_count:,} "
        f"dtype={next(model.parameters()).dtype}",
        flush=True,
    )
    optimizer = build_optimizer(cfg, model)
    if resume_dir:
        optimizer.load_state_dict(
            torch.load(
                resume_dir / "optimizer.pt",
                map_location=device,
                weights_only=True,
            )
        )
        print(f"[resume] {resume_dir} at update {initial_step}", flush=True)

    batches = cyclic_batches(
        blocks,
        cfg.training.micro_batch_size,
        seed=cfg.training.seed,
    )
    skipped_micro_steps = (
        initial_step * cfg.training.gradient_accumulation_steps
    )
    for _ in range(skipped_micro_steps):
        next(batches)

    log_path = output_dir / "train_log.jsonl"
    log_mode = "a" if initial_step else "w"
    log_handle = log_path.open(log_mode, encoding="utf-8")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    total_tokens = 0

    for update_step in range(initial_step, cfg.training.max_steps):
        learning_rate = learning_rate_at(cfg, update_step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        loss_sum = 0.0
        for _ in range(cfg.training.gradient_accumulation_steps):
            batch = next(batches).to(device, non_blocking=True)
            output = model(input_ids=batch, labels=batch, use_cache=False)
            loss = output.loss / cfg.training.gradient_accumulation_steps
            loss.backward()
            loss_sum += float(loss.detach())
            total_tokens += batch.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.gradient_clip
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed = update_step + 1
        if completed % cfg.training.log_every == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            record = {
                "update_step": completed,
                "loss": loss_sum,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "tokens_this_run": total_tokens,
                "tokens_per_second": total_tokens / elapsed,
                "elapsed_seconds": elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated()
                / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved()
                / (1024**2),
            }
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(record, flush=True)

        if cfg.training.save_every and completed % cfg.training.save_every == 0:
            checkpoint = save_checkpoint(
                output_dir, completed, model, tokenizer, optimizer, cfg
            )
            print(f"[checkpoint] {checkpoint}", flush=True)

    if cfg.training.save_final:
        checkpoint = save_checkpoint(
            output_dir,
            cfg.training.max_steps,
            model,
            tokenizer,
            optimizer,
            cfg,
        )
        print(f"[final] {checkpoint}", flush=True)
    log_handle.close()
    print(f"[done] log={log_path}", flush=True)


if __name__ == "__main__":
    main()

