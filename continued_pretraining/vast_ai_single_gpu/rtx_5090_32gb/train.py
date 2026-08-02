from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import time
from dataclasses import replace
from pathlib import Path

import torch

from src.config import ExperimentConfig, load_config
from src.data import (
    cyclic_batches,
    fixed_batches,
    load_or_build_blocks,
    materialize_slice,
)
from src.runtime import (
    build_optimizer,
    config_as_dict,
    configure_project_environment,
    environment_report,
    latest_checkpoint,
    learning_rate_at,
    load_model,
    load_tokenizer,
    project_data_paths,
    run_output_dir,
    set_seed,
    write_json,
)

MIB = 1024**2


def apply_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    training = cfg.training
    for field, value in (
        ("max_tokens", args.max_tokens),
        ("micro_batch_size", args.batch_size),
        ("gradient_accumulation_steps", args.grad_accum),
        ("run_name", args.run_name),
    ):
        if value is not None:
            training = replace(training, **{field: value})
    return replace(cfg, training=training)


@torch.no_grad()
def evaluate(
    model,
    validation_blocks: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict:
    model.eval()
    weighted_loss = 0.0
    weight = 0
    started = time.perf_counter()
    for batch in fixed_batches(
        validation_blocks,
        cfg.training.micro_batch_size,
        max_blocks=cfg.training.eval_blocks,
    ):
        batch = batch.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        output = model(input_ids=batch, labels=batch, use_cache=False)
        batch_weight = int(batch.shape[0])
        weighted_loss += float(output.loss) * batch_weight
        weight += batch_weight
    torch.cuda.synchronize()
    mean_loss = weighted_loss / max(1, weight)
    model.train()
    return {
        "validation_loss": mean_loss,
        "validation_ppl": math.exp(min(20.0, mean_loss)),
        "validation_blocks": weight,
        "validation_tokens": weight * cfg.data.sequence_length,
        "evaluation_seconds": time.perf_counter() - started,
    }


def checkpoint_state(
    cfg: ExperimentConfig,
    *,
    completed_updates: int,
    completed_micro_steps: int,
    tokens_processed: int,
    best_validation_loss: float | None,
    data_metadata: dict,
) -> dict:
    return {
        "completed_update_steps": completed_updates,
        "completed_micro_steps": completed_micro_steps,
        "tokens_processed": tokens_processed,
        "best_validation_loss": best_validation_loss,
        "data_slice_sha256": data_metadata["slice_sha256"],
        "resume_compatibility": {
            "model_id": cfg.model.id,
            "model_revision": cfg.model.revision,
            "sequence_length": cfg.data.sequence_length,
            "micro_batch_size": cfg.training.micro_batch_size,
            "gradient_accumulation_steps": (
                cfg.training.gradient_accumulation_steps
            ),
            "optimizer": cfg.training.optimizer,
        },
    }


def assert_resume_compatible(
    cfg: ExperimentConfig,
    state: dict,
    data_metadata: dict,
) -> None:
    current = checkpoint_state(
        cfg,
        completed_updates=0,
        completed_micro_steps=0,
        tokens_processed=0,
        best_validation_loss=None,
        data_metadata=data_metadata,
    )
    if state["resume_compatibility"] != current["resume_compatibility"]:
        raise ValueError(
            "Checkpoint/config mismatch:\n"
            f"checkpoint={state['resume_compatibility']}\n"
            f"current={current['resume_compatibility']}"
        )
    if state["data_slice_sha256"] != data_metadata["slice_sha256"]:
        raise ValueError("Checkpoint was trained on a different data slice")


def rotate_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    )
    for old in checkpoints[:-keep]:
        shutil.rmtree(old)
        print(f"[checkpoint] removed old {old.name}", flush=True)


def save_checkpoint(
    output_dir: Path,
    model,
    tokenizer,
    optimizer,
    cfg: ExperimentConfig,
    state: dict,
) -> Path:
    step = int(state["completed_update_steps"])
    target = output_dir / f"checkpoint-{step:06d}"
    if target.exists() and (target / "trainer_state.json").exists():
        (output_dir / "latest_checkpoint.txt").write_text(
            target.name + "\n", encoding="utf-8"
        )
        return target
    if target.exists():
        shutil.rmtree(target)

    partial = output_dir / f".checkpoint-{step:06d}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    torch.cuda.synchronize()
    model.save_pretrained(
        partial,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(partial)
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    write_json(partial / "trainer_state.json", state)
    partial.replace(target)
    (output_dir / "latest_checkpoint.txt").write_text(
        target.name + "\n", encoding="utf-8"
    )
    rotate_checkpoints(output_dir, cfg.training.keep_last_checkpoints)
    print(f"[checkpoint] saved {target}", flush=True)
    return target


def resolve_resume(
    output_dir: Path,
    resume_from: str | None,
    *,
    fresh: bool,
) -> Path | None:
    if fresh:
        if (output_dir / "train_log.jsonl").exists():
            raise RuntimeError(
                f"{output_dir} already contains a run. Choose a new run_name "
                "instead of overwriting it."
            )
        return None
    if resume_from in (None, "auto"):
        return latest_checkpoint(output_dir)
    checkpoint = Path(resume_from).resolve()
    if not (checkpoint / "trainer_state.json").exists():
        raise FileNotFoundError(f"Not a complete checkpoint: {checkpoint}")
    return checkpoint


def append_jsonl(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable full-parameter continued pretraining on one GPU."
    )
    parser.add_argument("--config", default="configs/vast_5090_32gb.yaml")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume-from",
        default="auto",
        help="'auto' (default), a checkpoint path, or use --fresh",
    )
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    configure_project_environment(cfg.root)
    set_seed(cfg.training.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(
        cfg.training.cuda_memory_fraction
    )
    device = torch.device("cuda")

    output_dir = run_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = resolve_resume(
        output_dir,
        args.resume_from,
        fresh=args.fresh,
    )

    write_json(output_dir / "config.json", config_as_dict(cfg))
    write_json(output_dir / "environment.json", environment_report(cfg.root))

    slice_path, blocks_path = project_data_paths(cfg)
    slice_report = materialize_slice(cfg.data, slice_path)
    print(f"[data] {slice_report}", flush=True)
    tokenizer = load_tokenizer(cfg, resume_dir)
    train_blocks, validation_blocks, data_metadata = load_or_build_blocks(
        cfg.data,
        slice_path,
        tokenizer,
        blocks_path,
    )
    print(f"[data] {data_metadata}", flush=True)
    if data_metadata["train_tokens"] < cfg.training.max_tokens:
        print(
            "[warning] max_tokens exceeds unique packed train tokens; "
            "the iterator will enter a second shuffled epoch",
            flush=True,
        )

    model = load_model(cfg, device, resume_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[model] {cfg.model.id} params={parameter_count:,} "
        f"dtype={next(model.parameters()).dtype}",
        flush=True,
    )
    optimizer = build_optimizer(cfg, model)

    completed_updates = 0
    completed_micro_steps = 0
    tokens_processed = 0
    best_validation_loss: float | None = None
    if resume_dir:
        state = json.loads(
            (resume_dir / "trainer_state.json").read_text(encoding="utf-8")
        )
        assert_resume_compatible(cfg, state, data_metadata)
        optimizer.load_state_dict(
            torch.load(
                resume_dir / "optimizer.pt",
                map_location=device,
                weights_only=True,
            )
        )
        completed_updates = int(state["completed_update_steps"])
        completed_micro_steps = int(state["completed_micro_steps"])
        tokens_processed = int(state["tokens_processed"])
        best_validation_loss = state.get("best_validation_loss")
        print(
            f"[resume] {resume_dir.name}: updates={completed_updates}, "
            f"tokens={tokens_processed:,}",
            flush=True,
        )

    batches = cyclic_batches(
        train_blocks,
        cfg.training.micro_batch_size,
        seed=cfg.training.seed,
    )
    for _ in range(completed_micro_steps):
        next(batches)

    train_log = (output_dir / "train_log.jsonl").open(
        "a" if resume_dir else "w",
        encoding="utf-8",
    )
    eval_log = (output_dir / "eval_log.jsonl").open(
        "a" if resume_dir else "w",
        encoding="utf-8",
    )

    stop_requested = {"value": False, "signal": None}

    def request_stop(signum, _frame):
        stop_requested["value"] = True
        stop_requested["signal"] = int(signum)
        print(
            f"[signal] received {signum}; will checkpoint after current update",
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    phase = "resume_baseline" if resume_dir else "baseline"
    evaluation = evaluate(model, validation_blocks, cfg, device)
    best_validation_loss = min(
        best_validation_loss if best_validation_loss is not None else math.inf,
        evaluation["validation_loss"],
    )
    append_jsonl(
        eval_log,
        {
            "phase": phase,
            "update_step": completed_updates,
            "tokens_processed": tokens_processed,
            **evaluation,
        },
    )
    print(f"[eval] {phase}: {evaluation}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    session_started = time.perf_counter()
    session_tokens = 0
    last_checkpoint: Path | None = resume_dir

    while (
        completed_updates < cfg.total_updates
        and tokens_processed < cfg.training.max_tokens
    ):
        learning_rate = learning_rate_at(cfg, completed_updates)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        loss_sum = 0.0
        for _ in range(cfg.training.gradient_accumulation_steps):
            batch = next(batches).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            output = model(input_ids=batch, labels=batch, use_cache=False)
            loss = output.loss / cfg.training.gradient_accumulation_steps
            loss.backward()
            loss_sum += float(loss.detach())
            batch_tokens = int(batch.numel())
            tokens_processed += batch_tokens
            session_tokens += batch_tokens
            completed_micro_steps += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.training.gradient_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_updates += 1

        if completed_updates % cfg.training.log_every == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - session_started
            record = {
                "update_step": completed_updates,
                "total_updates": cfg.total_updates,
                "loss": loss_sum,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "tokens_processed": tokens_processed,
                "max_tokens": cfg.training.max_tokens,
                "session_tokens_per_second": session_tokens / elapsed,
                "session_elapsed_seconds": elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
            }
            append_jsonl(train_log, record)
            print(f"[train] {record}", flush=True)

        should_eval = (
            completed_updates % cfg.training.eval_every == 0
            or completed_updates >= cfg.total_updates
            or tokens_processed >= cfg.training.max_tokens
        )
        if should_eval:
            evaluation = evaluate(model, validation_blocks, cfg, device)
            best_validation_loss = min(
                best_validation_loss
                if best_validation_loss is not None
                else math.inf,
                evaluation["validation_loss"],
            )
            record = {
                "phase": "periodic",
                "update_step": completed_updates,
                "tokens_processed": tokens_processed,
                "best_validation_loss": best_validation_loss,
                **evaluation,
            }
            append_jsonl(eval_log, record)
            print(f"[eval] {record}", flush=True)

        should_save = (
            completed_updates % cfg.training.save_every == 0
            or stop_requested["value"]
            or completed_updates >= cfg.total_updates
            or tokens_processed >= cfg.training.max_tokens
        )
        if should_save:
            state = checkpoint_state(
                cfg,
                completed_updates=completed_updates,
                completed_micro_steps=completed_micro_steps,
                tokens_processed=tokens_processed,
                best_validation_loss=best_validation_loss,
                data_metadata=data_metadata,
            )
            last_checkpoint = save_checkpoint(
                output_dir,
                model,
                tokenizer,
                optimizer,
                cfg,
                state,
            )

        if stop_requested["value"]:
            print(
                f"[stop] checkpoint saved after signal "
                f"{stop_requested['signal']}",
                flush=True,
            )
            train_log.close()
            eval_log.close()
            return 130

    if cfg.training.save_final:
        state = checkpoint_state(
            cfg,
            completed_updates=completed_updates,
            completed_micro_steps=completed_micro_steps,
            tokens_processed=tokens_processed,
            best_validation_loss=best_validation_loss,
            data_metadata=data_metadata,
        )
        last_checkpoint = save_checkpoint(
            output_dir,
            model,
            tokenizer,
            optimizer,
            cfg,
            state,
        )

    summary = {
        "status": "complete",
        "model": cfg.model.id,
        "run_name": cfg.training.run_name,
        "completed_updates": completed_updates,
        "tokens_processed": tokens_processed,
        "target_tokens": cfg.training.max_tokens,
        "best_validation_loss": best_validation_loss,
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "session_seconds": time.perf_counter() - session_started,
    }
    write_json(output_dir / "summary.json", summary)
    train_log.close()
    eval_log.close()
    print(f"[done] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
