"""
Distillation-recovery experiment driver.

Pipeline
--------
1. Load a trained Qwen teacher (frozen, eval).
2. Clone it and inject Gaussian noise -> damaged student (noise.py).
3. Distil the student back toward the teacher on FineWeb-Edu using a mixed
   KD + CE loss with annealed beta (losses.py). Two regimes:
       off_policy : teacher-forced on real data blocks  (KD + CE)
       on_policy  : student generates 1..k tokens, teacher scores them,
                    KD on the student-generated region   (KD only)
   `on_policy_ratio` mixes them per step (the GKD recipe).
4. Track recovery: perplexity, KL-to-teacher, per-layer CKA (metrics.py),
   logged to results/<run>/log.jsonl plus a baseline snapshot taken right
   after noise injection (step 0) so the recovery curve has an anchor.

Runs on CPU, CUDA, or XPU. GPU runs use bf16 autocast.

Usage:
    python -m src.distill --config configs/smoke.yaml
    python -m src.distill --config configs/full_3090ti.yaml --alpha 0.1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# allow `python src/distill.py` as well as `python -m src.distill`
try:
    from . import data as data_mod
    from . import losses as loss_mod
    from . import metrics as metric_mod
    from .noise import NoiseConfig, inject_noise
except ImportError:  # pragma: no cover
    import data as data_mod
    import losses as loss_mod
    import metrics as metric_mod
    from noise import NoiseConfig, inject_noise


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    run_name: str = "smoke"
    out_dir: str = "results"
    device: str = "auto"              # auto | cuda | xpu | cpu

    # noise
    alpha: float = 0.05
    noise_seed: int = 0
    perturb_norms: bool = False

    # data
    seq_len: int = 512
    min_score: float = 0.0
    data_seed: int = 0
    local_jsonl: str | None = "data/fineweb_edu_local.jsonl"
    probe_skip_docs: int = 5000
    train_skip_docs: int = 200

    # distillation
    objective: str = "distillation"  # distillation | synthetic_ce
    mode: str = "mixed"               # off_policy | on_policy | mixed
    on_policy_ratio: float = 0.5      # P(step is on-policy) once warmed up
    on_policy_warmup_frac: float = 0.0  # first frac of steps forced off-policy
                                        # (cold-start fix: get the student back
                                        #  to "reasonable" before it rolls out)
    divergence: str = "forward_kl"    # forward_kl | reverse_kl | jsd
    on_policy_divergence: str = "reverse_kl"
    temperature: float = 1.0
    jsd_beta: float = 0.5
    beta_start: float = 1.0           # KD weight at step 0
    beta_end: float = 0.5             # KD weight at the end
    beta_schedule: str = "linear"
    rollout_min: int = 1              # on-policy: generate 1..k tokens
    rollout_max: int = 5
    rollout_prompt_len: int = 64
    sample_temperature: float = 1.0

    # optim
    optimizer: str = "adamw"          # adamw | adamw8bit (bnb, needs CUDA)
    grad_checkpointing: bool = False  # trade compute for memory (big models)
    lora_r: int = 0                   # 0 disables LoRA (existing full FT)
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"])
    steps: int = 50
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup: int = 5
    grad_clip: float = 1.0

    # eval
    eval_every: int = 25
    eval_blocks: int = 16
    probe_blocks: int = 8
    cka_max_tokens: int = 4096

    # housekeeping
    save_student: bool = False
    max_seconds: float = 0.0          # wall-clock cap (0 = none); for smoke


def load_config(path: str | None, overrides: dict) -> RunConfig:
    cfg = {}
    if path:
        cfg = yaml.safe_load(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    known = {f for f in RunConfig.__dataclass_fields__}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    result = RunConfig(**cfg)
    if result.objective not in {"distillation", "synthetic_ce"}:
        raise ValueError(
            "objective must be one of: distillation, synthetic_ce")
    if result.mode not in {"off_policy", "on_policy", "mixed"}:
        raise ValueError("mode must be one of: off_policy, on_policy, mixed")
    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.xpu.is_available():
            requested = "xpu"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU was requested but is not available")
    if requested not in {"cuda", "xpu", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, xpu, cpu")
    return torch.device(requested)


def model_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type in {"cuda", "xpu"} else torch.float32


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if device.type == "xpu":
        return round(torch.xpu.max_memory_allocated() / 2**20, 1)
    return 0.0


def load_models(cfg: RunConfig, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = model_dtype(device)
    teacher = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)
    if cfg.grad_checkpointing:
        # checkpointing requires use_cache=False during the training forward;
        # generation re-enables the cache explicitly (see on_policy_batch).
        student.gradient_checkpointing_enable()
        student.config.use_cache = False
    return tok, teacher, student


def add_lora(cfg: RunConfig, student):
    if cfg.lora_r <= 0:
        return student
    from peft import LoraConfig, TaskType, get_peft_model
    student = get_peft_model(
        student,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    student.print_trainable_parameters()
    return student


def build_optimizer(cfg: RunConfig, student):
    params = [p for p in student.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb  # CUDA-only; halves optimizer-state memory
        return bnb.optim.PagedAdamW8bit(params, lr=cfg.lr,
                                        weight_decay=cfg.weight_decay,
                                        betas=(0.9, 0.95))
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                             betas=(0.9, 0.95))


@torch.no_grad()
def on_policy_batch(student, teacher, prompt_block, cfg: RunConfig, device):
    """Student rolls out k tokens from a prompt; build full seq + a mask of the
    generated region. Returns (input_ids, kd_mask) on `device`. Generation is
    no_grad; the actual forward pass for the loss is done by the caller."""
    L = cfg.rollout_prompt_len
    prompt = prompt_block[:, :L].to(device)
    # pick k deterministically-ish per call via step-free randint
    k = int(torch.randint(cfg.rollout_min, cfg.rollout_max + 1, (1,)).item())
    gen = student.generate(
        prompt, do_sample=True, temperature=cfg.sample_temperature,
        top_k=0, top_p=1.0, max_new_tokens=k, min_new_tokens=k,
        pad_token_id=teacher.config.eos_token_id,
        use_cache=True,  # fast decoding even when grad-checkpointing is on
    )
    seq = gen  # (B, L+k)
    kd_mask = torch.zeros(seq.shape, dtype=torch.bool, device=device)
    # logits at position i predict token i+1; generated tokens are at indices
    # L..L+k-1, predicted by logits at L-1..L+k-2. We mark target positions.
    kd_mask[:, L:L + k] = True
    return seq, kd_mask


def run_eval(student, teacher, tok, cfg: RunConfig, device, eval_blocks,
             probe_blocks):
    dtype = model_dtype(device)
    ppl = metric_mod.perplexity(student, eval_blocks, device,
                                batch_size=cfg.batch_size, dtype=dtype)
    kl = metric_mod.kl_to_teacher(student, teacher, probe_blocks, device,
                                  batch_size=cfg.batch_size,
                                  temperature=cfg.temperature)
    cka = metric_mod.layerwise_cka(student, teacher, probe_blocks, device,
                                   batch_size=cfg.batch_size,
                                   max_tokens=cfg.cka_max_tokens)
    return {"ppl": ppl, **kl, "cka_mean": sum(cka) / len(cka),
            "cka_per_layer": cka}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--mode", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--max_seconds", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "alpha": args.alpha, "steps": args.steps, "mode": args.mode,
        "run_name": args.run_name, "max_seconds": args.max_seconds,
        "device": args.device,
    })
    device = get_device(cfg.device)
    # Safety cap (Windows/WDDM): without this, oversubscribing VRAM silently
    # spills into system RAM ("sysmem fallback") and the whole machine can hang
    # while it thrashes the pagefile. Capping the per-process fraction makes
    # PyTorch raise a clean OOM instead. Opt-in via env (default 1.0 = no-op, so
    # the 1.7B path is unchanged). Set e.g. CUDA_MEM_FRACTION=0.9 when launching.
    if device.type == "cuda":
        frac = float(os.environ.get("CUDA_MEM_FRACTION", "1.0"))
        if frac < 1.0:
            torch.cuda.set_per_process_memory_fraction(frac)
            print(f"[mem] capped CUDA to {frac:.0%} of device memory", flush=True)
    torch.manual_seed(cfg.noise_seed)

    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    logf = open(out / "log.jsonl", "w", encoding="utf-8")

    def log(rec: dict):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        keys = {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in rec.items() if k != "cka_per_layer"}
        print(keys, flush=True)

    print(f"[device] {device}  [model] {cfg.model_name}", flush=True)
    tok, teacher, student = load_models(cfg, device)

    # --- fixed eval / probe sets (reproducible, score-filtered) ---
    dcfg = data_mod.DataConfig(seq_len=cfg.seq_len, min_score=cfg.min_score,
                               seed=cfg.data_seed,
                               local_jsonl=cfg.local_jsonl)
    print("[data] building fixed eval/probe blocks ...", flush=True)
    eval_blocks = data_mod.fixed_blocks(tok, dcfg, cfg.eval_blocks, skip_docs=0)
    probe_blocks = data_mod.fixed_blocks(tok, dcfg, cfg.probe_blocks,
                                         skip_docs=cfg.probe_skip_docs)

    # --- baseline of the *clean* teacher-as-student (sanity, CKA==1) ---
    print("[eval] teacher self-baseline ...", flush=True)
    base = run_eval(teacher, teacher, tok, cfg, device, eval_blocks,
                    probe_blocks)
    log({"phase": "teacher_baseline", "step": -1, **base})

    # --- inject noise ---
    ncfg = NoiseConfig(alpha=cfg.alpha, perturb_norms=cfg.perturb_norms,
                       seed=cfg.noise_seed)
    report = inject_noise(student, ncfg)
    print("[noise]", report.summary(), flush=True)
    (out / "noise_report.json").write_text(
        json.dumps({"summary": report.summary(),
                    "n_tensors": report.n_tensors_perturbed,
                    "n_params": report.n_params_perturbed,
                    "total_params": report.total_params}, indent=2))

    # --- post-noise baseline (anchor of the recovery curve) ---
    print("[eval] post-noise (step 0) ...", flush=True)
    post = run_eval(student, teacher, tok, cfg, device, eval_blocks,
                    probe_blocks)
    log({"phase": "post_noise", "step": 0, **post})

    # Keep the noised base fixed and train only the adapter when LoRA is on.
    student = add_lora(cfg, student)

    # Count teacher forwards only while a train-step loss is being computed.
    # Evaluation forwards run with this flag disabled and are not included.
    teacher_train_forwards = 0
    teacher_train_step_active = False

    def count_teacher_train_forward(_module, _args):
        nonlocal teacher_train_forwards
        if teacher_train_step_active:
            teacher_train_forwards += 1

    teacher_forward_hook = teacher.register_forward_pre_hook(
        count_teacher_train_forward)

    # --- optimiser + LR schedule ---
    opt = build_optimizer(cfg, student)

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / max(1, cfg.warmup)
        frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        import math
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))

    # --- training stream ---
    train_iter = data_mod.make_batches(tok, dcfg, cfg.batch_size,
                                       n_batches=None,
                                       skip_docs=cfg.train_skip_docs)
    dtype = model_dtype(device)
    t0 = time.time()
    student.train()
    rng = torch.Generator().manual_seed(cfg.data_seed)

    step = 0
    opt.zero_grad(set_to_none=True)
    while step < cfg.steps:
        try:
            block = next(train_iter)
        except StopIteration:
            train_iter = data_mod.make_batches(tok, dcfg, cfg.batch_size,
                                               skip_docs=cfg.train_skip_docs)
            block = next(train_iter)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        beta = (0.0 if cfg.objective == "synthetic_ce" else
                loss_mod.beta_schedule(step, cfg.steps, cfg.beta_start,
                                       cfg.beta_end, cfg.beta_schedule))

        teacher_train_step_active = True
        try:
            if cfg.objective == "synthetic_ce":
                seq = block.to(device)
                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=(device.type != "cpu")):
                    s_logits = student(seq).logits[:, :-1, :]
                targets = seq[:, 1:]
                loss, parts = loss_mod.synthetic_ce_loss(
                    s_logits.float(), targets)
                regime = "synthetic_ce"
            else:
                # Keep cold-start protection for the existing KD regimes.
                warmup_steps = int(cfg.on_policy_warmup_frac * cfg.steps)
                on_policy_allowed = step >= warmup_steps
                use_on_policy = on_policy_allowed and (
                    cfg.mode == "on_policy" or
                    (cfg.mode == "mixed" and
                     torch.rand(1, generator=rng).item() <
                     cfg.on_policy_ratio))

                if use_on_policy:
                    seq, kd_mask = on_policy_batch(
                        student, teacher, block, cfg, device)
                    with torch.autocast(
                            device_type=device.type, dtype=dtype,
                            enabled=(device.type != "cpu")):
                        s_logits = student(seq).logits[:, :-1, :]
                        with torch.no_grad():
                            t_logits = teacher(seq).logits[:, :-1, :]
                    targets = torch.full_like(
                        seq[:, 1:], -100)  # no ground truth
                    kd_mask = kd_mask[:, 1:]
                    loss, parts = loss_mod.mixed_loss(
                        s_logits.float(), t_logits.float(), targets, beta=1.0,
                        divergence=cfg.on_policy_divergence,
                        temperature=cfg.temperature, jsd_beta=cfg.jsd_beta,
                        kd_mask=kd_mask)
                    regime = "on_policy"
                else:
                    seq = block.to(device)
                    with torch.autocast(
                            device_type=device.type, dtype=dtype,
                            enabled=(device.type != "cpu")):
                        s_logits = student(seq).logits[:, :-1, :]
                        with torch.no_grad():
                            t_logits = teacher(seq).logits[:, :-1, :]
                    targets = seq[:, 1:]
                    loss, parts = loss_mod.mixed_loss(
                        s_logits.float(), t_logits.float(), targets,
                        beta=beta, divergence=cfg.divergence,
                        temperature=cfg.temperature, jsd_beta=cfg.jsd_beta)
                    regime = "off_policy"
        finally:
            teacher_train_step_active = False

        (loss / cfg.grad_accum).backward()
        if (step + 1) % cfg.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm at step {step}: {grad_norm}")
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 5 == 0:
            log({"phase": "train", "step": step, "regime": regime,
                 "beta": beta, "lr": lr_at(step),
                 "loss": float(parts["loss"]), "kd": float(parts["kd"]),
                 "ce": float(parts["ce"]),
                 "teacher_train_forwards": teacher_train_forwards,
                 "elapsed": round(time.time() - t0, 1)})

        step += 1

        if step % cfg.eval_every == 0 or step == cfg.steps:
            student.eval()
            ev = run_eval(student, teacher, tok, cfg, device, eval_blocks,
                          probe_blocks)
            log({"phase": "eval", "step": step, **ev,
                 "peak_memory_mb": peak_memory_mb(device),
                 "elapsed": round(time.time() - t0, 1)})
            student.train()

        if cfg.max_seconds and (time.time() - t0) > cfg.max_seconds:
            print(f"[stop] wall-clock cap {cfg.max_seconds}s hit at step {step}",
                  flush=True)
            student.eval()
            ev = run_eval(student, teacher, tok, cfg, device, eval_blocks,
                          probe_blocks)
            log({"phase": "eval_final_capped", "step": step, **ev,
                 "peak_memory_mb": peak_memory_mb(device),
                 "elapsed": round(time.time() - t0, 1)})
            break

    if cfg.objective == "synthetic_ce" and teacher_train_forwards != 0:
        raise RuntimeError(
            "teacher forward was called inside a synthetic_ce train-step")
    print(f"[teacher-check] train forwards: {teacher_train_forwards}",
          flush=True)
    teacher_forward_hook.remove()

    if cfg.save_student:
        student.save_pretrained(out / "student")
        tok.save_pretrained(out / "student")

    logf.close()
    print(f"[done] run '{cfg.run_name}' -> {out}", flush=True)


if __name__ == "__main__":
    main()
