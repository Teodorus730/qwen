"""Load a saved LoRA adapter into a newly noised base model and verify it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel

try:
    from . import data as data_mod
    from .data import DataConfig
    from .distill import RunConfig, get_device, load_models, run_eval
    from .noise import NoiseConfig, inject_noise
except ImportError:  # pragma: no cover
    import data as data_mod
    from data import DataConfig
    from distill import RunConfig, get_device, load_models, run_eval
    from noise import NoiseConfig, inject_noise


METRIC_KEYS = ("ppl", "kl_teacher_student", "kl_student_teacher", "cka_mean")


def build_eval_sets(tokenizer, cfg: RunConfig):
    dcfg = DataConfig(seq_len=cfg.seq_len, min_score=cfg.min_score,
                       seed=cfg.data_seed, local_jsonl=cfg.local_jsonl)
    if cfg.train_one_pass:
        if not cfg.eval_jsonl or not cfg.probe_jsonl:
            raise ValueError("saved one-pass run has no held-out paths")
        eval_blocks, _ = data_mod.fixed_local_blocks_with_source_ids(
            tokenizer, dcfg, cfg.eval_jsonl, cfg.eval_blocks,
            skip_docs=cfg.eval_skip_docs)
        probe_blocks, _ = data_mod.fixed_local_blocks_with_source_ids(
            tokenizer, dcfg, cfg.probe_jsonl, cfg.probe_blocks,
            skip_docs=cfg.probe_skip_docs)
        return eval_blocks, probe_blocks
    return (
        data_mod.fixed_blocks(tokenizer, dcfg, cfg.eval_blocks,
                              skip_docs=cfg.eval_skip_docs),
        data_mod.fixed_blocks(tokenizer, dcfg, cfg.probe_blocks,
                              skip_docs=cfg.probe_skip_docs),
    )


def assert_matches(expected: dict, actual: dict, atol: float, rtol: float):
    for key in METRIC_KEYS:
        if key not in expected:
            raise ValueError(f"run_metadata has no final metric {key}")
        if not math.isclose(float(expected[key]), float(actual[key]),
                            rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                f"{key}: loaded={actual[key]:.8g}, "
                f"post-training={expected[key]:.8g}, "
                f"atol={atol}, rtol={rtol}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    metadata_path = adapter_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = RunConfig(**metadata["run_config"])
    cfg.base_model_revision = metadata["base_model_revision"]
    if args.device is not None:
        cfg.device = args.device
    if cfg.lora_r <= 0:
        raise ValueError("adapter verification requires a LoRA run")

    device = get_device(cfg.device)
    tok, teacher, student = load_models(cfg, device)
    inject_noise(student, NoiseConfig(alpha=metadata["noise_alpha"],
                                      perturb_norms=cfg.perturb_norms,
                                      seed=metadata["noise_seed"]))
    eval_blocks, probe_blocks = build_eval_sets(tok, cfg)
    with torch.no_grad():
        reference_logits = student(eval_blocks[:1].to(device)).logits.float()
    student = PeftModel.from_pretrained(student, adapter_dir,
                                        is_trainable=False).eval()
    with torch.no_grad():
        adapted_logits = student(eval_blocks[:1].to(device)).logits.float()
    adapter_effect = float((adapted_logits - reference_logits).abs().max())
    if not math.isfinite(adapter_effect) or adapter_effect == 0.0:
        raise AssertionError("loaded adapter does not change the noised model")
    actual = run_eval(student, teacher, tok, cfg, device, eval_blocks,
                      probe_blocks)
    if not all(math.isfinite(float(actual[key])) for key in METRIC_KEYS):
        raise FloatingPointError("loaded adapter evaluation produced NaN/Inf")
    assert_matches(metadata["final_metrics"], actual, args.atol, args.rtol)
    report = {
        "adapter_dir": str(adapter_dir),
        "noise_alpha": metadata["noise_alpha"],
        "noise_seed": metadata["noise_seed"],
        "adapter_effect_max_abs_logit_delta": adapter_effect,
        "verified_metrics": {key: actual[key] for key in METRIC_KEYS},
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
