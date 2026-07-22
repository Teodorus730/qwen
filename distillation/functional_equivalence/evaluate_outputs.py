"""Output-level functional-equivalence evaluation for Qwen teacher/students.

The script intentionally keeps the primary verdict outside model internals:

* next-token distributions and top-k decisions on held-out text;
* overlap of errors against the actual next token;
* paired robustness under identical perturbations;
* greedy free-generation agreement and trajectory KL.

It evaluates one student at a time, writes an atomic partial result after every
checkpoint, and can resume an interrupted sweep with ``--resume``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional model ids to evaluate.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Use one eval/robustness block and two prompts.")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--no-hash", action="store_true")
    return parser.parse_args()


def resolve_relative(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_text_blocks(
    tokenizer,
    jsonl_path: Path,
    text_field: str,
    score_field: str,
    min_score: float,
    skip_docs: int,
    seq_len: int,
    n_blocks: int,
) -> torch.Tensor:
    """Pack a deterministic held-out suffix into fixed causal-LM blocks."""
    eos = tokenizer.eos_token_id
    buffer: list[int] = []
    blocks: list[torch.Tensor] = []
    accepted = 0
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if float(record.get(score_field, math.inf)) < min_score:
                continue
            if accepted < skip_docs:
                accepted += 1
                continue
            text = record.get(text_field)
            if not text:
                continue
            accepted += 1
            buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            buffer.append(eos)
            while len(buffer) >= seq_len:
                blocks.append(torch.tensor(buffer[:seq_len], dtype=torch.long))
                del buffer[:seq_len]
                if len(blocks) >= n_blocks:
                    return torch.stack(blocks)
    raise RuntimeError(
        f"Only built {len(blocks)}/{n_blocks} blocks from {jsonl_path}; "
        "reduce skip_docs or n_blocks."
    )


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


@torch.inference_mode()
def forward_ids(model, input_ids: torch.Tensor, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
    with autocast_context(device, dtype):
        return model(input_ids=input_ids).logits[:, :-1, :]


@dataclass(frozen=True)
class Perturbation:
    key: str
    family: str
    level: float


def build_perturbations(cfg: dict[str, Any]) -> list[Perturbation]:
    result: list[Perturbation] = []
    for sigma in cfg["embedding_noise_sigma"]:
        result.append(Perturbation(
            f"embedding_noise_sigma_{float(sigma):g}",
            "embedding_noise_sigma",
            float(sigma),
        ))
    for rate in cfg["token_dropout_rate"]:
        result.append(Perturbation(
            f"token_dropout_rate_{float(rate):g}",
            "token_dropout_rate",
            float(rate),
        ))
    for rate in cfg.get("token_replacement_rate", []):
        result.append(Perturbation(
            f"token_replacement_rate_{float(rate):g}",
            "token_replacement_rate",
            float(rate),
        ))
    return result


def perturbation_payload(
    input_ids: torch.Tensor,
    perturbation: Perturbation,
    hidden_size: int,
    eos_token_id: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create one shared perturbation payload for teacher and student."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if perturbation.family == "embedding_noise_sigma":
        noise = torch.randn(
            (*input_ids.shape, hidden_size),
            generator=generator,
            dtype=torch.float32,
        )
        return {"noise": noise}

    eligible = input_ids.cpu().ne(eos_token_id)
    random_values = torch.rand(input_ids.shape, generator=generator)
    mask = (random_values < perturbation.level) & eligible
    # Keep the first token intact so every causal prefix has a real anchor.
    mask[:, 0] = False
    if perturbation.family == "token_dropout_rate":
        return {"mask": mask}

    if perturbation.family == "token_replacement_rate":
        # Draw replacements from other positions in the same block. This is an
        # external token corruption that preserves the empirical unigram pool
        # and avoids sampling reserved tokenizer ids.
        offsets = torch.randint(
            1, input_ids.shape[1], input_ids.shape, generator=generator
        )
        positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
        source_positions = (positions + offsets) % input_ids.shape[1]
        replacements = input_ids.cpu().gather(1, source_positions)
        corrupted = input_ids.cpu().clone()
        corrupted[mask] = replacements[mask]
        return {"input_ids": corrupted}

    raise ValueError(f"Unknown perturbation family: {perturbation.family}")


@torch.inference_mode()
def forward_perturbed(
    model,
    input_ids: torch.Tensor,
    payload: dict[str, torch.Tensor],
    perturbation: Perturbation,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if perturbation.family == "token_replacement_rate":
        return forward_ids(model, payload["input_ids"].to(device), device, dtype)

    embeddings = model.get_input_embeddings()(input_ids)
    if perturbation.family == "embedding_noise_sigma":
        embeddings = embeddings + (
            payload["noise"].to(device=device, dtype=embeddings.dtype)
            * perturbation.level
        )
    elif perturbation.family == "token_dropout_rate":
        embeddings = embeddings.masked_fill(
            payload["mask"].to(device).unsqueeze(-1), 0
        )
    else:
        raise ValueError(perturbation.family)
    with autocast_context(device, dtype):
        return model(inputs_embeds=embeddings).logits[:, :-1, :]


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _token_chunks(n_tokens: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, n_tokens, chunk_size):
        yield slice(start, min(n_tokens, start + chunk_size))


@torch.inference_mode()
def baseline_block_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_chunk: int,
    topk: int,
    temperatures: list[float],
) -> tuple[dict[str, float], dict[str, list[float]], torch.Tensor, torch.Tensor]:
    """Compute all clean metrics for one equal-length block."""
    teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    student_logits = student_logits.reshape(-1, student_logits.shape[-1])
    labels = labels.reshape(-1)
    sums: dict[str, float] = {
        "top1_match": 0.0,
        "teacher_correct": 0.0,
        "student_correct": 0.0,
        "error_intersection": 0.0,
        "error_union": 0.0,
        "teacher_error": 0.0,
        "student_error": 0.0,
        "same_wrong_prediction": 0.0,
        "both_correct": 0.0,
        "teacher_only_correct": 0.0,
        "student_only_correct": 0.0,
        "both_wrong": 0.0,
        "teacher_nll": 0.0,
        "student_nll": 0.0,
        "kl_teacher_student_t1": 0.0,
        "kl_student_teacher_t1": 0.0,
        "kl_teacher_student_t2": 0.0,
        "kl_student_teacher_t2": 0.0,
        "js_divergence_t1": 0.0,
        "total_variation_t1": 0.0,
        "teacher_entropy": 0.0,
        "student_entropy": 0.0,
        "teacher_confidence": 0.0,
        "student_confidence": 0.0,
        "logit_pearson": 0.0,
        "topk_overlap": 0.0,
    }
    scatter: dict[str, list[float]] = {
        "teacher_confidence_on_teacher_errors": [],
        "student_confidence_on_teacher_errors": [],
    }
    teacher_top1_parts: list[torch.Tensor] = []
    student_top1_parts: list[torch.Tensor] = []
    n_tokens = labels.numel()

    for slc in _token_chunks(n_tokens, vocab_chunk):
        t = teacher_logits[slc].float()
        s = student_logits[slc].float()
        y = labels[slc]
        log_t = F.log_softmax(t, dim=-1)
        log_s = F.log_softmax(s, dim=-1)
        p_t = log_t.exp()
        p_s = log_s.exp()
        t_conf, t_top1 = p_t.max(dim=-1)
        s_conf, s_top1 = p_s.max(dim=-1)
        teacher_top1_parts.append(t_top1.cpu())
        student_top1_parts.append(s_top1.cpu())

        t_correct = t_top1.eq(y)
        s_correct = s_top1.eq(y)
        t_error = ~t_correct
        s_error = ~s_correct

        sums["top1_match"] += t_top1.eq(s_top1).sum().item()
        sums["teacher_correct"] += t_correct.sum().item()
        sums["student_correct"] += s_correct.sum().item()
        sums["error_intersection"] += (t_error & s_error).sum().item()
        sums["error_union"] += (t_error | s_error).sum().item()
        sums["teacher_error"] += t_error.sum().item()
        sums["student_error"] += s_error.sum().item()
        sums["same_wrong_prediction"] += (
            t_error & t_top1.eq(s_top1)
        ).sum().item()
        sums["both_correct"] += (t_correct & s_correct).sum().item()
        sums["teacher_only_correct"] += (t_correct & s_error).sum().item()
        sums["student_only_correct"] += (t_error & s_correct).sum().item()
        sums["both_wrong"] += (t_error & s_error).sum().item()
        sums["teacher_nll"] += (-log_t.gather(1, y[:, None])).sum().item()
        sums["student_nll"] += (-log_s.gather(1, y[:, None])).sum().item()
        sums["kl_teacher_student_t1"] += (
            p_t * (log_t - log_s)
        ).sum(-1).sum().item()
        sums["kl_student_teacher_t1"] += (
            p_s * (log_s - log_t)
        ).sum(-1).sum().item()
        log_m = torch.logaddexp(log_t, log_s) - math.log(2.0)
        sums["js_divergence_t1"] += (
            0.5 * (p_t * (log_t - log_m)).sum(-1)
            + 0.5 * (p_s * (log_s - log_m)).sum(-1)
        ).sum().item()
        sums["total_variation_t1"] += (
            0.5 * (p_t - p_s).abs().sum(-1)
        ).sum().item()
        sums["teacher_entropy"] += (-(p_t * log_t).sum(-1)).sum().item()
        sums["student_entropy"] += (-(p_s * log_s).sum(-1)).sum().item()
        sums["teacher_confidence"] += t_conf.sum().item()
        sums["student_confidence"] += s_conf.sum().item()

        # Pearson over the full vocabulary. It is kept because it was in the
        # supplied plan, but top-k overlap and distribution distances are less
        # dominated by the long tail.
        t_centered = t - t.mean(-1, keepdim=True)
        s_centered = s - s.mean(-1, keepdim=True)
        numerator = (t_centered * s_centered).sum(-1)
        denominator = (
            t_centered.square().sum(-1).sqrt()
            * s_centered.square().sum(-1).sqrt()
        ).clamp_min(1e-12)
        sums["logit_pearson"] += (numerator / denominator).sum().item()

        t_topk = t.topk(topk, dim=-1).indices
        s_topk = s.topk(topk, dim=-1).indices
        overlap = (
            t_topk.unsqueeze(-1) == s_topk.unsqueeze(-2)
        ).any(-1).sum(-1).float() / topk
        sums["topk_overlap"] += overlap.sum().item()

        if t_error.any():
            scatter["teacher_confidence_on_teacher_errors"].extend(
                t_conf[t_error].float().cpu().tolist()
            )
            scatter["student_confidence_on_teacher_errors"].extend(
                s_conf[t_error].float().cpu().tolist()
            )

        if any(abs(float(temp) - 2.0) < 1e-9 for temp in temperatures):
            log_t2 = F.log_softmax(t / 2.0, dim=-1)
            log_s2 = F.log_softmax(s / 2.0, dim=-1)
            p_t2 = log_t2.exp()
            p_s2 = log_s2.exp()
            sums["kl_teacher_student_t2"] += (
                p_t2 * (log_t2 - log_s2)
            ).sum(-1).sum().item()
            sums["kl_student_teacher_t2"] += (
                p_s2 * (log_s2 - log_t2)
            ).sum(-1).sum().item()

    per_token_keys = [
        key for key in sums
        if key not in {
            "error_intersection",
            "error_union",
            "teacher_error",
            "student_error",
            "same_wrong_prediction",
        }
    ]
    result = {key: sums[key] / n_tokens for key in per_token_keys}
    result["error_intersection"] = sums["error_intersection"]
    result["error_union"] = sums["error_union"]
    result["teacher_error"] = sums["teacher_error"]
    result["student_error"] = sums["student_error"]
    result["same_wrong_prediction"] = sums["same_wrong_prediction"]
    result["n_tokens"] = float(n_tokens)
    return (
        result,
        scatter,
        torch.cat(teacher_top1_parts),
        torch.cat(student_top1_parts),
    )


@torch.inference_mode()
def robustness_block_metrics(
    clean_teacher_logits: torch.Tensor,
    clean_student_logits: torch.Tensor,
    perturbed_teacher_logits: torch.Tensor,
    perturbed_student_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_chunk: int,
) -> dict[str, float]:
    clean_t = clean_teacher_logits.reshape(-1, clean_teacher_logits.shape[-1])
    clean_s = clean_student_logits.reshape(-1, clean_student_logits.shape[-1])
    pert_t = perturbed_teacher_logits.reshape(
        -1, perturbed_teacher_logits.shape[-1]
    )
    pert_s = perturbed_student_logits.reshape(
        -1, perturbed_student_logits.shape[-1]
    )
    labels = labels.reshape(-1)
    sums = {
        "teacher_top1_stability": 0.0,
        "student_top1_stability": 0.0,
        "perturbed_top1_match": 0.0,
        "perturbed_kl_teacher_student": 0.0,
        "teacher_clean_to_perturbed_kl": 0.0,
        "student_clean_to_perturbed_kl": 0.0,
        "perturbed_teacher_nll": 0.0,
        "perturbed_student_nll": 0.0,
        "perturbed_teacher_correct": 0.0,
        "perturbed_student_correct": 0.0,
    }
    n_tokens = labels.numel()
    for slc in _token_chunks(n_tokens, vocab_chunk):
        ct = clean_t[slc].float()
        cs = clean_s[slc].float()
        pt = pert_t[slc].float()
        ps = pert_s[slc].float()
        y = labels[slc]

        log_ct = F.log_softmax(ct, dim=-1)
        log_cs = F.log_softmax(cs, dim=-1)
        log_pt = F.log_softmax(pt, dim=-1)
        log_ps = F.log_softmax(ps, dim=-1)
        p_ct = log_ct.exp()
        p_cs = log_cs.exp()
        p_pt = log_pt.exp()

        ct_top = ct.argmax(-1)
        cs_top = cs.argmax(-1)
        pt_top = pt.argmax(-1)
        ps_top = ps.argmax(-1)
        sums["teacher_top1_stability"] += ct_top.eq(pt_top).sum().item()
        sums["student_top1_stability"] += cs_top.eq(ps_top).sum().item()
        sums["perturbed_top1_match"] += pt_top.eq(ps_top).sum().item()
        sums["perturbed_teacher_correct"] += pt_top.eq(y).sum().item()
        sums["perturbed_student_correct"] += ps_top.eq(y).sum().item()
        sums["perturbed_teacher_nll"] += (
            -log_pt.gather(1, y[:, None])
        ).sum().item()
        sums["perturbed_student_nll"] += (
            -log_ps.gather(1, y[:, None])
        ).sum().item()
        sums["perturbed_kl_teacher_student"] += (
            p_pt * (log_pt - log_ps)
        ).sum(-1).sum().item()
        sums["teacher_clean_to_perturbed_kl"] += (
            p_ct * (log_ct - log_pt)
        ).sum(-1).sum().item()
        sums["student_clean_to_perturbed_kl"] += (
            p_cs * (log_cs - log_ps)
        ).sum(-1).sum().item()
    return {
        key: value / n_tokens for key, value in sums.items()
    } | {"n_tokens": float(n_tokens)}


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> list[float]:
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def bootstrap_mean(
    values: list[float], rng: np.random.Generator, samples: int
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    estimate = float(array.mean())
    if len(array) == 1:
        return {"estimate": estimate, "ci95": [estimate, estimate]}
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    draws = array[indices].mean(axis=1)
    return {"estimate": estimate, "ci95": percentile_ci(draws)}


def bootstrap_ratio(
    numerators: list[float],
    denominators: list[float],
    rng: np.random.Generator,
    samples: int,
) -> dict[str, Any]:
    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    estimate = float(numerator.sum() / max(denominator.sum(), 1.0))
    if len(numerator) == 1:
        return {"estimate": estimate, "ci95": [estimate, estimate]}
    indices = rng.integers(0, len(numerator), size=(samples, len(numerator)))
    n_draw = numerator[indices].sum(axis=1)
    d_draw = denominator[indices].sum(axis=1)
    draws = n_draw / np.maximum(d_draw, 1.0)
    return {"estimate": estimate, "ci95": percentile_ci(draws)}


def summarize_baseline(
    records: list[dict[str, float]], seed: int, samples: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    keys = [
        key for key in records[0]
        if key not in {
            "error_intersection",
            "error_union",
            "teacher_error",
            "student_error",
            "same_wrong_prediction",
            "n_tokens",
        }
    ]
    summary = {
        key: bootstrap_mean([record[key] for record in records], rng, samples)
        for key in keys
    }
    summary["error_jaccard"] = bootstrap_ratio(
        [record["error_intersection"] for record in records],
        [record["error_union"] for record in records],
        rng,
        samples,
    )
    summary["same_wrong_prediction_given_teacher_error"] = bootstrap_ratio(
        [record["same_wrong_prediction"] for record in records],
        [record["teacher_error"] for record in records],
        rng,
        samples,
    )
    summary["teacher_error_rate"] = bootstrap_ratio(
        [record["teacher_error"] for record in records],
        [record["n_tokens"] for record in records],
        rng,
        samples,
    )
    summary["student_error_rate"] = bootstrap_ratio(
        [record["student_error"] for record in records],
        [record["n_tokens"] for record in records],
        rng,
        samples,
    )
    summary["tokens"] = int(sum(record["n_tokens"] for record in records))
    summary["blocks"] = len(records)
    summary["teacher_ppl"] = math.exp(summary["teacher_nll"]["estimate"])
    summary["student_ppl"] = math.exp(summary["student_nll"]["estimate"])
    return summary


def summarize_robustness(
    records: list[dict[str, float]],
    baseline_records: list[dict[str, float]],
    seed: int,
    samples: int,
    equivalence_band: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    keys = [key for key in records[0] if key != "n_tokens"]
    summary = {
        key: bootstrap_mean([record[key] for record in records], rng, samples)
        for key in keys
    }
    stability_gap = [
        record["student_top1_stability"]
        - record["teacher_top1_stability"]
        for record in records
    ]
    summary["stability_gap_student_minus_teacher"] = bootstrap_mean(
        stability_gap, rng, samples
    )
    teacher_nll_increase = [
        record["perturbed_teacher_nll"] - clean["teacher_nll"]
        for record, clean in zip(records, baseline_records)
    ]
    student_nll_increase = [
        record["perturbed_student_nll"] - clean["student_nll"]
        for record, clean in zip(records, baseline_records)
    ]
    summary["teacher_nll_increase"] = bootstrap_mean(
        teacher_nll_increase, rng, samples
    )
    summary["student_nll_increase"] = bootstrap_mean(
        student_nll_increase, rng, samples
    )
    degradation_gap = [
        student - teacher
        for student, teacher in zip(student_nll_increase, teacher_nll_increase)
    ]
    summary["nll_degradation_gap_student_minus_teacher"] = bootstrap_mean(
        degradation_gap, rng, samples
    )
    ci = summary["stability_gap_student_minus_teacher"]["ci95"]
    summary["equivalent_within_top1_band"] = bool(
        ci[0] >= -equivalence_band and ci[1] <= equivalence_band
    )
    summary["tokens"] = int(sum(record["n_tokens"] for record in records))
    summary["blocks"] = len(records)
    return summary


def levenshtein_distance(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (token_a != token_b),
            ))
        previous = current
    return previous[-1]


@torch.inference_mode()
def generate_fixed(model, prompts: torch.Tensor, new_tokens: int,
                   batch_size: int, device: torch.device) -> torch.Tensor:
    generated: list[torch.Tensor] = []
    for start in range(0, prompts.shape[0], batch_size):
        batch = prompts[start:start + batch_size].to(device)
        output = model.generate(
            batch,
            attention_mask=torch.ones_like(batch),
            do_sample=False,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            use_cache=True,
            pad_token_id=model.config.eos_token_id,
        )
        generated.append(output[:, -new_tokens:].cpu())
    return torch.cat(generated)


@torch.inference_mode()
def trajectory_kl(
    teacher,
    student,
    sequences: torch.Tensor,
    prompt_len: int,
    new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    vocab_chunk: int,
) -> list[float]:
    results: list[float] = []
    for sequence in sequences:
        ids = sequence.unsqueeze(0).to(device)
        teacher_logits = forward_ids(teacher, ids, device, dtype)
        student_logits = forward_ids(student, ids, device, dtype)
        start = prompt_len - 1
        stop = start + new_tokens
        t = teacher_logits[:, start:stop].reshape(
            -1, teacher_logits.shape[-1]
        )
        s = student_logits[:, start:stop].reshape(
            -1, student_logits.shape[-1]
        )
        total = 0.0
        for slc in _token_chunks(t.shape[0], vocab_chunk):
            log_t = F.log_softmax(t[slc].float(), dim=-1)
            log_s = F.log_softmax(s[slc].float(), dim=-1)
            total += (
                log_t.exp() * (log_t - log_s)
            ).sum(-1).sum().item()
        results.append(total / new_tokens)
        del teacher_logits, student_logits
    return results


def summarize_generation(
    teacher_tokens: torch.Tensor,
    student_tokens: torch.Tensor,
    teacher_path_kl: list[float],
    student_path_kl: list[float],
    seed: int,
    samples: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    records: list[dict[str, float]] = []
    for teacher_row, student_row, t_kl, s_kl in zip(
        teacher_tokens.tolist(),
        student_tokens.tolist(),
        teacher_path_kl,
        student_path_kl,
    ):
        common_prefix = 0
        for t_token, s_token in zip(teacher_row, student_row):
            if t_token != s_token:
                break
            common_prefix += 1
        distance = levenshtein_distance(teacher_row, student_row)
        records.append({
            "exact_sequence_match": float(teacher_row == student_row),
            "token_match": float(np.mean(
                np.asarray(teacher_row) == np.asarray(student_row)
            )),
            "common_prefix_tokens": float(common_prefix),
            "normalized_edit_similarity": float(
                1.0 - distance / max(len(teacher_row), len(student_row), 1)
            ),
            "teacher_path_kl": float(t_kl),
            "student_path_kl": float(s_kl),
        })
    rng = np.random.default_rng(seed)
    summary = {
        key: bootstrap_mean([record[key] for record in records], rng, samples)
        for key in records[0]
    }
    summary["prompts"] = len(records)
    return summary, records


def model_verdict(
    baseline: dict[str, Any],
    robustness: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    top1 = baseline["top1_match"]
    kl = baseline["kl_teacher_student_t1"]
    top1_pass = top1["ci95"][0] >= thresholds["top1_match_min"]
    kl_pass = kl["ci95"][1] <= thresholds["kl_teacher_student_max"]
    robustness_pass = all(
        item["equivalent_within_top1_band"] for item in robustness.values()
    )
    return {
        "top1_threshold_pass": top1_pass,
        "kl_threshold_pass": kl_pass,
        "all_robustness_equivalence_bands_pass": robustness_pass,
        "qwen_project_criteria_pass": bool(
            top1_pass and kl_pass and robustness_pass
        ),
        "scope": (
            "Empirical equivalence on this held-out dataset and perturbation "
            "suite; not a proof for all possible inputs."
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "raw_results.partial.json"
    final_path = output_dir / "raw_results.json"

    runtime = cfg["runtime"]
    seed = int(runtime["seed"])
    seed_everything(seed)
    device_name = runtime.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)
    dtype_name = runtime.get("dtype", "bfloat16")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    if device.type == "cuda":
        fraction = float(runtime.get("cuda_memory_fraction", 1.0))
        torch.cuda.set_per_process_memory_fraction(fraction)

    cache_dir = args.cache_dir
    print(f"[load] tokenizer: {cfg['teacher_model']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["teacher_model"], cache_dir=cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = cfg["data"]
    n_eval_blocks = int(data_cfg["eval_blocks"])
    n_robustness_blocks = int(data_cfg["robustness_blocks"])
    generation_cfg = cfg["generation"]
    n_prompts = int(generation_cfg["prompts"])
    if args.smoke:
        n_eval_blocks = 1
        n_robustness_blocks = 1
        n_prompts = 2
    n_blocks_needed = max(n_eval_blocks, n_robustness_blocks, n_prompts)
    data_path = resolve_relative(data_cfg["jsonl"], config_dir)
    print(f"[data] held-out blocks from {data_path}", flush=True)
    blocks = load_text_blocks(
        tokenizer=tokenizer,
        jsonl_path=data_path,
        text_field=data_cfg["text_field"],
        score_field=data_cfg["score_field"],
        min_score=float(data_cfg["min_score"]),
        skip_docs=int(data_cfg["skip_docs"]),
        seq_len=int(data_cfg["seq_len"]),
        n_blocks=n_blocks_needed,
    )
    eval_blocks = blocks[:n_eval_blocks]
    robustness_blocks = blocks[:n_robustness_blocks]
    prompt_len = int(generation_cfg["prompt_len"])
    prompts = blocks[:n_prompts, :prompt_len].clone()

    print(f"[load] teacher on {device}", flush=True)
    teacher = load_model(
        cfg["teacher_model"], device, dtype, cache_dir=cache_dir
    )
    teacher_config = teacher.config.to_dict()
    hidden_size = int(teacher.config.hidden_size)

    teacher_generations = None
    if not args.skip_generation:
        print("[generation] teacher greedy continuations", flush=True)
        teacher_generations = generate_fixed(
            teacher,
            prompts,
            int(generation_cfg["new_tokens"]),
            int(generation_cfg["batch_size"]),
            device,
        )

    if args.resume and partial_path.exists():
        results = json.loads(partial_path.read_text(encoding="utf-8"))
    elif args.resume and final_path.exists():
        results = json.loads(final_path.read_text(encoding="utf-8"))
    else:
        results = {
            "meta": {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": cfg,
                "config_path": str(config_path),
                "data_path": str(data_path),
                "data_sha256": sha256_file(data_path) if not args.no_hash else None,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None
                ),
                "teacher_config": teacher_config,
                "eval_definition": {
                    "next_token_examples": int(
                        n_eval_blocks * (int(data_cfg["seq_len"]) - 1)
                    ),
                    "blocks": n_eval_blocks,
                    "held_out_skip_docs": int(data_cfg["skip_docs"]),
                },
                "smoke": bool(args.smoke),
            },
            "models": {},
        }

    selected = set(args.only or [])
    model_specs = [
        spec for spec in cfg["models"]
        if not selected or spec["id"] in selected
    ]
    perturbations = build_perturbations(cfg["robustness"])
    metrics_cfg = cfg["metrics"]
    thresholds = metrics_cfg["equivalence"]
    vocab_chunk = int(runtime["vocab_chunk"])
    bootstrap_samples = int(runtime["bootstrap_samples"])

    for model_index, spec in enumerate(model_specs):
        model_id = spec["id"]
        if args.resume and model_id in results["models"]:
            print(f"[resume] skip completed {model_id}", flush=True)
            continue
        model_path = resolve_relative(spec["path"], config_dir)
        print(
            f"\n[model {model_index + 1}/{len(model_specs)}] "
            f"{model_id}: {model_path}",
            flush=True,
        )
        started = time.time()
        student = load_model(model_path, device, dtype)
        weight_path = model_path / "model.safetensors"
        baseline_records: list[dict[str, float]] = []
        scatter = {
            "teacher_confidence_on_teacher_errors": [],
            "student_confidence_on_teacher_errors": [],
        }
        robustness_records: dict[str, list[dict[str, float]]] = {
            perturbation.key: [] for perturbation in perturbations
        }

        for block_index, block in enumerate(eval_blocks):
            input_ids = block.unsqueeze(0).to(device)
            labels = input_ids[:, 1:]
            clean_teacher = forward_ids(teacher, input_ids, device, dtype)
            clean_student = forward_ids(student, input_ids, device, dtype)
            record, block_scatter, _, _ = baseline_block_metrics(
                clean_teacher,
                clean_student,
                labels,
                vocab_chunk=vocab_chunk,
                topk=int(metrics_cfg["topk"]),
                temperatures=[float(x) for x in metrics_cfg["temperatures"]],
            )
            baseline_records.append(record)
            for key in scatter:
                scatter[key].extend(block_scatter[key])

            if block_index < n_robustness_blocks:
                for perturbation_index, perturbation in enumerate(perturbations):
                    perturb_seed = (
                        seed
                        + block_index * 10_000
                        + perturbation_index * 101
                    )
                    payload = perturbation_payload(
                        input_ids=input_ids,
                        perturbation=perturbation,
                        hidden_size=hidden_size,
                        eos_token_id=tokenizer.eos_token_id,
                        seed=perturb_seed,
                    )
                    perturbed_teacher = forward_perturbed(
                        teacher, input_ids, payload, perturbation, device, dtype
                    )
                    perturbed_student = forward_perturbed(
                        student, input_ids, payload, perturbation, device, dtype
                    )
                    robust_record = robustness_block_metrics(
                        clean_teacher,
                        clean_student,
                        perturbed_teacher,
                        perturbed_student,
                        labels,
                        vocab_chunk=vocab_chunk,
                    )
                    robustness_records[perturbation.key].append(robust_record)
                    del perturbed_teacher, perturbed_student, payload
            del clean_teacher, clean_student, input_ids, labels
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"  [blocks] {block_index + 1}/{n_eval_blocks}",
                end="\r",
                flush=True,
            )
        print("", flush=True)

        baseline_summary = summarize_baseline(
            baseline_records,
            seed=seed + model_index * 1000,
            samples=bootstrap_samples,
        )
        robustness_summary = {}
        for perturbation_index, perturbation in enumerate(perturbations):
            robustness_summary[perturbation.key] = {
                "family": perturbation.family,
                "level": perturbation.level,
                **summarize_robustness(
                    robustness_records[perturbation.key],
                    baseline_records[:n_robustness_blocks],
                    seed=seed + model_index * 1000 + perturbation_index + 1,
                    samples=bootstrap_samples,
                    equivalence_band=float(
                        thresholds["robustness_gap_abs_max"]
                    ),
                ),
            }

        generation_summary = None
        generation_records = None
        if not args.skip_generation:
            print("  [generation] student greedy continuations", flush=True)
            student_generations = generate_fixed(
                student,
                prompts,
                int(generation_cfg["new_tokens"]),
                int(generation_cfg["batch_size"]),
                device,
            )
            teacher_sequences = torch.cat(
                [prompts, teacher_generations], dim=1
            )
            student_sequences = torch.cat(
                [prompts, student_generations], dim=1
            )
            print("  [generation] trajectory KL", flush=True)
            teacher_path_kl = trajectory_kl(
                teacher,
                student,
                teacher_sequences,
                prompt_len,
                int(generation_cfg["new_tokens"]),
                device,
                dtype,
                vocab_chunk,
            )
            student_path_kl = trajectory_kl(
                teacher,
                student,
                student_sequences,
                prompt_len,
                int(generation_cfg["new_tokens"]),
                device,
                dtype,
                vocab_chunk,
            )
            generation_summary, generation_records = summarize_generation(
                teacher_generations,
                student_generations,
                teacher_path_kl,
                student_path_kl,
                seed=seed + model_index * 1000 + 500,
                samples=bootstrap_samples,
            )

        # Keep scatter payload bounded and deterministic for report figures.
        scatter_limit = 2000
        scatter = {
            key: values[:scatter_limit] for key, values in scatter.items()
        }
        results["models"][model_id] = {
            "id": model_id,
            "objective": spec["objective"],
            "alpha": float(spec["alpha"]),
            "path": str(model_path),
            "weights_sha256": (
                sha256_file(weight_path)
                if weight_path.exists() and not args.no_hash else None
            ),
            "baseline": baseline_summary,
            "baseline_block_records": baseline_records,
            "error_confidence_scatter": scatter,
            "robustness": robustness_summary,
            "robustness_block_records": robustness_records,
            "generation": generation_summary,
            "generation_prompt_records": generation_records,
            "verdict": model_verdict(
                baseline_summary, robustness_summary, thresholds
            ),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        results["meta"]["last_completed_model"] = model_id
        atomic_json_dump(results, partial_path)
        print(
            f"  [done] {model_id} in {time.time() - started:.1f}s | "
            f"top1={baseline_summary['top1_match']['estimate']:.4f} | "
            f"KL={baseline_summary['kl_teacher_student_t1']['estimate']:.4f}",
            flush=True,
        )
        del student
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["meta"]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    results["meta"]["complete"] = len(results["models"]) == len(model_specs)
    atomic_json_dump(results, final_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"\n[complete] {final_path}", flush=True)


if __name__ == "__main__":
    main()
