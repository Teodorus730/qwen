"""
Weight perturbation for the distillation-recovery experiment.

Core noise rule:

    W += alpha * std(W) * eps,   eps ~ N(0, 1)

We perturb a trained teacher's weights with Gaussian noise scaled by the
per-tensor standard deviation, producing a damaged "student". The research
question is then: can on-policy distillation from the *original* teacher pull
the student back to the teacher's behaviour, and how does recoverability depend
on alpha?

Design decisions:

* Noise is scaled by the *per-tensor* std of each weight matrix, so every
  layer is perturbed proportionally to its own scale. This keeps the relative
  signal-to-noise ratio constant across layers regardless of their magnitude.
* By default we perturb only 2-D weight matrices (attention / MLP / embedding
  projections). LayerNorm/RMSNorm gains and all biases are 1-D and extremely
  sensitive; perturbing them tends to destroy the model in a way that is more
  about numerical blow-up than about a meaningful "return to the teacher".
  Both behaviours are configurable.
* Tied weights (Qwen2.5 ties lm_head to embed_tokens) are perturbed exactly
  once: we deduplicate by the underlying storage pointer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class NoiseConfig:
    alpha: float = 0.05
    # Only perturb tensors whose ndim is >= this. 2 => weight matrices only.
    min_ndim: int = 2
    # If False, skip any parameter whose name matches `norm_pattern`.
    perturb_norms: bool = False
    # If False, skip biases (ndim == 1 handled by min_ndim, but explicit).
    perturb_bias: bool = False
    # Regex of parameter-name fragments to always skip.
    skip_pattern: str = r"(rotary_emb|inv_freq)"
    norm_pattern: str = r"(norm|ln_|layernorm)"
    seed: int = 0
    # Optional explicit include/exclude name regexes (override the heuristics).
    include_pattern: str | None = None
    exclude_pattern: str | None = None


@dataclass
class NoiseReport:
    alpha: float
    n_tensors_perturbed: int
    n_params_perturbed: int
    total_params: int
    per_tensor: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        frac = self.n_params_perturbed / max(1, self.total_params)
        return (
            f"alpha={self.alpha}: perturbed {self.n_tensors_perturbed} tensors "
            f"({self.n_params_perturbed:,} / {self.total_params:,} params, "
            f"{frac:5.1%})"
        )


def _should_perturb(name: str, p: torch.Tensor, cfg: NoiseConfig) -> bool:
    if cfg.include_pattern is not None:
        return re.search(cfg.include_pattern, name) is not None
    if cfg.exclude_pattern is not None and re.search(cfg.exclude_pattern, name):
        return False
    if re.search(cfg.skip_pattern, name):
        return False
    if not cfg.perturb_norms and re.search(cfg.norm_pattern, name):
        return False
    if not cfg.perturb_bias and name.endswith("bias"):
        return False
    if p.ndim < cfg.min_ndim:
        return False
    return True


@torch.no_grad()
def inject_noise(model: nn.Module, cfg: NoiseConfig) -> NoiseReport:
    """Perturb `model` in place: W += alpha * std(W) * eps. Returns a report.

    Idempotent w.r.t. tied weights: each storage is perturbed at most once.
    """
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    seen_storage: set[int] = set()
    report = NoiseReport(alpha=cfg.alpha, n_tensors_perturbed=0,
                         n_params_perturbed=0, total_params=0)

    for name, p in model.named_parameters():
        report.total_params += p.numel()
        if not _should_perturb(name, p, cfg):
            continue
        ptr = p.data_ptr()
        if ptr in seen_storage:  # tied weight, already perturbed
            continue
        seen_storage.add(ptr)

        std = p.detach().float().std().item()
        if std == 0.0 or cfg.alpha == 0.0:
            continue
        # Generate noise on CPU in fp32 for reproducibility, then cast/move.
        eps = torch.randn(p.shape, generator=gen, dtype=torch.float32)
        delta = (cfg.alpha * std) * eps
        p.add_(delta.to(p.dtype).to(p.device))

        report.n_tensors_perturbed += 1
        report.n_params_perturbed += p.numel()
        report.per_tensor.append(
            {"name": name, "shape": tuple(p.shape), "std": std,
             "noise_std": cfg.alpha * std}
        )
    return report


def perturbable_param_names(model: nn.Module, cfg: NoiseConfig) -> list[str]:
    """List names that *would* be perturbed (for inspection/debugging)."""
    out: list[str] = []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if _should_perturb(name, p, cfg):
            ptr = p.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            out.append(name)
    return out
