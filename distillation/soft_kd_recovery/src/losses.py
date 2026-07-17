"""
Distillation losses for the recovery experiment.

Mixed objective from experiment.md:

    L = beta * D(student || teacher) + (1 - beta) * CE(student, ground_truth)

with `beta` annealed over training. `D` is a configurable divergence:

  * forward_kl  : KL(teacher || student)  -- mode-covering, the classic
                  Hinton KD target. Good when teacher-forcing on real data.
  * reverse_kl  : KL(student || teacher)  -- mode-seeking; this is what
                  on-policy distillation effectively optimises and is the
                  right choice when the student samples its own continuations
                  (Agarwal et al., GKD).
  * jsd         : generalised Jensen-Shannon with mixing `jsd_beta`; bounded,
                  symmetric-ish, numerically gentle. Default for on-policy.

Temperature `T` softens both distributions; the KD term is scaled by T^2 so its
gradient magnitude is comparable across temperatures (Hinton 2015).

Because teacher and student share architecture *and* tokenizer (the student is
a noised copy of the teacher), logits are perfectly aligned -- no vocab
projection, no token-id remapping. This makes the divergence exact rather than
an approximation, which is a real advantage of this experimental setup.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def synthetic_ce_loss(student_logits: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = -100):
    """Next-token CE on synthetic text without teacher logits."""
    vocab_size = student_logits.shape[-1]
    ce = F.cross_entropy(
        student_logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    zero = torch.zeros((), device=student_logits.device)
    return ce, {"loss": ce.detach(), "kd": zero, "ce": ce.detach()}


def kd_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  kind: str = "forward_kl", temperature: float = 1.0,
                  jsd_beta: float = 0.5) -> torch.Tensor:
    """Per-position divergence, returned with shape (...,) (vocab summed out).

    Inputs are raw logits of identical shape (..., V). Teacher is treated as a
    constant (no grad should flow into it -- caller detaches/uses no_grad)."""
    T = temperature
    s = student_logits / T
    t = teacher_logits / T
    log_s = F.log_softmax(s, dim=-1)
    log_t = F.log_softmax(t, dim=-1)

    if kind == "forward_kl":
        # KL(teacher || student) = sum p_t (log p_t - log p_s)
        p_t = log_t.exp()
        div = (p_t * (log_t - log_s)).sum(-1)
    elif kind == "reverse_kl":
        # KL(student || teacher) = sum p_s (log p_s - log p_t)
        p_s = log_s.exp()
        div = (p_s * (log_s - log_t)).sum(-1)
    elif kind == "jsd":
        p_s = log_s.exp()
        p_t = log_t.exp()
        m = jsd_beta * p_t + (1.0 - jsd_beta) * p_s
        log_m = m.clamp_min(1e-12).log()
        kl_t = (p_t * (log_t - log_m)).sum(-1)
        kl_s = (p_s * (log_s - log_m)).sum(-1)
        div = jsd_beta * kl_t + (1.0 - jsd_beta) * kl_s
    else:
        raise ValueError(f"unknown divergence kind: {kind}")
    return div * (T * T)


def mixed_loss(student_logits, teacher_logits, targets, beta: float,
               divergence: str = "forward_kl", temperature: float = 1.0,
               jsd_beta: float = 0.5, ignore_index: int = -100,
               kd_mask: torch.Tensor | None = None):
    """Compute L = beta * KD + (1-beta) * CE.

    student_logits, teacher_logits: (B, S, V), already shifted to align with
        `targets` (i.e. logits at position i predict targets[i]).
    targets: (B, S) ground-truth ids (or ignore_index where unknown).
    kd_mask: optional (B, S) bool, positions where KD is applied (e.g. only
        student-generated tokens in on-policy mode). Defaults to all valid.

    Returns (loss, parts_dict).
    """
    B, S, V = student_logits.shape
    valid = (targets != ignore_index)

    # --- KD term ---
    div = kd_divergence(student_logits, teacher_logits, divergence,
                        temperature, jsd_beta)  # (B, S)
    if kd_mask is None:
        kd_mask = valid
    kd_denom = kd_mask.sum().clamp_min(1)
    kd = (div * kd_mask).sum() / kd_denom

    # --- CE term (only where we have ground-truth) ---
    if beta < 1.0 and valid.any():
        ce = F.cross_entropy(student_logits.reshape(-1, V),
                             targets.reshape(-1), ignore_index=ignore_index)
    else:
        ce = torch.zeros((), device=student_logits.device)

    loss = beta * kd + (1.0 - beta) * ce
    return loss, {"loss": loss.detach(), "kd": kd.detach(), "ce": ce.detach()}


def beta_schedule(step: int, total: int, beta_start: float, beta_end: float,
                  kind: str = "linear") -> float:
    """Anneal the KD weight beta from beta_start -> beta_end over `total` steps.

    experiment.md wants beta to *decrease*: start by snapping the noised student
    back onto the teacher (KD-heavy), then let ground-truth CE refine it."""
    if total <= 1:
        return beta_end
    frac = min(1.0, step / (total - 1))
    if kind == "cosine":
        import math
        w = 0.5 * (1 + math.cos(math.pi * frac))  # 1 -> 0
        return beta_end + (beta_start - beta_end) * w
    return beta_start + (beta_end - beta_start) * frac
