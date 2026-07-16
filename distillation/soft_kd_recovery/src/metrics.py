"""
Evaluation metrics for the distillation-recovery experiment.

Three families, matching experiment.md:

  1. Perplexity            -> "did the model (re)learn the language?"
  2. KL(teacher || student) on a fixed probe -> behavioural distance to teacher
  3. Linear CKA per layer  -> representational similarity to teacher
                              ("did the student's internals return to the
                               teacher's?")

Why CKA on *representations*, not on weights:
the project's own metrics.md notes that comparing weight vectors directly via
L2/cosine is meaningless (architecture symmetries, permutation/scaling
freedom). CKA compares the *geometry of activations* and is invariant to
orthogonal transforms and isotropic scaling, which is exactly what we want when
asking "is the student computing the same thing as the teacher again?".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, blocks: torch.Tensor, device, batch_size: int = 8,
               dtype=torch.float32) -> float:
    """Token-level perplexity on packed blocks (next-token prediction)."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, blocks.size(0), batch_size):
        b = blocks[i:i + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(device.type != "cpu")):
            logits = model(b).logits
        # shift
        lg = logits[:, :-1, :].float()
        tgt = b[:, 1:]
        nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                              reduction="sum")
        total_nll += nll.item()
        total_tok += tgt.numel()
    return float(torch.exp(torch.tensor(total_nll / max(1, total_tok))))


@torch.no_grad()
def kl_to_teacher(student, teacher, blocks: torch.Tensor, device,
                  batch_size: int = 8, temperature: float = 1.0) -> dict:
    """Mean KL(teacher || student) and reverse KL over all next-token
    positions on a fixed probe set. Lower => student behaves like teacher."""
    student.eval(); teacher.eval()
    fkl_sum, rkl_sum, n = 0.0, 0.0, 0
    T = temperature
    for i in range(0, blocks.size(0), batch_size):
        b = blocks[i:i + batch_size].to(device)
        s = student(b).logits[:, :-1, :].float() / T
        t = teacher(b).logits[:, :-1, :].float() / T
        log_s = F.log_softmax(s, dim=-1)
        log_t = F.log_softmax(t, dim=-1)
        p_t = log_t.exp()
        p_s = log_s.exp()
        # KL(teacher||student) = sum p_t (log_t - log_s)
        fkl = (p_t * (log_t - log_s)).sum(-1)
        rkl = (p_s * (log_s - log_t)).sum(-1)
        fkl_sum += fkl.sum().item()
        rkl_sum += rkl.sum().item()
        n += fkl.numel()
    return {"kl_teacher_student": fkl_sum / max(1, n),
            "kl_student_teacher": rkl_sum / max(1, n)}


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between feature matrices X (n,d1), Y (n,d2).

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F), on column-centered X,Y.
    Uses the Gram-free (feature-space) form, which is cheap when d << n.
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    # Frobenius norm of cross-covariance, squared (HSIC numerator)
    cross = (Y.T @ X).pow(2).sum()
    xx = (X.T @ X).pow(2).sum().sqrt()
    yy = (Y.T @ Y).pow(2).sum().sqrt()
    denom = (xx * yy).clamp_min(1e-12)
    return float((cross / denom).item())


@torch.no_grad()
def layerwise_cka(student, teacher, blocks: torch.Tensor, device,
                  batch_size: int = 8, max_tokens: int = 8192) -> list[float]:
    """Per-layer linear CKA between student and teacher hidden states on a
    fixed probe. Returns one value per hidden-state layer (incl. embeddings).

    Hidden states are pooled across the (batch, seq) axis into a token x dim
    matrix; we subsample to `max_tokens` rows to bound the O(n d^2) cost.
    """
    student.eval(); teacher.eval()
    s_feats: list[list[torch.Tensor]] = None
    t_feats: list[list[torch.Tensor]] = None
    for i in range(0, blocks.size(0), batch_size):
        b = blocks[i:i + batch_size].to(device)
        s_h = student(b, output_hidden_states=True).hidden_states
        t_h = teacher(b, output_hidden_states=True).hidden_states
        if s_feats is None:
            s_feats = [[] for _ in s_h]
            t_feats = [[] for _ in t_h]
        for li, (sh, th) in enumerate(zip(s_h, t_h)):
            s_feats[li].append(sh.reshape(-1, sh.size(-1)).float().cpu())
            t_feats[li].append(th.reshape(-1, th.size(-1)).float().cpu())

    ckas = []
    g = torch.Generator().manual_seed(0)
    for li in range(len(s_feats)):
        X = torch.cat(s_feats[li], 0)
        Y = torch.cat(t_feats[li], 0)
        if X.size(0) > max_tokens:
            idx = torch.randperm(X.size(0), generator=g)[:max_tokens]
            X, Y = X[idx], Y[idx]
        ckas.append(_linear_cka(X, Y))
    return ckas
