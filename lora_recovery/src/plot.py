"""
Plot recovery curves from one or more run logs.

    python -m src.plot results/diag_lr results/smoke_onpolicy results/smoke_mixed_warmup

Produces results/recovery_curves.png with PPL, KL-to-teacher and mean-CKA vs
training step, with the post-noise anchor (step 0) and the teacher floor drawn
for reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir: Path):
    evals, teacher, post = [], None, None
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") == "teacher_baseline":
            teacher = r
        elif r.get("phase") == "post_noise":
            post = r
            evals.append(r)
        elif r.get("phase", "").startswith("eval"):
            evals.append(r)
    return {"name": run_dir.name, "teacher": teacher, "post": post,
            "evals": evals}


def main():
    runs = [load_run(Path(p)) for p in sys.argv[1:]]
    if not runs:
        print("usage: python -m src.plot <run_dir> [run_dir ...]")
        return
    metrics = [("ppl", "Perplexity (log)", True),
               ("kl_teacher_student", "KL(teacher || student)", False),
               ("cka_mean", "Mean layer CKA to teacher", False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, title, logy) in zip(axes, metrics):
        for run in runs:
            xs = [e["step"] for e in run["evals"]]
            ys = [e[key] for e in run["evals"]]
            ax.plot(xs, ys, marker="o", label=run["name"])
        t = runs[0]["teacher"]
        if t and key in t:
            ax.axhline(t[key], ls="--", c="gray", lw=1,
                       label="teacher floor")
        ax.set_title(title)
        ax.set_xlabel("training step")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Distillation recovery after Gaussian weight perturbation")
    fig.tight_layout()
    out = Path("results") / "recovery_curves.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
