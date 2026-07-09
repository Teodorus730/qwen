"""
Live / rich recovery plots for the distillation-recovery experiment.

Unlike `src/plot.py` (a 3-panel static comparison), this renders a 5-panel
figure that is meaningful for a *single run watched live* as well as for an
alpha-sweep overlay:

  1. Perplexity vs step (log y)         -- "did it relearn the language?"
  2. KL(teacher||student) vs step       -- behavioural distance to teacher
  3. Mean layer CKA vs step             -- representational distance
  4. Per-layer CKA at the latest eval   -- *where* recovery happens / breaks
  5. Train loss / kd / ce vs step       -- optimisation health (single run only)

Each metric panel draws the teacher floor (dashed) and uses the post-noise
point (step 0) as the left anchor, so the recovery trajectory is readable.

Usage
-----
  # one-shot render of one or more runs
  python -m src.live_plot results/sweep_qwen3_0.6b/a0.2

  # overlay a whole sweep
  python -m src.live_plot results/sweep_qwen3_0.6b/a*

  # live: re-render every 20 s until killed
  python -m src.live_plot --watch 20 --out results/sweep_qwen3_0.6b/live.png \
      results/sweep_qwen3_0.6b/a0.2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir: Path):
    evals, trains, teacher, post = [], [], None, None
    log = run_dir / "log.jsonl"
    if not log.exists():
        return {"name": run_dir.name, "teacher": None, "post": None,
                "evals": [], "trains": []}
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:        # last line may be mid-write
            continue
        phase = r.get("phase", "")
        if phase == "teacher_baseline":
            teacher = r
        elif phase == "post_noise":
            post = r
            evals.append(r)
        elif phase.startswith("eval"):
            evals.append(r)
        elif phase == "train":
            trains.append(r)
    return {"name": run_dir.name, "teacher": teacher, "post": post,
            "evals": evals, "trains": trains}


def render(run_dirs, out_path: Path):
    runs = [load_run(Path(p)) for p in run_dirs]
    runs = [r for r in runs if r["evals"] or r["trains"]]
    if not runs:
        return False
    single = len(runs) == 1

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.6))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(runs) - 1)) for i in range(len(runs))]

    # ---- panels 1-3: metric vs step --------------------------------------
    metric_panels = [
        ("ppl", "Perplexity (log y)", True),
        ("kl_teacher_student", "KL(teacher || student)", False),
        ("cka_mean", "Mean layer CKA to teacher", False),
    ]
    for ax, (key, title, logy) in zip(axes[:3], metric_panels):
        for run, c in zip(runs, colors):
            ev = [e for e in run["evals"] if key in e]
            xs = [e["step"] for e in ev]
            ys = [e[key] for e in ev]
            if xs:
                ax.plot(xs, ys, marker="o", ms=4, color=c, label=run["name"])
        t = runs[0]["teacher"]
        if t and key in t:
            ax.axhline(t[key], ls="--", c="crimson", lw=1.2,
                       label="teacher floor")
        ax.set_title(title)
        ax.set_xlabel("training step")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    # ---- panel 4: per-layer CKA at the latest eval -----------------------
    ax = axes[3]
    for run, c in zip(runs, colors):
        ev = [e for e in run["evals"] if "cka_per_layer" in e]
        if not ev:
            continue
        last = ev[-1]
        cka = last["cka_per_layer"]
        ax.plot(range(len(cka)), cka, marker=".", color=c,
                label=f"{run['name']} @{last['step']}")
        post = run["post"]
        if post and "cka_per_layer" in post and single:
            ax.plot(range(len(post["cka_per_layer"])), post["cka_per_layer"],
                    ls=":", color="gray", label="post-noise")
    ax.axhline(1.0, ls="--", c="crimson", lw=1.0)
    ax.set_title("Per-layer CKA (latest eval)")
    ax.set_xlabel("hidden-state layer (0=embed)")
    ax.set_ylabel("linear CKA")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # ---- panel 5: train loss (single run) or PPL recovery % (sweep) ------
    ax = axes[4]
    if single and runs[0]["trains"]:
        tr = runs[0]["trains"]
        xs = [t["step"] for t in tr]
        for key, lab in [("loss", "loss"), ("kd", "KD"), ("ce", "CE")]:
            ys = [t.get(key) for t in tr]
            ax.plot(xs, ys, lw=1.2, label=lab)
        ax.set_title("Train loss / KD / CE")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    else:
        # sweep: final PPL vs alpha-ish (use run order), recovery summary
        names, post_ppl, last_ppl, last_cka = [], [], [], []
        for run in runs:
            if not run["post"] or not run["evals"]:
                continue
            names.append(run["name"])
            post_ppl.append(run["post"]["ppl"])
            last_ppl.append(run["evals"][-1]["ppl"])
            last_cka.append(run["evals"][-1].get("cka_mean"))
        x = range(len(names))
        ax.plot(x, post_ppl, marker="o", label="post-noise PPL")
        ax.plot(x, last_ppl, marker="s", label="final PPL")
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=45, fontsize=7)
        ax.set_title("Damage vs recovery per run")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Distillation recovery after Gaussian weight perturbation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--out", default="results/live.png")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="re-render every N seconds (0 = one-shot)")
    args = ap.parse_args()

    if args.watch <= 0:
        ok = render(args.runs, Path(args.out))
        print("wrote", args.out if ok else "(no data yet)", flush=True)
        return
    while True:
        ok = render(args.runs, Path(args.out))
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"{'wrote ' + args.out if ok else 'no data yet'}", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
