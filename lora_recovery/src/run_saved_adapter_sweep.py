"""Sequentially train, save and verify all eight LoRA recovery adapters."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


RUNS = (
    (0.00, "0p00"), (0.05, "0p05"), (0.10, "0p10"), (0.20, "0p20"),
    (0.25, "0p25"), (0.30, "0p30"), (0.35, "0p35"),
    (0.50, "0p50"),
)
REQUIRED_ADAPTER_FILES = {
    "adapter_model.safetensors", "adapter_config.json", "run_metadata.json",
}


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("run log contains NaN/Inf")


def audit_run(run_dir: Path, alpha: float, adapter_dir: Path):
    for name in ("config.json", "log.jsonl", "noise_report.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"missing {run_dir / name}")
    records = read_log(run_dir / "log.jsonl")
    for record in records:
        assert_finite(record)
    phases = {record.get("phase"): record for record in records}
    audit = phases.get("data_audit")
    complete = phases.get("data_complete")
    final = phases.get("eval") or phases.get("eval_final_capped")
    if not audit or not complete or not final:
        raise RuntimeError("run log is missing audit, completion, or final eval")
    if (audit["source_rows"], audit["unique_source_ids"],
            audit["packed_batches"], audit["optimizer_steps_planned"]) != (1500, 1500, 374, 374):
        raise RuntimeError(f"unexpected one-pass audit: {audit}")
    if (complete["packed_batches_consumed"], complete["optimizer_steps_completed"],
            complete["teacher_train_forwards"]) != (374, 374, 0):
        raise RuntimeError(f"incomplete or non-autonomous run: {complete}")
    if any(complete["source_id_intersections"].values()):
        raise RuntimeError(f"source ID overlap: {complete['source_id_intersections']}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if (config["alpha"], config["noise_seed"], config["training_seed"],
            config["objective"], config["lora_r"], config["save_student"],
            config["save_adapter"], config["train_one_pass"]) != (
                alpha, 0, 0, "synthetic_ce", 8, False, True, True):
        raise RuntimeError(f"unexpected final config: {config}")
    if not adapter_dir.is_dir() or not REQUIRED_ADAPTER_FILES <= {
            path.name for path in adapter_dir.iterdir()}:
        raise FileNotFoundError(f"adapter files missing in {adapter_dir}")
    metadata = json.loads((adapter_dir / "run_metadata.json").read_text(
        encoding="utf-8"))
    if (metadata["noise_alpha"], metadata["noise_seed"],
            metadata["training_seed"], metadata["optimizer_steps"],
            metadata["processed_examples"]) != (alpha, 0, 0, 374, 1500):
        raise RuntimeError(f"unexpected adapter metadata: {metadata}")


def run_process(command: list[str], cwd: Path, log_path: Path, env: dict):
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=log_file, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_synthetic_ce_epoch_xpu_qwen3.5_0.8b.yaml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    print(f"[start] sequential adapter sweep from {root}", flush=True)
    for alpha, suffix in RUNS:
        run_name = ("adapter_check_a0p00_seed0" if alpha == 0.0
                    else f"saved_adapter_a{suffix}_seed0")
        run_dir = root / "results" / run_name
        adapter_dir = root / "artifacts" / "lora_adapters" / f"alpha_{suffix}"
        if run_dir.exists() or adapter_dir.exists():
            raise FileExistsError(f"refusing to overwrite {run_dir} or {adapter_dir}")
        print(f"[run] alpha={alpha:.2f} -> {run_dir}", flush=True)
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        train_command = [
            sys.executable, "-m", "src.distill", "--config", args.config,
            "--alpha", str(alpha), "--run-name", run_name,
            "--adapter-output-dir", str(adapter_dir.relative_to(root)),
        ]
        run_process(train_command, root, root / "results" / f"{run_name}.console.log", env)
        audit_run(run_dir, alpha, adapter_dir)
        verify_command = [
            sys.executable, "-m", "src.verify_adapter", "--adapter-dir",
            str(adapter_dir.relative_to(root)), "--output",
            str((adapter_dir / "verify_report.json").relative_to(root)),
        ]
        run_process(verify_command, root,
                    root / "results" / f"{run_name}.verify.console.log", env)
        if not (adapter_dir / "verify_report.json").is_file():
            raise FileNotFoundError(f"missing verification report for alpha={alpha}")
        print(f"[ok] alpha={alpha:.2f} -> {adapter_dir}", flush=True)
    run_process([sys.executable, "-m", "src.package_adapters"], root,
                root / "results" / "saved_adapter_package.console.log", env)
    print("[ok] reports and Yandex archive created", flush=True)


if __name__ == "__main__":
    main()
