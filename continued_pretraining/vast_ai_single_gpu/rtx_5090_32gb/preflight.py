from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import psutil
import torch
from packaging.version import Version

from src.config import load_config
from src.runtime import (
    configure_project_environment,
    environment_report,
    write_json,
)

GIB = 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-fast Vast.ai environment and pinned-HF availability check."
    )
    parser.add_argument("--config", default="configs/vast_5090_32gb.yaml")
    parser.add_argument("--min-free-disk-gb", type=float, default=15.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_project_environment(cfg.root)
    from huggingface_hub import HfApi

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str, *, fatal: bool = True) -> None:
        checks.append(
            {"name": name, "ok": bool(ok), "detail": detail, "fatal": fatal}
        )
        marker = "OK" if ok else ("FAIL" if fatal else "WARN")
        print(f"[{marker}] {name}: {detail}", flush=True)

    check(
        "torch_version",
        Version(torch.__version__.split("+")[0]) >= Version("2.6"),
        torch.__version__,
    )
    check("cuda_available", torch.cuda.is_available(), str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        total_gib = properties.total_memory / GIB
        check("gpu", total_gib >= 10, f"{properties.name}, {total_gib:.2f} GiB")
        check(
            "bf16",
            torch.cuda.is_bf16_supported(),
            str(torch.cuda.is_bf16_supported()),
        )
        left = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        product = left @ right
        torch.cuda.synchronize()
        check(
            "cuda_compute",
            bool(torch.isfinite(product).all()),
            "BF16 matrix multiplication completed",
        )

        try:
            import bitsandbytes as bnb

            parameter = torch.nn.Parameter(torch.zeros(4096, device="cuda"))
            optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=1e-4)
            parameter.sum().backward()
            optimizer.step()
            check("bitsandbytes_optimizer", True, bnb.__version__)
        except Exception as error:
            check(
                "bitsandbytes_optimizer",
                False,
                f"{type(error).__name__}: {error}",
            )

    disk = shutil.disk_usage(cfg.root)
    disk_free_gib = disk.free / GIB
    check(
        "free_disk",
        disk_free_gib >= args.min_free_disk_gb,
        f"{disk_free_gib:.2f} GiB free (required {args.min_free_disk_gb:.2f})",
    )
    ram_gib = psutil.virtual_memory().total / GIB
    check(
        "system_ram",
        ram_gib >= 16,
        f"{ram_gib:.2f} GiB total",
        fatal=False,
    )

    api = HfApi()
    try:
        model = api.model_info(cfg.model.id, revision=cfg.model.revision)
        check("model_revision", model.sha == cfg.model.revision, model.sha)
    except Exception as error:
        check("model_revision", False, f"{type(error).__name__}: {error}")
    try:
        dataset = api.dataset_info(
            cfg.data.dataset_id,
            revision=cfg.data.revision,
        )
        check("dataset_revision", dataset.sha == cfg.data.revision, dataset.sha)
    except Exception as error:
        check("dataset_revision", False, f"{type(error).__name__}: {error}")

    report = {
        "config": str(Path(args.config).resolve()),
        "checks": checks,
        "environment": environment_report(cfg.root),
    }
    write_json(cfg.root / "results" / "preflight.json", report)
    failed = [item for item in checks if item["fatal"] and not item["ok"]]
    if failed:
        print(f"[preflight] FAILED: {len(failed)} fatal checks", flush=True)
        return 1
    print("[preflight] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
