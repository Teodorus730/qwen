"""Create a portable package from the eight independently verified adapters."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from .rebuild_final_results import RUNS, collect


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = collect(root)
    artifacts = root / "artifacts"
    upload = artifacts / "yandex_upload"
    archive = artifacts / "qwen35_08b_lora_recovery_adapters_seed0.zip"
    if upload.exists() or archive.exists():
        raise FileExistsError("remove existing local package before rebuilding it")
    (upload / "adapters").mkdir(parents=True)

    first_metadata = json.loads((artifacts / "lora_adapters" / "alpha_0p00" /
                                 "run_metadata.json").read_text(encoding="utf-8"))
    manifest = {"base_model": first_metadata["base_model"],
                "base_model_revision": first_metadata["base_model_revision"],
                "application_order": "clean base -> Gaussian noise -> PEFT adapter",
                "adapters": []}
    for row, (alpha, _run_name, suffix) in zip(rows, RUNS):
        source = artifacts / "lora_adapters" / f"alpha_{suffix}"
        target = upload / "adapters" / source.name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("README.md"))
        metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(target.iterdir()) if path.is_file()}
        manifest["adapters"].append({
            "alpha": alpha, "noise_seed": metadata["noise_seed"],
            "training_seed": metadata["training_seed"], "base_model": metadata["base_model"],
            "base_model_revision": metadata["base_model_revision"],
            "adapter_path": f"adapters/{source.name}", "files": files,
            "lora_rank": metadata["lora_rank"], "lora_alpha": metadata["lora_alpha"],
            "lora_dropout": metadata["lora_dropout"],
            "target_modules": metadata["target_modules"],
            "dataset_sha256": metadata["dataset_sha256"],
            "final_metrics": metadata["final_metrics"],
            "source_run": metadata["source_run"].replace("\\", "/"),
            "verification_status": "passed",
            "verification_logit_delta": row["verification_logit_delta"],
        })
    shutil.copy2(root / "configs" / "main_synthetic_ce_epoch_xpu_qwen3.5_0.8b.yaml",
                 upload / "base_config.yaml")
    env = {key: rows[0][key] for key in ("noise_seed", "training_seed")}
    env.update({"python": sys.version, "device": "xpu", "dtype": "bfloat16",
                "torch": first_metadata["versions"]["torch"],
                "transformers": first_metadata["versions"]["transformers"],
                "peft": first_metadata["versions"]["peft"]})
    (upload / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (upload / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksums = [f"{sha256(path)}  {path.relative_to(upload).as_posix()}"
                 for path in sorted(upload.rglob("*")) if path.is_file()]
    (upload / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(upload.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(artifacts))
    print(json.dumps({"upload": str(upload), "archive": str(archive),
                      "archive_sha256": sha256(archive)}, indent=2))


if __name__ == "__main__":
    main()
