from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, resolve


def configure_project_caches(root: Path) -> None:
    artifacts = root / "artifacts"
    # Keep every downloaded artifact inside this experiment. Assignment is
    # intentional (rather than setdefault): this repository has a strict local
    # cache policy and should not silently use a user's global Hugging Face cache.
    os.environ["HF_HOME"] = str(artifacts / "hf_cache")
    os.environ["HF_DATASETS_CACHE"] = str(artifacts / "hf_datasets_cache")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str):
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose {sorted(values)}") from error


def load_tokenizer(cfg: ExperimentConfig, source: str | Path | None = None):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer_source = str(source or cfg.model.id)
    revision = None if source else cfg.model.revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=revision,
        )
    except ValueError as error:
        # This checkpoint was saved by Transformers 5.1 with
        # tokenizer_class="TokenizersBackend". Transformers 4.57 can consume
        # the underlying tokenizer.json but does not know that new class name.
        if "TokenizersBackend" not in str(error):
            raise
        if source:
            tokenizer_file = Path(source) / "tokenizer.json"
        else:
            from huggingface_hub import hf_hub_download

            tokenizer_file = Path(
                hf_hub_download(
                    repo_id=cfg.model.id,
                    filename="tokenizer.json",
                    revision=cfg.model.revision,
                )
            )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
    expected = {"eos": 151643, "pad": 151669, "length": 151670}
    actual = {
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "length": len(tokenizer),
    }
    if actual != expected:
        raise ValueError(
            f"Tokenizer identity check failed: expected {expected}, got {actual}"
        )
    return tokenizer


def load_model(
    cfg: ExperimentConfig,
    device: torch.device,
    source: str | Path | None = None,
):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(source or cfg.model.id),
        revision=None if source else cfg.model.revision,
        dtype=torch_dtype(cfg.model.dtype),
        attn_implementation=cfg.model.attention_implementation,
    )
    model.config.use_cache = False
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model.to(device)


def build_optimizer(cfg: ExperimentConfig, model):
    training = cfg.training
    kwargs = {
        "lr": training.learning_rate,
        "betas": (training.adam_beta1, training.adam_beta2),
        "weight_decay": training.weight_decay,
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if training.optimizer == "paged_adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(parameters, **kwargs)
    if training.optimizer == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {training.optimizer}")


def learning_rate_at(cfg: ExperimentConfig, update_step: int) -> float:
    training = cfg.training
    if update_step < training.warmup_steps:
        return training.learning_rate * (update_step + 1) / max(
            1, training.warmup_steps
        )
    progress = (update_step - training.warmup_steps) / max(
        1, training.max_steps - training.warmup_steps
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    ratio = training.min_learning_rate_ratio
    return training.learning_rate * (ratio + (1.0 - ratio) * cosine)


def project_data_paths(cfg: ExperimentConfig) -> tuple[Path, Path]:
    slice_path = resolve(cfg.root, cfg.data.local_slice)
    blocks_name = (
        f"{slice_path.stem}.seq{cfg.data.sequence_length}."
        f"{cfg.model.revision[:8] if cfg.model.revision else 'main'}.pt"
    )
    return slice_path, slice_path.with_name(blocks_name)


def environment_report() -> dict:
    import bitsandbytes
    import datasets
    import transformers

    cuda = {}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda = {
            "available": True,
            "device_name": properties.name,
            "total_vram_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    else:
        cuda = {"available": False}
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": cuda,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        report["nvidia_driver"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        report["nvidia_driver"] = None
    return report


def config_as_dict(cfg: ExperimentConfig) -> dict:
    value = asdict(cfg)
    value["root"] = str(cfg.root)
    return value


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
