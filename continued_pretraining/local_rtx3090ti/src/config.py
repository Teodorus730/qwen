from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str | None = None
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str
    revision: str | None
    subset: str
    split: str = "train"
    text_field: str = "text"
    sequence_length: int = 512
    local_slice: str = "artifacts/data/fineweb_edu_dedup_512_docs.jsonl"
    local_documents: int = 512
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 10_000


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "outputs/continued_pretraining"
    max_steps: int = 100
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 10
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    gradient_clip: float = 1.0
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = False
    seed: int = 42
    log_every: int = 1
    save_every: int = 0
    save_final: bool = False
    cuda_memory_fraction: float = 0.90


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_sizes: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12)
    warmup_steps: int = 2
    measured_steps: int = 5
    stop_after_oom: bool = True
    results_dir: str = "results"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    benchmark: BenchmarkConfig
    root: Path


def _construct(cls, values: dict[str, Any]):
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    if cls is BenchmarkConfig and "batch_sizes" in values:
        values = {**values, "batch_sizes": tuple(values["batch_sizes"])}
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"model", "data", "training", "benchmark"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    root = path.parent.parent
    config = ExperimentConfig(
        model=_construct(ModelConfig, raw["model"]),
        data=_construct(DataConfig, raw["data"]),
        training=_construct(TrainingConfig, raw["training"]),
        benchmark=_construct(BenchmarkConfig, raw["benchmark"]),
        root=root,
    )
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    if config.data.sequence_length < 2:
        raise ValueError("data.sequence_length must be >= 2")
    if config.data.local_documents < 1:
        raise ValueError("data.local_documents must be positive")
    if config.training.max_steps < 1:
        raise ValueError("training.max_steps must be positive")
    if config.training.micro_batch_size < 1:
        raise ValueError("training.micro_batch_size must be positive")
    if config.training.gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    if not 0 < config.training.cuda_memory_fraction <= 1:
        raise ValueError("training.cuda_memory_fraction must be in (0, 1]")
    if config.benchmark.warmup_steps < 1:
        raise ValueError("benchmark.warmup_steps must be >= 1")
    if config.benchmark.measured_steps < 1:
        raise ValueError("benchmark.measured_steps must be >= 1")
    if not config.benchmark.batch_sizes or any(
        batch < 1 for batch in config.benchmark.batch_sizes
    ):
        raise ValueError("benchmark.batch_sizes must contain positive integers")


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (root / value).resolve()
