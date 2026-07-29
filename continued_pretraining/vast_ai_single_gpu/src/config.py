from __future__ import annotations

import math
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
    local_slice: str = "artifacts/data/fineweb_edu_dedup_20512_docs.jsonl"
    total_documents: int = 20_512
    validation_documents: int = 512
    shuffle_seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str = "vast_10m"
    output_dir: str = "outputs"
    max_tokens: int = 10_000_000
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    min_learning_rate_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    gradient_clip: float = 1.0
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = False
    seed: int = 42
    log_every: int = 1
    eval_every: int = 100
    eval_blocks: int = 64
    save_every: int = 100
    keep_last_checkpoints: int = 2
    save_final: bool = True
    cuda_memory_fraction: float = 0.90


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_sizes: tuple[int, ...] = (1, 2, 4)
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

    @property
    def tokens_per_update(self) -> int:
        return (
            self.data.sequence_length
            * self.training.micro_batch_size
            * self.training.gradient_accumulation_steps
        )

    @property
    def total_updates(self) -> int:
        return math.ceil(self.training.max_tokens / self.tokens_per_update)


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
    config = ExperimentConfig(
        model=_construct(ModelConfig, raw["model"]),
        data=_construct(DataConfig, raw["data"]),
        training=_construct(TrainingConfig, raw["training"]),
        benchmark=_construct(BenchmarkConfig, raw["benchmark"]),
        root=path.parent.parent,
    )
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    data = config.data
    training = config.training
    benchmark = config.benchmark
    if data.sequence_length < 2:
        raise ValueError("data.sequence_length must be >= 2")
    if data.validation_documents < 1:
        raise ValueError("data.validation_documents must be positive")
    if data.total_documents <= data.validation_documents:
        raise ValueError("total_documents must exceed validation_documents")
    for name in ("max_tokens", "micro_batch_size", "gradient_accumulation_steps"):
        if getattr(training, name) < 1:
            raise ValueError(f"training.{name} must be positive")
    if not 0 <= training.warmup_ratio < 1:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if not 0 < training.cuda_memory_fraction <= 1:
        raise ValueError("training.cuda_memory_fraction must be in (0, 1]")
    if training.eval_every < 1 or training.eval_blocks < 1:
        raise ValueError("eval_every and eval_blocks must be positive")
    if training.save_every < 1 or training.keep_last_checkpoints < 1:
        raise ValueError("save_every and keep_last_checkpoints must be positive")
    if benchmark.warmup_steps < 1 or benchmark.measured_steps < 1:
        raise ValueError("benchmark warmup/measured steps must be positive")
    if not benchmark.batch_sizes or any(batch < 1 for batch in benchmark.batch_sizes):
        raise ValueError("benchmark.batch_sizes must contain positive integers")


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (root / value).resolve()

