from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("model", "dataset", "generation", "output")
DATASET_SOURCES = {"fineweb", "math", "mixed"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")

    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    validate_config(config)
    return config


def _validate_dataset_source(name: str, source: dict[str, Any]) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"dataset.sources.{name} must be a mapping.")
    if not source.get("id"):
        raise ValueError(f"dataset.sources.{name}.id must not be empty.")
    if not source.get("text_column"):
        raise ValueError(
            f"dataset.sources.{name}.text_column must not be empty."
        )


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    dataset = config["dataset"]
    generation = config["generation"]
    output = config["output"]

    if not model.get("id"):
        raise ValueError("model.id must not be empty.")

    source = dataset.get("source", "fineweb")
    if source not in DATASET_SOURCES:
        allowed = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"dataset.source must be one of: {allowed}.")

    sources = dataset.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("dataset.sources must be a mapping.")
    for required_source in ("fineweb", "math"):
        if required_source not in sources:
            raise ValueError(
                f"Missing dataset.sources.{required_source} configuration."
            )
        _validate_dataset_source(required_source, sources[required_source])

    if int(dataset.get("max_examples", 0)) <= 0:
        raise ValueError("dataset.max_examples must be positive.")

    math_ratio = float(dataset.get("mixed", {}).get("math_ratio", 0.5))
    if not 0.0 <= math_ratio <= 1.0:
        raise ValueError("dataset.mixed.math_ratio must be in [0, 1].")

    mode = generation.get("mode")
    if mode not in {"fixed", "entropy"}:
        raise ValueError("generation.mode must be 'fixed' or 'entropy'.")
    if int(generation.get("prefix_tokens", 0)) <= 0:
        raise ValueError("generation.prefix_tokens must be positive.")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be positive.")
    if float(generation.get("temperature", 0.0)) < 0:
        raise ValueError("generation.temperature must be >= 0.")
    if not 0 < float(generation.get("top_p", 1.0)) <= 1:
        raise ValueError("generation.top_p must be in (0, 1].")
    if mode == "entropy" and float(
        generation.get("entropy_threshold", 0.0)
    ) <= 0:
        raise ValueError("generation.entropy_threshold must be positive.")

    if not output.get("path"):
        raise ValueError("output.path must not be empty.")

    hf = config.get("huggingface", {})
    if hf.get("enabled", False):
        repo_id = hf.get("repo_id")
        if not repo_id or repo_id == "your_username/qwen_continuation_dataset":
            raise ValueError(
                "huggingface.repo_id must be set to a real repo when "
                "huggingface.enabled is true."
            )


def with_overrides(
    config: dict[str, Any],
    *,
    max_examples: int | None = None,
    output_path: str | None = None,
    mode: str | None = None,
    dataset_source: str | None = None,
    math_ratio: float | None = None,
    prefix_tokens: int | None = None,
    max_new_tokens: int | None = None,
    entropy_threshold: float | None = None,
    hf_upload: bool | None = None,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    hf_shard_size: int | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    if max_examples is not None:
        updated["dataset"]["max_examples"] = max_examples
    if output_path is not None:
        updated["output"]["path"] = output_path
    if mode is not None:
        updated["generation"]["mode"] = mode
    if dataset_source is not None:
        updated["dataset"]["source"] = dataset_source
    if math_ratio is not None:
        updated["dataset"].setdefault("mixed", {})["math_ratio"] = math_ratio
    if prefix_tokens is not None:
        updated["generation"]["prefix_tokens"] = prefix_tokens
    if max_new_tokens is not None:
        updated["generation"]["max_new_tokens"] = max_new_tokens
    if entropy_threshold is not None:
        updated["generation"]["entropy_threshold"] = entropy_threshold
    if any(v is not None for v in (hf_upload, hf_repo_id, hf_token, hf_shard_size)):
        hf_section = updated.setdefault("huggingface", {})
        if hf_upload is not None:
            hf_section["enabled"] = hf_upload
        if hf_repo_id is not None:
            hf_section["repo_id"] = hf_repo_id
        if hf_token is not None:
            hf_section["token"] = hf_token
        if hf_shard_size is not None:
            hf_section["shard_size"] = hf_shard_size
    validate_config(updated)
    return updated
