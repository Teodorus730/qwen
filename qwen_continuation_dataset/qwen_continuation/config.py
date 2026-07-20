from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    cycle = config["generation"].get("cycle_detection", {})
    if cycle.get("enabled", False):
        ngram_chars = int(cycle.get("ngram_chars", 20))
        window_chars = int(cycle.get("window_chars", 100))
        if ngram_chars <= 0:
            raise ValueError(
                "generation.cycle_detection.ngram_chars должен быть положительным."
            )
        if window_chars <= ngram_chars:
            raise ValueError(
                "generation.cycle_detection.window_chars должен быть больше "
                "ngram_chars."
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
    batch_size: int | None = None,
    cycle_enabled: bool | None = None,
    cycle_window_chars: int | None = None,
    cycle_ngram_chars: int | None = None,
    cycle_min_chars: int | None = None,
    save_extra_fields: bool | None = None,
    extra_fields_path: str | None = None,
    extra_fields_batch_size: int | None = None,
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
    if batch_size is not None:
        updated["generation"]["batch_size"] = batch_size
    if any(
        v is not None
        for v in (cycle_enabled, cycle_window_chars, cycle_ngram_chars, cycle_min_chars)
    ):
        cycle_section = updated["generation"].setdefault("cycle_detection", {})
        if cycle_enabled is not None:
            cycle_section["enabled"] = cycle_enabled
        if cycle_window_chars is not None:
            cycle_section["window_chars"] = cycle_window_chars
        if cycle_ngram_chars is not None:
            cycle_section["ngram_chars"] = cycle_ngram_chars
        if cycle_min_chars is not None:
            cycle_section["min_chars"] = cycle_min_chars
    if any(
        v is not None
        for v in (save_extra_fields, extra_fields_path, extra_fields_batch_size)
    ):
        extra_section = updated["output"].setdefault("extra_fields", {})
        if save_extra_fields is not None:
            extra_section["enabled"] = save_extra_fields
        if extra_fields_path is not None:
            extra_section["path"] = extra_fields_path
        if extra_fields_batch_size is not None:
            extra_section["batch_size"] = extra_fields_batch_size
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
