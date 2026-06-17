from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def _stream_one_source(
    source_name: str,
    source_config: dict[str, Any],
    *,
    seed: int,
) -> Iterator[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "path": source_config["id"],
        "split": source_config.get("split", "train"),
        "streaming": bool(source_config.get("streaming", True)),
    }
    subset = source_config.get("subset")
    if subset:
        kwargs["name"] = subset

    dataset = load_dataset(**kwargs)

    if source_config.get("shuffle", False):
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=int(source_config.get("shuffle_buffer_size", 1000)),
        )

    text_column = source_config.get("text_column", "text")
    id_column = source_config.get("id_column")

    for index, row in enumerate(dataset):
        text = row.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue

        source_id = row.get(id_column) if id_column else None
        if source_id is None:
            source_id = row.get("id") or row.get("url") or f"row-{index}"

        metadata = {
            key: value
            for key, value in row.items()
            if key != text_column
            and isinstance(value, (str, int, float, bool, type(None)))
        }

        yield {
            "source_id": f"{source_name}:{source_id}",
            "source_name": source_name,
            "source_dataset": source_config["id"],
            "source_subset": subset,
            "text": text,
            "metadata": metadata,
        }


def _stream_mixed(
    fineweb: Iterator[dict[str, Any]],
    math: Iterator[dict[str, Any]],
    *,
    math_ratio: float,
    seed: int,
) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    fineweb_done = False
    math_done = False

    while not (fineweb_done and math_done):
        choose_math = rng.random() < math_ratio

        if choose_math and not math_done:
            try:
                yield next(math)
                continue
            except StopIteration:
                math_done = True

        if not fineweb_done:
            try:
                yield next(fineweb)
                continue
            except StopIteration:
                fineweb_done = True

        if not math_done:
            try:
                yield next(math)
                continue
            except StopIteration:
                math_done = True


def stream_documents(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    dataset_config = config["dataset"]
    selected = dataset_config.get("source", "fineweb")
    sources = dataset_config["sources"]
    seed = int(config["generation"].get("seed", 42))

    if selected in {"fineweb", "math"}:
        yield from _stream_one_source(
            selected,
            sources[selected],
            seed=seed,
        )
        return

    fineweb_stream = _stream_one_source(
        "fineweb",
        sources["fineweb"],
        seed=seed,
    )
    math_stream = _stream_one_source(
        "math",
        sources["math"],
        seed=seed + 1,
    )
    math_ratio = float(dataset_config.get("mixed", {}).get("math_ratio", 0.5))

    yield from _stream_mixed(
        fineweb_stream,
        math_stream,
        math_ratio=math_ratio,
        seed=seed + 2,
    )
