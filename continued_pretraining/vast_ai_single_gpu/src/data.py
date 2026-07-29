from __future__ import annotations

import hashlib
import json
import random
from array import array
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def stream_documents(cfg: DataConfig, *, skip: int = 0) -> Iterator[dict]:
    # datasets reads HF_HOME/HF_DATASETS_CACHE during import, so this import
    # must remain lazy and happen after configure_project_environment().
    from datasets import load_dataset

    dataset = load_dataset(
        cfg.dataset_id,
        cfg.subset,
        split=cfg.split,
        streaming=True,
        revision=cfg.revision,
    )
    if skip:
        dataset = dataset.skip(skip)
    for row_index, row in enumerate(dataset, start=skip):
        text = row.get(cfg.text_field)
        if not isinstance(text, str) or not text:
            raise RuntimeError(
                f"Dataset row {row_index} has no non-empty "
                f"{cfg.text_field!r}; refusing to break deterministic resume"
            )
        yield row


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def materialize_slice(
    cfg: DataConfig,
    path: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Download a resumable local JSONL slice.

    The final path appears only after all requested documents are present.
    Interrupted downloads remain as *.partial and resume on the next call.
    """
    if path.exists() and not overwrite:
        lines = _count_lines(path)
        if lines == cfg.total_documents:
            return {
                "path": str(path),
                "documents": lines,
                "sha256": file_sha256(path),
                "reused": True,
            }
        raise RuntimeError(
            f"{path} contains {lines} lines, expected exactly "
            f"{cfg.total_documents}; use --overwrite to rebuild it"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if overwrite:
        partial.unlink(missing_ok=True)
        path.unlink(missing_ok=True)

    written = _count_lines(partial)
    if written > cfg.total_documents:
        raise RuntimeError(
            f"Partial slice has {written} lines, more than requested "
            f"{cfg.total_documents}; remove it with --overwrite"
        )
    mode = "a" if written else "w"
    print(
        f"[download] resuming at document {written}/{cfg.total_documents}",
        flush=True,
    )
    with partial.open(mode, encoding="utf-8") as handle:
        for row in stream_documents(cfg, skip=written):
            record = {
                "text": row[cfg.text_field],
                "id": row.get("id"),
                "metadata": _json_safe(row.get("metadata")),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                handle.flush()
                print(
                    f"[download] {written}/{cfg.total_documents} documents",
                    flush=True,
                )
            if written >= cfg.total_documents:
                break
    if written != cfg.total_documents:
        raise RuntimeError(
            f"Dataset stream ended at {written}/{cfg.total_documents} documents"
        )
    partial.replace(path)
    return {
        "path": str(path),
        "documents": written,
        "sha256": file_sha256(path),
        "reused": False,
    }


def read_documents(path: Path) -> list[str]:
    documents: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row.get("text")
            if text:
                documents.append(text)
    if not documents:
        raise ValueError(f"No non-empty documents in {path}")
    return documents


def pack_documents(
    documents: list[str],
    tokenizer,
    sequence_length: int,
) -> torch.Tensor:
    """Pack to compact int32 blocks; batches are cast to int64 on GPU."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    token_buffer = array("I")
    for index, text in enumerate(documents, start=1):
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        token_buffer.extend(ids)
        token_buffer.append(eos)
        if index % 1000 == 0:
            print(
                f"[tokenize] {index}/{len(documents)} documents, "
                f"{len(token_buffer):,} tokens",
                flush=True,
            )
    block_count = len(token_buffer) // sequence_length
    if block_count < 1:
        raise ValueError("Not enough tokens for one packed block")
    usable = block_count * sequence_length
    values = np.frombuffer(token_buffer, dtype=np.uint32, count=usable)
    # Token IDs are < 151,680, so signed int32 is lossless and compact.
    values = values.astype(np.int32, copy=True).reshape(
        block_count, sequence_length
    )
    return torch.from_numpy(values)


def load_or_build_blocks(
    cfg: DataConfig,
    slice_path: Path,
    tokenizer,
    cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    slice_hash = file_sha256(slice_path)
    expected = {
        "sequence_length": cfg.sequence_length,
        "tokenizer_vocab_size": len(tokenizer),
        "slice_sha256": slice_hash,
        "total_documents": cfg.total_documents,
        "validation_documents": cfg.validation_documents,
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if all(metadata.get(key) == value for key, value in expected.items()):
            return (
                payload["train_input_ids"],
                payload["validation_input_ids"],
                {**metadata, "reused": True},
            )

    documents = read_documents(slice_path)
    if len(documents) != cfg.total_documents:
        raise RuntimeError(
            f"Read {len(documents)} documents, expected {cfg.total_documents}"
        )
    split_at = len(documents) - cfg.validation_documents
    train_documents = documents[:split_at]
    validation_documents = documents[split_at:]
    random.Random(cfg.shuffle_seed).shuffle(train_documents)

    print("[tokenize] building train blocks", flush=True)
    train_blocks = pack_documents(
        train_documents, tokenizer, cfg.sequence_length
    )
    print("[tokenize] building held-out validation blocks", flush=True)
    validation_blocks = pack_documents(
        validation_documents, tokenizer, cfg.sequence_length
    )
    metadata = {
        **expected,
        "train_documents": len(train_documents),
        "train_blocks": int(train_blocks.shape[0]),
        "train_tokens": int(train_blocks.numel()),
        "validation_blocks": int(validation_blocks.shape[0]),
        "validation_tokens": int(validation_blocks.numel()),
        "storage_dtype": str(train_blocks.dtype),
        "reused": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
    torch.save(
        {
            "train_input_ids": train_blocks,
            "validation_input_ids": validation_blocks,
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(cache_path)
    return train_blocks, validation_blocks, metadata


def cyclic_batches(
    blocks: torch.Tensor,
    batch_size: int,
    *,
    seed: int,
) -> Iterator[torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(blocks) < batch_size:
        raise ValueError(f"Need >= {batch_size} blocks, found {len(blocks)}")
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(blocks), generator=generator)
        usable = (len(order) // batch_size) * batch_size
        for start in range(0, usable, batch_size):
            yield blocks[order[start : start + batch_size]]


def fixed_batches(
    blocks: torch.Tensor,
    batch_size: int,
    *,
    max_blocks: int,
) -> Iterator[torch.Tensor]:
    count = min(len(blocks), max_blocks)
    for start in range(0, count, batch_size):
        batch = blocks[start : min(start + batch_size, count)]
        if len(batch):
            yield batch
