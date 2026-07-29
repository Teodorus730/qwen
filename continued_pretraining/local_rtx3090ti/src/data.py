from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path

import torch

from .config import DataConfig


def stream_documents(cfg: DataConfig, *, shuffled: bool) -> Iterator[dict]:
    # Import after configure_project_caches(); datasets snapshots its cache
    # environment during import.
    from datasets import load_dataset

    dataset = load_dataset(
        cfg.dataset_id,
        cfg.subset,
        split=cfg.split,
        streaming=True,
        revision=cfg.revision,
    )
    if shuffled:
        dataset = dataset.shuffle(
            seed=cfg.shuffle_seed,
            buffer_size=cfg.shuffle_buffer_size,
        )
    for row in dataset:
        text = row.get(cfg.text_field)
        if text:
            yield row


def materialize_slice(cfg: DataConfig, path: Path, *, overwrite: bool = False) -> dict:
    if path.exists() and not overwrite:
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        if lines >= cfg.local_documents:
            return {
                "path": str(path),
                "documents": lines,
                "sha256": file_sha256(path),
                "reused": True,
            }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    written = 0
    total_chars = 0
    with temporary.open("w", encoding="utf-8") as handle:
        # The unshuffled head is deliberate: it is stable across runs and the
        # benchmark depends on tensor shapes, not semantic sample quality.
        for row in stream_documents(cfg, shuffled=False):
            record = {
                "text": row[cfg.text_field],
                "id": row.get("id"),
                "metadata": _json_safe(row.get("metadata")),
            }
            encoded = json.dumps(record, ensure_ascii=False)
            handle.write(encoded + "\n")
            written += 1
            total_chars += len(record["text"])
            if written >= cfg.local_documents:
                break
    if written < cfg.local_documents:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Dataset ended after {written} documents; wanted {cfg.local_documents}"
        )
    temporary.replace(path)
    return {
        "path": str(path),
        "documents": written,
        "characters": total_chars,
        "sha256": file_sha256(path),
        "reused": False,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_local_documents(path: Path, *, seed: int | None = None) -> list[str]:
    documents: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row.get("text")
            if text:
                documents.append(text)
    if not documents:
        raise ValueError(f"No non-empty documents found in {path}")
    if seed is not None:
        random.Random(seed).shuffle(documents)
    return documents


def pack_documents(
    documents: list[str],
    tokenizer,
    sequence_length: int,
) -> torch.Tensor:
    """Pack documents without padding, inserting exactly one EOS per document."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    blocks: list[list[int]] = []
    buffer: list[int] = []
    for text in documents:
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        buffer.extend(token_ids)
        buffer.append(eos)
        while len(buffer) >= sequence_length:
            blocks.append(buffer[:sequence_length])
            del buffer[:sequence_length]
    if not blocks:
        raise ValueError(
            f"Slice has fewer than {sequence_length} packed tokens; add documents"
        )
    return torch.tensor(blocks, dtype=torch.long)


def load_or_build_blocks(
    cfg: DataConfig,
    slice_path: Path,
    tokenizer,
    cache_path: Path,
) -> tuple[torch.Tensor, dict]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        blocks = payload["input_ids"]
        metadata = payload["metadata"]
        expected = {
            "sequence_length": cfg.sequence_length,
            "tokenizer_vocab_size": len(tokenizer),
            "slice_sha256": file_sha256(slice_path),
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            return blocks, {**metadata, "reused": True}

    documents = read_local_documents(slice_path, seed=cfg.shuffle_seed)
    blocks = pack_documents(documents, tokenizer, cfg.sequence_length)
    metadata = {
        "documents": len(documents),
        "blocks": int(blocks.shape[0]),
        "tokens": int(blocks.numel()),
        "sequence_length": cfg.sequence_length,
        "tokenizer_vocab_size": len(tokenizer),
        "slice_sha256": file_sha256(slice_path),
        "reused": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": blocks, "metadata": metadata}, cache_path)
    return blocks, metadata


def cyclic_batches(
    blocks: torch.Tensor,
    batch_size: int,
    *,
    seed: int,
) -> Iterator[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(blocks) < batch_size:
        raise ValueError(f"Need >= {batch_size} blocks, found {len(blocks)}")
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(blocks), generator=generator)
        usable = (len(order) // batch_size) * batch_size
        for start in range(0, usable, batch_size):
            yield blocks[order[start : start + batch_size]]
