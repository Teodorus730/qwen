"""
Data loading for the distillation-recovery experiment.

We stream FineWeb-Edu (never download the full corpus) and pack documents into
fixed-length token blocks, the standard pretraining format. The same packing is
used for:

  * the distillation training stream,
  * a held-out validation block set (for perplexity),
  * a fixed "probe" batch (for KL-to-teacher and CKA), which must be identical
    across checkpoints so the metrics are comparable.

Streaming + on-the-fly tokenisation keeps memory flat and lets the full
3090 Ti run consume an arbitrary number of tokens without disk blow-up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset


@dataclass
class DataConfig:
    dataset: str = "HuggingFaceFW/fineweb-edu"
    name: str | None = "sample-10BT"
    split: str = "train"
    text_field: str = "text"
    seq_len: int = 1024
    min_score: float = 0.0           # filter FineWeb-Edu by edu score
    seed: int = 0
    # If set and present, read docs from this local JSONL (one {"text",...}
    # per line) instead of streaming the Hub. Far more robust for repeated
    # runs and works offline. See src/fetch_corpus.py to build it.
    local_jsonl: str | None = "data/fineweb_edu_local.jsonl"


def _local_doc_stream(path: str, cfg: DataConfig, skip: int):
    import json
    n = 0
    while True:                       # cycle the file so training never starves
        any_yielded = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if cfg.min_score and ex.get("score", 1e9) < cfg.min_score:
                    continue
                if n < skip:
                    n += 1
                    continue
                # Continuation datasets store the generated training text under
                # `synthetic_text`; ordinary FineWeb JSONL keeps using `text`.
                txt = ex.get(cfg.text_field) or ex.get("synthetic_text")
                if txt:
                    any_yielded = True
                    yield txt
        if not any_yielded:           # skip exhausted the file
            return


def _doc_stream(cfg: DataConfig, skip: int = 0):
    if cfg.local_jsonl and os.path.exists(cfg.local_jsonl):
        yield from _local_doc_stream(cfg.local_jsonl, cfg, skip)
        return
    ds = load_dataset(cfg.dataset, name=cfg.name, split=cfg.split, streaming=True)
    ds = ds.shuffle(seed=cfg.seed, buffer_size=10_000)
    n = 0
    for ex in ds:
        if cfg.min_score and ex.get("score", 1e9) < cfg.min_score:
            continue
        if n < skip:
            n += 1
            continue
        txt = ex.get(cfg.text_field)
        if txt:
            yield txt


def token_block_stream(
    tokenizer, cfg: DataConfig, skip_docs: int = 0
) -> Iterator[torch.Tensor]:
    """Yield 1-D LongTensors of length seq_len, packed across document
    boundaries with EOS separators (GPT-style concatenation)."""
    eos = tokenizer.eos_token_id
    buf: list[int] = []
    for txt in _doc_stream(cfg, skip=skip_docs):
        ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= cfg.seq_len:
            block = buf[: cfg.seq_len]
            buf = buf[cfg.seq_len:]
            yield torch.tensor(block, dtype=torch.long)


def make_batches(
    tokenizer, cfg: DataConfig, batch_size: int, n_batches: int | None = None,
    skip_docs: int = 0,
) -> Iterator[torch.Tensor]:
    """Group token blocks into (batch_size, seq_len) tensors."""
    gen = token_block_stream(tokenizer, cfg, skip_docs=skip_docs)
    produced = 0
    rows: list[torch.Tensor] = []
    for block in gen:
        rows.append(block)
        if len(rows) == batch_size:
            yield torch.stack(rows, dim=0)
            rows = []
            produced += 1
            if n_batches is not None and produced >= n_batches:
                return


def fixed_blocks(tokenizer, cfg: DataConfig, n_blocks: int,
                 skip_docs: int = 0) -> torch.Tensor:
    """Materialise a fixed (n_blocks, seq_len) tensor for eval/probe.

    Reproducible given the same seed/skip, so metrics are comparable over time.
    """
    gen = token_block_stream(tokenizer, cfg, skip_docs=skip_docs)
    rows = []
    for block in gen:
        rows.append(block)
        if len(rows) >= n_blocks:
            break
    return torch.stack(rows, dim=0)


def one_pass_local_batches(tokenizer, cfg: DataConfig, batch_size: int):
    """Pack every row of a local JSONL exactly once.

    Unlike ``make_batches``, this helper never cycles the file. The last short
    block is padded with EOS so the final source row is not silently dropped.
    Returns materialised CPU batches plus source-ID audit information; this is
    intentionally opt-in so existing streaming configs keep their behaviour.
    """
    import json

    if not cfg.local_jsonl or not Path(cfg.local_jsonl).is_file():
        raise FileNotFoundError(
            "one-pass training requires an existing local_jsonl")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    eos = tokenizer.eos_token_id
    token_buffer: list[int] = []
    blocks: list[torch.Tensor] = []
    source_ids: list[str] = []

    with open(cfg.local_jsonl, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            ex = json.loads(line)
            if cfg.min_score and ex.get("score", 1e9) < cfg.min_score:
                continue
            source_id = ex.get("source_id")
            if not source_id:
                raise ValueError(
                    f"missing source_id at {cfg.local_jsonl}:{line_number}")
            source_ids.append(str(source_id))
            txt = ex.get(cfg.text_field) or ex.get("synthetic_text")
            if not txt:
                raise ValueError(
                    f"missing training text at {cfg.local_jsonl}:{line_number}")
            token_buffer.extend(
                tokenizer(txt, add_special_tokens=False)["input_ids"])
            token_buffer.append(eos)
            while len(token_buffer) >= cfg.seq_len:
                blocks.append(torch.tensor(
                    token_buffer[:cfg.seq_len], dtype=torch.long))
                del token_buffer[:cfg.seq_len]

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("local_jsonl contains duplicate source_id values")
    if token_buffer:
        token_buffer.extend([eos] * (cfg.seq_len - len(token_buffer)))
        blocks.append(torch.tensor(token_buffer, dtype=torch.long))

    batches = [torch.stack(blocks[i:i + batch_size])
               for i in range(0, len(blocks), batch_size)]
    return batches, {
        "source_ids": source_ids,
        "source_rows": len(source_ids),
        "unique_source_ids": len(set(source_ids)),
        "packed_blocks": len(blocks),
        "packed_batches": len(batches),
    }


def fixed_local_blocks_with_source_ids(
        tokenizer, cfg: DataConfig, path: str, n_blocks: int,
        skip_docs: int = 0):
    """Build fixed local blocks and return the source IDs contributing tokens."""
    import json

    eos = tokenizer.eos_token_id
    token_buffer: list[int] = []
    owner_buffer: list[str] = []
    blocks: list[torch.Tensor] = []
    used_source_ids: set[str] = set()

    with open(path, encoding="utf-8") as f:
        for doc_index, line in enumerate(f):
            if doc_index < skip_docs:
                continue
            ex = json.loads(line)
            if cfg.min_score and ex.get("score", 1e9) < cfg.min_score:
                continue
            source_id = ex.get("source_id")
            if not source_id:
                raise ValueError(f"missing source_id in held-out file {path}")
            txt = ex.get(cfg.text_field) or ex.get("synthetic_text")
            if not txt:
                raise ValueError(f"missing held-out text in {path}")
            ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
            ids.append(eos)
            token_buffer.extend(ids)
            owner_buffer.extend([str(source_id)] * len(ids))
            while len(token_buffer) >= cfg.seq_len:
                used_source_ids.update(owner_buffer[:cfg.seq_len])
                blocks.append(torch.tensor(
                    token_buffer[:cfg.seq_len], dtype=torch.long))
                del token_buffer[:cfg.seq_len]
                del owner_buffer[:cfg.seq_len]
                if len(blocks) == n_blocks:
                    return torch.stack(blocks), used_source_ids

    raise ValueError(
        f"held-out file {path} produced {len(blocks)} of {n_blocks} blocks")
