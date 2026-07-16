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
