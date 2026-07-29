from __future__ import annotations

import argparse

from src.config import load_config
from src.data import load_or_build_blocks, materialize_slice
from src.runtime import (
    configure_project_caches,
    load_tokenizer,
    project_data_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialise and tokenize a small, exact SmolLM-Corpus slice."
    )
    parser.add_argument("--config", default="configs/rtx3090ti.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_project_caches(cfg.root)
    slice_path, blocks_path = project_data_paths(cfg)

    slice_report = materialize_slice(
        cfg.data,
        slice_path,
        overwrite=args.overwrite,
    )
    print(f"[slice] {slice_report}", flush=True)

    tokenizer = load_tokenizer(cfg)
    blocks, blocks_report = load_or_build_blocks(
        cfg.data,
        slice_path,
        tokenizer,
        blocks_path,
    )
    print(f"[blocks] shape={tuple(blocks.shape)} {blocks_report}", flush=True)
    print(f"[done] slice={slice_path}", flush=True)
    print(f"[done] token blocks={blocks_path}", flush=True)


if __name__ == "__main__":
    main()

