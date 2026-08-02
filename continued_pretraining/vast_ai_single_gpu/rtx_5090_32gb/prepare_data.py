from __future__ import annotations

import argparse

from src.config import load_config
from src.data import load_or_build_blocks, materialize_slice
from src.runtime import (
    configure_project_environment,
    load_tokenizer,
    project_data_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and pack pinned FineWeb-Edu train/validation data."
    )
    parser.add_argument("--config", default="configs/vast_5090_32gb.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_project_environment(cfg.root)
    slice_path, blocks_path = project_data_paths(cfg)
    slice_report = materialize_slice(
        cfg.data,
        slice_path,
        overwrite=args.overwrite,
    )
    print(f"[slice] {slice_report}", flush=True)

    tokenizer = load_tokenizer(cfg)
    train_blocks, validation_blocks, metadata = load_or_build_blocks(
        cfg.data,
        slice_path,
        tokenizer,
        blocks_path,
    )
    print(f"[blocks] {metadata}", flush=True)
    print(
        f"[done] train={tuple(train_blocks.shape)} "
        f"validation={tuple(validation_blocks.shape)}",
        flush=True,
    )
    print(f"[done] slice={slice_path}", flush=True)
    print(f"[done] cache={blocks_path}", flush=True)


if __name__ == "__main__":
    main()
