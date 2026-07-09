"""
Materialise a local FineWeb-Edu slice to JSONL, once, so training/eval read
from disk instead of streaming the Hub every run (robust + offline + fast).

    python -m src.fetch_corpus --n 20000 --out data/fineweb_edu_local.jsonl

Each line is {"text": ..., "score": ...}. The distiller's DataConfig.local_jsonl
points at this file by default and falls back to streaming if it's absent.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="num docs to save")
    ap.add_argument("--out", type=str, default="data/fineweb_edu_local.jsonl")
    ap.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", type=str, default="sample-10BT")
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    attempt = 0
    t0 = time.time()
    f = open(out, "w", encoding="utf-8")
    while written < args.n and attempt < args.retries:
        attempt += 1
        try:
            ds = load_dataset(args.dataset, name=args.name, split="train",
                              streaming=True)
            # skip what we already wrote so retries don't duplicate the head
            ds = ds.skip(written)
            for ex in ds:
                if args.min_score and ex.get("score", 1e9) < args.min_score:
                    continue
                txt = ex.get("text")
                if not txt:
                    continue
                f.write(json.dumps({"text": txt,
                                    "score": ex.get("score")}) + "\n")
                written += 1
                if written % 1000 == 0:
                    f.flush()
                    print(f"  {written}/{args.n} docs "
                          f"({time.time()-t0:.0f}s)", flush=True)
                if written >= args.n:
                    break
        except Exception as e:  # transient stream/client errors -> retry
            print(f"[retry {attempt}] {type(e).__name__}: {e} "
                  f"(have {written})", flush=True)
            time.sleep(3)
    f.close()
    print(f"[done] wrote {written} docs -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
