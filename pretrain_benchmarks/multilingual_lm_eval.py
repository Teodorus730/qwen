"""Deterministic multilingual Wikipedia language-model evaluation.

The canonical mode ranks every article in a pinned Wikimedia snapshot by a
stable SHA-256 key. It writes a tokenizer-independent, text-free corpus
manifest and materializes only the selected texts locally. Per-run tokenization
and chunk metadata are recorded with the evaluation result. A capped source
scan is smoke-only and must never be recorded as a baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_benchmark import accelerator_name, model_revision, package_version, select_backend


DATASET_REPO = "wikimedia/wikipedia"
DATASET_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DATASET_CONFIGS = {"en": "20231101.en", "zh": "20231101.zh", "ru": "20231101.ru"}
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BYTE_BUDGET = 1_048_576
EVALUATOR_VERSION = 1
PROTOCOL_PATH = Path(__file__).with_name("multilingual_wikipedia_protocol.json")
REPRODUCIBILITY_PACKAGES = (
    "torch", "transformers", "lm-eval", "datasets", "huggingface_hub",
    "tokenizers", "accelerate", "safetensors",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash the complete manifest, excluding only its self-referential field."""
    return sha256_bytes(canonical_json({key: value for key, value in manifest.items() if key != "manifest_sha256"}).encode("utf-8"))


def protocol_config() -> tuple[dict[str, Any], str]:
    raw = PROTOCOL_PATH.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def corpus_protocol_sha256(protocol: dict[str, Any]) -> str:
    """Only source/selection rules belong to a tokenizer-independent corpus."""
    return sha256_bytes(canonical_json({"dataset": protocol["dataset"], "selection": protocol["selection"]}).encode("utf-8"))


def tokenizer_identity(tokenizer: Any, model_id: str, revision: str) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    return {
        "source_model_id": model_id, "source_model_revision": revision,
        "class": type(tokenizer).__name__, "vocab_sha256": sha256_bytes(canonical_json(vocabulary).encode("utf-8")),
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
    }


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(args, text=True, capture_output=True)
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("git", "status", "--porcelain")
    return {"git_sha": run("git", "rev-parse", "HEAD"), "git_branch": run("git", "branch", "--show-current"), "git_dirty": bool(status)}


def article_rank(language: str, article_id: str, seed: int) -> str:
    # Delimiters make the selected corpus unambiguous even for unusual IDs.
    material = f"{DATASET_REVISION}\0{language}\0{article_id}\0{seed}".encode("utf-8")
    return sha256_bytes(material)


def dataset_rows(language: str) -> Iterator[dict[str, Any]]:
    return iter(load_dataset(
        DATASET_REPO, DATASET_CONFIGS[language], split="train",
        revision=DATASET_REVISION, streaming=True,
    ))


def text_cache_path(cache_dir: Path, language: str, seed: int, byte_budget: int) -> Path:
    """Local runtime location; intentionally excluded from canonical manifests."""
    return cache_dir / f"{language}-{seed}-{byte_budget}.jsonl"


def make_manifest(
    language: str, byte_budget: int, seed: int, cache_dir: Path,
    source_scan_documents: int | None, candidate_count: int, corpus_protocol_sha: str,
) -> tuple[dict[str, Any], Path]:
    """Select by global hash rank and materialize selected text outside Git."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    # A bounded max-heap retains exactly the globally best hash ranks seen so
    # far.  Its contents after a complete scan are therefore independent of
    # stream order while keeping memory bounded.
    candidate_heap: list[tuple[int, str, dict[str, Any]]] = []
    scanned = 0
    for row in dataset_rows(language):
        text = row["text"]
        if not isinstance(text, str) or not text:
            continue
        candidate = {
            "article_id": str(row["id"]), "title": row["title"], "text": text,
            "rank": article_rank(language, str(row["id"]), seed),
        }
        heap_item = (-int(candidate["rank"], 16), candidate["article_id"], candidate)
        if len(candidate_heap) < candidate_count:
            heapq.heappush(candidate_heap, heap_item)
        elif heap_item[0] > candidate_heap[0][0]:
            heapq.heapreplace(candidate_heap, heap_item)
        scanned += 1
        if source_scan_documents is not None and scanned >= source_scan_documents:
            break

    # With no scan cap this order is a property only of the pinned dataset,
    # language and seed, not the Dataset streaming order.
    candidates = [item[2] for item in candidate_heap]
    candidates.sort(key=lambda row: (row["rank"], row["article_id"]))
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for row in candidates:
        size = len(row["text"].encode("utf-8"))
        if selected and selected_bytes + size > byte_budget:
            continue
        if size > byte_budget and not selected:
            # A single article may be truncated only at UTF-8 character bounds.
            encoded = row["text"].encode("utf-8")[:byte_budget]
            row = {**row, "text": encoded.decode("utf-8", errors="ignore")}
            size = len(row["text"].encode("utf-8"))
        selected.append(row)
        selected_bytes += size
        if selected_bytes >= byte_budget:
            break
    if not selected:
        raise RuntimeError("No non-empty Wikipedia article fit the requested byte budget.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = text_cache_path(cache_dir, language, seed, byte_budget)
    with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(canonical_json({key: row[key] for key in ("article_id", "title", "text", "rank")}) + "\n")

    documents = []
    for order, row in enumerate(selected):
        raw = row["text"].encode("utf-8")
        documents.append({
            "order": order, "article_id": row["article_id"], "title": row["title"],
            "selection_rank_sha256": row["rank"], "utf8_bytes": len(raw),
            "characters": len(row["text"]), "text_sha256": sha256_bytes(raw),
        })
    manifest = {
        "schema_version": 1, "canonical": source_scan_documents is None,
        "noncanonical_reason": None if source_scan_documents is None else "capped_source_scan_smoke_only",
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION,
                    "config": DATASET_CONFIGS[language], "split": "train", "dump_date": "20231101"},
        "language": language, "selection": {"algorithm": "sha256(dataset_revision\\0language\\0article_id\\0seed)",
                       "seed": seed, "byte_budget": byte_budget, "candidate_count": candidate_count,
                       "source_scan_documents": source_scan_documents},
        "scanned_documents": scanned, "selected_raw_utf8_bytes": selected_bytes, "documents": documents,
        "corpus_protocol": {"source_and_selection_sha256": corpus_protocol_sha},
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest, cache_path


def load_cached_documents(cache_path: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
    expected = manifest["documents"]
    if len(rows) != len(expected):
        raise RuntimeError("Local text cache does not match manifest document count.")
    for row, document in zip(rows, expected):
        raw = row["text"].encode("utf-8")
        if (str(row["article_id"]) != document["article_id"] or row.get("title") != document["title"]
                or row.get("rank") != document["selection_rank_sha256"]
                or len(raw) != document["utf8_bytes"] or len(row["text"]) != document["characters"]
                or sha256_bytes(raw) != document["text_sha256"]):
            raise RuntimeError("Local text cache content does not match manifest.")
    return rows


def tokenized_document(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) != len(offsets):
        raise RuntimeError("Tokenizer did not return an offset for every token.")
    return ids, [tuple(offset) for offset in offsets]


def utf8_bytes_for_spans(text: str, spans: list[tuple[int, int]]) -> int:
    """Count each source character span at most once for tokenizer offsets."""
    merged: list[list[int]] = []
    for start, end in sorted((start, end) for start, end in spans if end > start):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(len(text[start:end].encode("utf-8")) for start, end in merged)


def make_chunks(tokenizer: Any, documents: list[dict[str, str]], max_length: int) -> list[dict[str, Any]]:
    boundary = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if boundary is None:
        raise RuntimeError("This protocol requires a tokenizer BOS or EOS token for document boundaries.")
    chunks: list[dict[str, Any]] = []
    for document_order, document in enumerate(documents):
        ids, offsets = tokenized_document(tokenizer, document["text"])
        cursor = 0
        while cursor < len(ids):
            target_end = min(cursor + max_length - 1, len(ids))
            targets = ids[cursor:target_end]
            context = [boundary] if cursor == 0 else [ids[cursor - 1]]
            token_bytes = b"".join(int(token).to_bytes(4, "little", signed=False) for token in context + targets)
            chunks.append({"document_order": document_order, "target_start": cursor, "target_end": target_end,
                           "target_token_count": len(targets),
                           "target_utf8_bytes": utf8_bytes_for_spans(document["text"], offsets[cursor:target_end]),
                           "input_token_ids_sha256": sha256_bytes(token_bytes)})
            cursor = target_end
    return chunks


def validate_manifest(
    manifest_path: Path, language: str, protocol: dict[str, Any], corpus_protocol_sha: str, cache_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise RuntimeError(f"{manifest_path}: manifest SHA-256 mismatch.")
    expected_dataset = {"repo": DATASET_REPO, "revision": DATASET_REVISION,
                        "config": DATASET_CONFIGS[language], "split": "train", "dump_date": "20231101"}
    if not manifest.get("canonical") or manifest.get("dataset") != expected_dataset or manifest.get("language") != language:
        raise RuntimeError(f"{manifest_path}: not the required canonical {language} Wikipedia manifest.")
    expected_selection = {"algorithm": "sha256(dataset_revision\\0language\\0article_id\\0seed)",
                          "seed": protocol["selection"]["seed"],
                          "byte_budget": protocol["selection"]["raw_utf8_byte_budget_per_language"],
                          "candidate_count": protocol["selection"]["candidate_count"], "source_scan_documents": None}
    if manifest.get("selection") != expected_selection:
        raise RuntimeError(f"{manifest_path}: selection is incompatible with versioned protocol.")
    if manifest.get("corpus_protocol") != {"source_and_selection_sha256": corpus_protocol_sha}:
        raise RuntimeError(f"{manifest_path}: source/selection protocol is incompatible.")
    documents_manifest = manifest.get("documents")
    if (not isinstance(documents_manifest, list) or not documents_manifest
            or [item.get("order") for item in documents_manifest] != list(range(len(documents_manifest)))
            or sum(item.get("utf8_bytes", -1) for item in documents_manifest) != manifest.get("selected_raw_utf8_bytes")):
        raise RuntimeError(f"{manifest_path}: document order or raw UTF-8 byte accounting is invalid.")
    cache_path = text_cache_path(cache_dir, language, expected_selection["seed"], expected_selection["byte_budget"])
    if not cache_path.is_file():
        raise RuntimeError(f"{manifest_path}: selected text cache is unavailable at {cache_path}; refusing to recreate the corpus.")
    documents = load_cached_documents(cache_path, manifest)
    return manifest, documents


@torch.inference_mode()
def score_documents(model: Any, tokenizer: Any, documents: list[dict[str, str]], max_length: int, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    boundary = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if boundary is None:
        raise RuntimeError("This protocol requires a tokenizer BOS or EOS token for document boundaries.")
    total_nll = 0.0
    scored_tokens = 0
    scored_bytes = 0
    chunks: list[dict[str, Any]] = []
    for document_order, document in enumerate(documents):
        ids, offsets = tokenized_document(tokenizer, document["text"])
        if not ids:
            continue
        cursor = 0
        while cursor < len(ids):
            target_end = min(cursor + max_length - 1, len(ids))
            targets = ids[cursor:target_end]
            target_offsets = offsets[cursor:target_end]
            # Qwen has no BOS token; its EOS token is a documented, unscored
            # boundary context.  Documents are never concatenated.
            context = [boundary] if cursor == 0 else [ids[cursor - 1]]
            input_ids = torch.tensor([context + targets], device=device)
            logits = model(input_ids=input_ids).logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction="sum")
            bytes_here = utf8_bytes_for_spans(document["text"], target_offsets)
            total_nll += float(nll.item())
            scored_tokens += len(targets)
            scored_bytes += bytes_here
            token_bytes = b"".join(int(token).to_bytes(4, "little", signed=False) for token in context + targets)
            chunks.append({"document_order": document_order, "target_start": cursor, "target_end": target_end,
                           "target_token_count": len(targets), "target_utf8_bytes": bytes_here,
                           "input_token_ids_sha256": sha256_bytes(token_bytes)})
            cursor = target_end
    if not scored_tokens or not scored_bytes:
        raise RuntimeError("No scored text tokens/bytes; cannot calculate PPL or BPB.")
    return ({"document_boundary_token_id": boundary, "document_boundary_token_kind": "bos" if tokenizer.bos_token_id is not None else "eos",
             "total_nll_nats": total_nll, "scored_tokens": scored_tokens, "scored_target_utf8_bytes": scored_bytes,
             "ppl": math.exp(total_nll / scored_tokens), "bits_per_byte": total_nll / math.log(2) / scored_bytes}, chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--languages", nargs="+", choices=tuple(DATASET_CONFIGS), default=tuple(DATASET_CONFIGS))
    parser.add_argument("--byte-budget", type=int, default=DEFAULT_BYTE_BUDGET, help="Equal raw UTF-8 budget per language.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--backend", choices=("auto", "cuda", "xpu", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--source-scan-documents", type=int, help="Smoke only: cap candidate scan; resulting manifest is non-canonical.")
    parser.add_argument("--selection-candidate-count", type=int, default=512,
                        help="Fixed number of globally lowest hash ranks retained while scanning a snapshot.")
    parser.add_argument("--output-dir", type=Path, default=Path("pretrain_benchmarks/results/multilingual_wikipedia"))
    parser.add_argument("--text-cache-dir", type=Path,
                        default=Path("pretrain_benchmarks/results/multilingual_wikipedia/selected_text_cache"),
                        help="Ignored local selected-text cache; never recorded in a canonical manifest.")
    parser.add_argument("--versioned-output-dir", type=Path, help="Canonical full-run outputs only; for baseline_results/.")
    parser.add_argument("--build-manifests-dir", type=Path,
                        help="Build canonical manifests/text cache once, without model forward evaluation.")
    parser.add_argument("--manifest-dir", type=Path,
                        help="Evaluate strictly from existing canonical <language>.manifest.json files.")
    parser.add_argument("--verify-manifests", action="store_true",
                        help="Validate existing corpus manifests and text cache without model forward evaluation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.byte_budget <= 0 or args.max_length < 2:
        raise ValueError("--byte-budget must be positive and --max-length must be at least 2.")
    if sum(value is not None for value in (args.build_manifests_dir, args.manifest_dir)) > 1:
        raise ValueError("Use either --build-manifests-dir or --manifest-dir, not both.")
    if args.verify_manifests and args.manifest_dir is None:
        raise ValueError("--verify-manifests requires --manifest-dir.")
    if args.versioned_output_dir and args.source_scan_documents is not None:
        raise ValueError("A capped smoke manifest cannot be written as a versioned baseline output.")
    protocol, protocol_sha = protocol_config()
    corpus_protocol_sha = corpus_protocol_sha256(protocol)
    if args.build_manifests_dir and (args.seed != protocol["selection"]["seed"] or args.byte_budget != protocol["selection"]["raw_utf8_byte_budget_per_language"] or args.selection_candidate_count != protocol["selection"]["candidate_count"]):
        raise ValueError("Arguments differ from multilingual_wikipedia_protocol.json; update and review the protocol first.")
    if args.build_manifests_dir and args.source_scan_documents is not None:
        raise ValueError("Canonical manifests cannot use --source-scan-documents.")
    if args.build_manifests_dir:
        args.build_manifests_dir.mkdir(parents=True, exist_ok=False)
        cache_dir = args.text_cache_dir
        for language in args.languages:
            manifest, cache_path = make_manifest(language, args.byte_budget, args.seed, cache_dir,
                                                 None, args.selection_candidate_count, corpus_protocol_sha)
            # Validate text hashes immediately, before writing the committed artifact.
            load_cached_documents(cache_path, manifest)
            (args.build_manifests_dir / f"{language}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Canonical manifests: {args.build_manifests_dir}")
        return 0
    if args.manifest_dir:
        manifests_and_documents = [validate_manifest(args.manifest_dir / f"{language}.manifest.json", language,
                                                      protocol, corpus_protocol_sha, args.text_cache_dir)
                                   for language in args.languages]
        if args.verify_manifests:
            print(f"Verified canonical manifests: {args.manifest_dir}")
            return 0
    else:
        manifests_and_documents = []
    revision = model_revision(args.model)
    if revision is None:
        raise RuntimeError("Multilingual baseline evaluation requires an exact resolved model revision SHA.")
    device, backend, automatic_dtype, vram_gb = select_backend(args.backend)
    dtype = automatic_dtype if args.dtype == "auto" else args.dtype
    if backend == "cpu" and dtype != "float32":
        raise ValueError("CPU runs require float32.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_dir = args.text_cache_dir
    evaluator_sha = sha256_bytes(Path(__file__).read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    tokenizer_info = tokenizer_identity(tokenizer, args.model, revision)
    run_dir = (args.versioned_output_dir or args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=getattr(torch, dtype), trust_remote_code=False).to(device).eval()
    common = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model_id": args.model,
              "model_revision": revision, "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
              "tokenizer_name_or_path": tokenizer.name_or_path, "tokenizer_vocab_size": tokenizer.vocab_size,
              "model_config": model.config.to_dict(), "dataset_repo": DATASET_REPO, "dataset_revision": DATASET_REVISION,
              "seed": args.seed, "max_length": args.max_length, "dtype": dtype, "backend": backend, "device": device,
              "accelerator_name": accelerator_name(backend), "vram_gb": vram_gb, "batch_size_policy": "sequential_single_chunk",
              "evaluator_version": EVALUATOR_VERSION, "evaluator_sha256": evaluator_sha, "git": git_metadata(),
              "package_versions": {item: package_version(item) for item in REPRODUCIBILITY_PACKAGES},
              "python": platform.python_version(), "source_scan_documents": args.source_scan_documents,
              "manifest_reused": bool(args.manifest_dir), "manifest_dir": str(args.manifest_dir) if args.manifest_dir else None,
              "corpus_protocol_sha256": corpus_protocol_sha, "evaluation_protocol_config_sha256": protocol_sha,
              "tokenizer": tokenizer_info,
              "command": sys.argv}
    results = []
    for index, language in enumerate(args.languages):
        selection_started = time.perf_counter()
        if args.manifest_dir:
            manifest, documents = manifests_and_documents[index]
        else:
            manifest, cache_path = make_manifest(language, args.byte_budget, args.seed, cache_dir,
                                                 args.source_scan_documents, args.selection_candidate_count,
                                                 corpus_protocol_sha)
            documents = load_cached_documents(cache_path, manifest)
        selection_seconds = time.perf_counter() - selection_started
        scoring_started = time.perf_counter()
        metrics, chunks = score_documents(model, tokenizer, documents, args.max_length, device)
        scoring_seconds = time.perf_counter() - scoring_started
        if args.manifest_dir:
            manifest_path = args.manifest_dir / f"{language}.manifest.json"
        else:
            manifest_path = run_dir / f"{language}.manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        chunk_sha = sha256_bytes(canonical_json(chunks).encode("utf-8"))
        results.append({"language": language, "dataset_config": DATASET_CONFIGS[language],
                        "corpus_manifest": str(manifest_path), "corpus_manifest_sha256": manifest["manifest_sha256"],
                        "selection_seconds": selection_seconds, "scoring_seconds": scoring_seconds,
                        "scored_tokens_per_second": metrics["scored_tokens"] / scoring_seconds,
                        "tokenization_and_chunking": {"max_length": args.max_length, "stride": args.max_length - 1,
                                                       "chunk_manifest_sha256": chunk_sha, "chunks": chunks}, **metrics})
        print(f"{language}: BPB={metrics['bits_per_byte']:.6f}, PPL={metrics['ppl']:.6f}, tokens={metrics['scored_tokens']}")
    result = {**common, "canonical": args.source_scan_documents is None, "results": results}
    result["result_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    environment = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    (run_dir / "environment.txt").write_text(environment.stdout, encoding="utf-8")
    print(f"Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
