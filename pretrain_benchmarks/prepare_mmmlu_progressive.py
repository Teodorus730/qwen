"""Create and verify the canonical progressive MMMLU selection manifest.

The first stage is a deterministic 5% allocation (702 rows per locale).  Its
rows are the prefix of a stable per-stratum SHA ordering, so the complementary
stage can later be scored without re-evaluating this prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", str(Path("pretrain_benchmarks/.hf_cache").resolve()))

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET = "openai/MMMLU"
REVISION = "325a01dc3e173cac1578df94120499aaca2e2504"
MODEL = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SEED = 42
STAGE1_PER_LOCALE = 702
LOCALES = (
    "ar_xy", "bn_bd", "de_de", "es_la", "fr_fr", "hi_in", "id_id",
    "it_it", "ja_jp", "ko_kr", "pt_br", "sw_ke", "yo_ng", "zh_cn",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_subject(value: str) -> str:
    for marker in ("_test.csv", "_test-"):
        if marker in value:
            return value.split(marker, 1)[0]
    return value


def rank(locale: str, subject: str, index: int) -> str:
    return digest(f"{REVISION}\0{SEED}\0{locale}\0{subject}\0{index}")


def allocate(counts: dict[str, int]) -> dict[str, int]:
    total = sum(counts.values())
    if total != 14042:
        raise ValueError(f"Unexpected locale total {total}; expected 14042")
    floor = {subject: count * STAGE1_PER_LOCALE // total for subject, count in counts.items()}
    remaining = STAGE1_PER_LOCALE - sum(floor.values())
    remainders = sorted(
        counts,
        key=lambda subject: (-(counts[subject] * STAGE1_PER_LOCALE % total), subject),
    )
    for subject in remainders[:remaining]:
        floor[subject] += 1
    return floor


def text_hash(row: dict[str, Any]) -> str:
    fields = {key: row[key] for key in ("Question", "A", "B", "C", "D", "Answer", "Subject")}
    return digest(canonical_json(fields))


def task_name(locale: str, subject: str) -> str:
    return f"mmmlu_{locale}_{subject}"


def write_manifest(destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    strata: list[dict[str, Any]] = []
    samples: dict[str, list[int]] = {}
    allocation_reference: dict[str, int] | None = None
    for locale in LOCALES:
        dataset = load_dataset(DATASET, locale.upper(), split="test", revision=REVISION)
        by_subject: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(dataset):
            by_subject[normalized_subject(row["Subject"])].append((index, dict(row)))
        counts = {subject: len(rows) for subject, rows in sorted(by_subject.items())}
        allocation = allocate(counts)
        if allocation_reference is None:
            allocation_reference = allocation
        elif allocation != allocation_reference:
            raise ValueError(f"Subject allocation differs in {locale}")
        for subject, rows in sorted(by_subject.items()):
            ordered = sorted(rows, key=lambda item: rank(locale, subject, item[0]))
            selected = ordered[: allocation[subject]]
            selected_indices = [index for index, _ in selected]
            # lm-eval applies a task's process_docs subject filter before its
            # --samples selection.  Keep source indices in the manifest, but
            # write the corresponding post-filter positions for lm-eval.
            filtered_position = {source_index: position for position, (source_index, _) in enumerate(rows)}
            samples[task_name(locale, subject)] = [filtered_position[index] for index in selected_indices]
            full_order = [index for index, _ in ordered]
            strata.append(
                {
                    "locale": locale,
                    "subject": subject,
                    "source_count": len(rows),
                    "stage1_count": len(selected),
                    "selected_source_indices": selected_indices,
                    "selected_text_sha256": [text_hash(row) for _, row in selected],
                    "full_order_sha256": digest(canonical_json(full_order)),
                }
            )
    if len(samples) != 798 or sum(map(len, samples.values())) != 9828:
        raise ValueError("Unexpected MMMLU stage-1 selection size")
    payload = {
        "schema_version": 1,
        "dataset": {"path": DATASET, "revision": REVISION, "split": "test"},
        "selection": {
            "seed": SEED,
            "algorithm": "sha256(revision + NUL + seed + NUL + locale + NUL + subject + NUL + source_index)",
            "stage1_per_locale": STAGE1_PER_LOCALE,
            "allocation": "proportional floor plus largest remainder; ties by subject id",
            "stage1_total": 9828,
            "full_total": 196588,
        },
        "strata": strata,
    }
    manifest_without_hash = canonical_json(payload)
    payload["manifest_sha256"] = digest(manifest_without_hash)
    manifest_path = destination / "mmmlu_progressive_manifest.json"
    samples_path = destination / "mmmlu_stage1_samples.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(samples, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest_path, "samples": samples_path}


def verify_manifest(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_hash = payload.pop("manifest_sha256")
    if digest(canonical_json(payload)) != stored_hash:
        raise ValueError("Manifest SHA-256 mismatch")
    seen: set[tuple[str, str, int]] = set()
    stage1_total = 0
    full_total = 0
    for stratum in payload["strata"]:
        key_prefix = (stratum["locale"], stratum["subject"])
        indices = stratum["selected_source_indices"]
        if len(indices) != stratum["stage1_count"] or len(indices) != len(set(indices)):
            raise ValueError(f"Invalid selected indices for {key_prefix}")
        for index in indices:
            key = (*key_prefix, index)
            if key in seen:
                raise ValueError(f"Duplicate selected index: {key}")
            seen.add(key)
        stage1_total += len(indices)
        full_total += stratum["source_count"]
    if stage1_total != payload["selection"]["stage1_total"] or full_total != payload["selection"]["full_total"]:
        raise ValueError("Manifest totals mismatch")


def audit_prompts(manifest_path: Path, destination: Path, selected: dict[tuple[str, str], list[int]], stage: str) -> Path:
    """Tokenize the exact 0-shot lm-eval prompt text for a selected stage."""
    import lm_eval
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval import utils

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_root = Path(inspect.getfile(lm_eval)).parent / "tasks" / "openai-mmmlu" / "default"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    lengths: list[tuple[int, str, str, int, str]] = []
    for locale in LOCALES:
        dataset = load_dataset(DATASET, locale.upper(), split="test", revision=REVISION)
        rows_by_subject: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for index, row in enumerate(dataset):
            rows_by_subject[normalized_subject(row["Subject"])][index] = dict(row)
        for subject in sorted({subject for found_locale, subject in selected if found_locale == locale}):
            config = load_yaml(task_root / f"mmmlu_{locale}_{subject}.yaml", resolve_func=False, recursive=True)
            description = config["description"]
            template = config["doc_to_text"]
            for index in selected[(locale, subject)]:
                row = rows_by_subject[subject][index]
                prompt = description + utils.apply_template(template, row)
                length = len(tokenizer.encode(prompt))
                lengths.append((length, locale, subject, index, text_hash(row)))
    values = sorted(item[0] for item in lengths)
    def percentile(p: float) -> int:
        return values[round((len(values) - 1) * p)]
    longest = max(lengths)
    report = {
        "schema_version": 1,
        "manifest_sha256": payload["manifest_sha256"],
        "model_id": MODEL,
        "model_revision": MODEL_REVISION,
        "prompt_protocol": "lm-eval 0.4.12 MMMLU task description plus doc_to_text, native 0-shot",
        "tokenizer_encode": "AutoTokenizer.encode default special-token behavior, matching hf causal backend default",
        "sample_count": len(lengths),
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": longest[0],
        "max_sample": {"locale": longest[1], "subject": longest[2], "source_index": longest[3], "text_sha256": longest[4]},
        "counts": {
            ">1024": sum(value > 1024 for value in values),
            ">2048": sum(value > 2048 for value in values),
            ">4096": sum(value > 4096 for value in values),
        },
    }
    output = destination / f"mmmlu_{stage}_prompt_lengths.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def prepare_stage2(manifest_path: Path, destination: Path) -> dict[str, Path]:
    """Create the exact complement of committed Stage 1 in lm-eval indices."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(manifest_path)
    if payload["dataset"] != {"path": DATASET, "revision": REVISION, "split": "test"}:
        raise ValueError("Manifest dataset identity does not match the canonical MMMLU source.")
    manifest_strata = {(x["locale"], x["subject"]): x for x in payload["strata"]}
    if len(manifest_strata) != 798:
        raise ValueError("Manifest does not contain all 798 locale-subject strata.")
    stage2_samples: dict[str, list[int]] = {}
    stage2_source: dict[tuple[str, str], list[int]] = {}
    stage1_total = stage2_total = full_total = 0
    for locale in LOCALES:
        dataset = load_dataset(DATASET, locale.upper(), split="test", revision=REVISION)
        by_subject: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(dataset):
            by_subject[normalized_subject(row["Subject"])].append((index, dict(row)))
        if len(by_subject) != 57:
            raise ValueError(f"Unexpected subject count in {locale}: {len(by_subject)}")
        for subject, rows in sorted(by_subject.items()):
            key = (locale, subject)
            entry = manifest_strata.get(key)
            if entry is None:
                raise ValueError(f"Manifest missing {key}")
            ordered = sorted(rows, key=lambda item: rank(locale, subject, item[0]))
            full_order = [index for index, _ in ordered]
            if digest(canonical_json(full_order)) != entry["full_order_sha256"]:
                raise ValueError(f"Full ordering mismatch for {key}")
            stage1 = entry["selected_source_indices"]
            if full_order[: len(stage1)] != stage1:
                raise ValueError(f"Stage-1 prefix mismatch for {key}")
            stage2 = full_order[len(stage1) :]
            if set(stage1) & set(stage2) or set(stage1) | set(stage2) != set(full_order):
                raise ValueError(f"Complement coverage failure for {key}")
            filtered_position = {source_index: position for position, (source_index, _) in enumerate(rows)}
            stage2_samples[task_name(locale, subject)] = [filtered_position[index] for index in stage2]
            stage2_source[key] = stage2
            stage1_total += len(stage1)
            stage2_total += len(stage2)
            full_total += len(full_order)
    if (stage1_total, stage2_total, full_total) != (9828, 186760, 196588):
        raise ValueError("Unexpected Stage-2 totals")
    destination.mkdir(parents=True, exist_ok=True)
    samples_path = destination / "mmmlu_stage2_samples.json"
    samples_path.write_text(json.dumps(stage2_samples, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    audit_path = audit_prompts(manifest_path, destination, stage2_source, "stage2")
    return {"samples": samples_path, "audit": audit_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--audit-prompts", action="store_true")
    parser.add_argument("--prepare-stage2", type=Path, metavar="MANIFEST")
    args = parser.parse_args()
    if args.verify:
        verify_manifest(args.verify)
        print(f"Verified: {args.verify}")
        return
    if args.prepare_stage2:
        paths = prepare_stage2(args.prepare_stage2, args.output_dir)
        print(f"Stage-2 samples: {paths['samples']}")
        print(f"Stage-2 prompt-length audit: {paths['audit']}")
        return
    paths = write_manifest(args.output_dir)
    verify_manifest(paths["manifest"])
    print(f"Manifest: {paths['manifest']}")
    print(f"Stage-1 samples: {paths['samples']}")
    if args.audit_prompts:
        selected = {
            (entry["locale"], entry["subject"]): entry["selected_source_indices"]
            for entry in json.loads(paths["manifest"].read_text(encoding="utf-8"))["strata"]
        }
        print(f"Prompt-length audit: {audit_prompts(paths['manifest'], args.output_dir, selected, 'stage1')}")


if __name__ == "__main__":
    main()
