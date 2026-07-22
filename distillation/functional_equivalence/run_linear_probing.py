"""Run the original and basis-aware linear-probing checks from stage 4.

The auxiliary task is next-word Universal POS prediction on UD English EWT.
For a word whose first subtoken starts at position j, the probe reads the
hidden state at j-1, so the target word itself is not visible to the causal LM.

For every requested layer and Student the script reports three evaluations:

1. original Qwen test: freeze the Teacher probe and apply it directly to raw
   Student activations;
2. separate-probe control: train the same linear classifier on Student
   activations using the identical train/test examples;
3. aligned cross-probe: fit scale + orthogonal Procrustes on train activations,
   then apply the frozen Teacher probe to aligned Student test activations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoTokenizer

from evaluate_outputs import atomic_json_dump, load_model, resolve_relative, seed_everything


SCRIPT_DIR = Path(__file__).resolve().parent
UPOS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
UPOS_TO_ID = {label: index for index, label in enumerate(UPOS)}
UD_SOURCE = "https://github.com/UniversalDependencies/UD_English-EWT"


@dataclass
class ProbeRecord:
    input_ids: list[int]
    positions: list[int]
    labels: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--output", default=str(
        SCRIPT_DIR / "outputs" / "linear_probe_results.json"
    ))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_conllu(path: Path) -> Iterable[tuple[list[str], list[str], list[str]]]:
    words: list[str] = []
    tags: list[str] = []
    misc: list[str] = []
    has_multiword = False
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if not line:
                if words and not has_multiword:
                    yield words, tags, misc
                words, tags, misc = [], [], []
                has_multiword = False
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            token_id = fields[0]
            if "-" in token_id:
                has_multiword = True
                continue
            if "." in token_id:
                continue
            upos = fields[3]
            if upos not in UPOS_TO_ID:
                continue
            words.append(fields[1])
            tags.append(upos)
            misc.append(fields[9])
    if words and not has_multiword:
        yield words, tags, misc


def reconstruct_sentence(
    words: list[str], misc: list[str]
) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word, metadata in zip(words, misc):
        start = cursor
        parts.append(word)
        cursor += len(word)
        spans.append((start, cursor))
        if "SpaceAfter=No" not in metadata:
            parts.append(" ")
            cursor += 1
    return "".join(parts).rstrip(), spans


def prepare_records(
    tokenizer,
    path: Path,
    max_length: int,
    target_limit: int,
) -> tuple[list[ProbeRecord], dict[str, Any]]:
    records: list[ProbeRecord] = []
    total_targets = 0
    skipped_alignment = 0
    sentence_count = 0
    class_counts = np.zeros(len(UPOS), dtype=np.int64)
    for words, tags, misc in iter_conllu(path):
        text, word_spans = reconstruct_sentence(words, misc)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        positions: list[int] = []
        labels: list[int] = []
        # Skip word zero: without an explicit BOS there is no preceding state.
        for word_index in range(1, len(words)):
            word_start, word_end = word_spans[word_index]
            first_subtoken = None
            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_end <= word_start:
                    continue
                if token_start >= word_end:
                    break
                if token_start < word_end and token_end > word_start:
                    first_subtoken = token_index
                    break
            if first_subtoken is None or first_subtoken <= 0:
                skipped_alignment += 1
                continue
            label = UPOS_TO_ID[tags[word_index]]
            positions.append(first_subtoken - 1)
            labels.append(label)
            class_counts[label] += 1
            total_targets += 1
            if total_targets >= target_limit:
                break
        if positions:
            records.append(ProbeRecord(input_ids, positions, labels))
            sentence_count += 1
        if total_targets >= target_limit:
            break
    if total_targets < target_limit:
        raise RuntimeError(
            f"Only prepared {total_targets}/{target_limit} POS targets from {path}"
        )
    # The last sentence may overshoot only if target_limit changes mid-loop;
    # trim defensively while preserving sentence boundaries as much as possible.
    excess = total_targets - target_limit
    if excess:
        records[-1].positions = records[-1].positions[:-excess]
        removed = records[-1].labels[-excess:]
        records[-1].labels = records[-1].labels[:-excess]
        for label in removed:
            class_counts[label] -= 1
        total_targets = target_limit
    return records, {
        "sentences": sentence_count,
        "targets": total_targets,
        "skipped_alignment": skipped_alignment,
        "class_counts": {label: int(class_counts[index]) for index, label in enumerate(UPOS)},
    }


def batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@torch.inference_mode()
def extract_features(
    model,
    records: list[ProbeRecord],
    layers: list[int],
    pad_token_id: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    collected_labels: list[torch.Tensor] = []
    for batch_index, batch in enumerate(batches(records, batch_size)):
        max_length = max(len(record.input_ids) for record in batch)
        input_ids = torch.full(
            (len(batch), max_length), pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, record in enumerate(batch):
            length = len(record.input_ids)
            input_ids[row, :length] = torch.tensor(record.input_ids, device=device)
            attention_mask[row, :length] = 1
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        for layer in layers:
            row_features = []
            for row, record in enumerate(batch):
                positions = torch.tensor(record.positions, device=device)
                row_features.append(output.hidden_states[layer][row, positions].float().cpu())
            collected[layer].append(torch.cat(row_features, dim=0))
        collected_labels.append(torch.tensor(
            [label for record in batch for label in record.labels], dtype=torch.long
        ))
        del output, input_ids, attention_mask
        print(f"    feature batches: {batch_index + 1}", end="\r", flush=True)
    print("", flush=True)
    return (
        {layer: torch.cat(values, dim=0) for layer, values in collected.items()},
        torch.cat(collected_labels),
    )


def normalize_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-4)
    return mean, std


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    mean, std = normalize_fit(train_x)
    x_train = ((train_x - mean) / std).to(device)
    y_train = train_y.to(device)
    classifier = nn.Linear(train_x.shape[1], len(UPOS)).to(device)
    torch.manual_seed(seed)
    nn.init.zeros_(classifier.bias)
    nn.init.normal_(classifier.weight, std=0.01)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(x_train), generator=generator)
        for index in permutation.split(batch_size):
            index = index.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(x_train[index]), y_train[index])
            loss.backward()
            optimizer.step()
    state = {
        "weight": classifier.weight.detach().cpu(),
        "bias": classifier.bias.detach().cpu(),
        "mean": mean,
        "std": std,
    }
    metrics = evaluate_probe(state, test_x, test_y, device)
    del classifier, optimizer, x_train, y_train
    return state, metrics


@torch.inference_mode()
def evaluate_probe(
    state: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    predictions: list[torch.Tensor] = []
    weight = state["weight"].to(device)
    bias = state["bias"].to(device)
    mean = state["mean"].to(device)
    std = state["std"].to(device)
    for chunk in x.split(1024):
        normalized = (chunk.to(device) - mean) / std
        predictions.append((normalized @ weight.T + bias).argmax(-1).cpu())
    prediction = torch.cat(predictions)
    y_np = y.numpy()
    pred_np = prediction.numpy()
    matrix = confusion_matrix(y_np, pred_np, labels=list(range(len(UPOS))))
    per_class = {}
    for index, label in enumerate(UPOS):
        denom = matrix[index].sum()
        per_class[label] = float(matrix[index, index] / denom) if denom else None
    return {
        "accuracy": float((prediction == y).float().mean().item()),
        "macro_f1": float(f1_score(y_np, pred_np, labels=list(range(len(UPOS))), average="macro", zero_division=0)),
        "per_class_accuracy": per_class,
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def fit_scaled_procrustes(
    student_train: torch.Tensor,
    teacher_train: torch.Tensor,
    max_rows: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n = min(len(student_train), max_rows)
    indices = torch.linspace(0, len(student_train) - 1, steps=n).long()
    student = student_train[indices].to(device)
    teacher = teacher_train[indices].to(device)
    student_mean = student.mean(0, keepdim=True)
    teacher_mean = teacher.mean(0, keepdim=True)
    sx = student - student_mean
    tx = teacher - teacher_mean
    u, singular, vh = torch.linalg.svd(sx.T @ tx, full_matrices=False)
    rotation = u @ vh
    scale = singular.sum() / sx.square().sum().clamp_min(1e-12)
    return {
        "student_mean": student_mean.cpu(),
        "teacher_mean": teacher_mean.cpu(),
        "rotation": rotation.cpu(),
        "scale": scale.cpu(),
    }


@torch.inference_mode()
def apply_alignment(
    x: torch.Tensor,
    alignment: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    student_mean = alignment["student_mean"].to(device)
    teacher_mean = alignment["teacher_mean"].to(device)
    rotation = alignment["rotation"].to(device)
    scale = alignment["scale"].to(device)
    for chunk in x.split(1024):
        aligned = (chunk.to(device) - student_mean) @ rotation * scale + teacher_mean
        outputs.append(aligned.cpu())
    return torch.cat(outputs)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    probe_cfg = cfg["linear_probe"]
    runtime = cfg["runtime"]
    seed = int(runtime["seed"])
    seed_everything(seed)
    device = torch.device(runtime.get("device", "cuda"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[runtime.get("dtype", "bfloat16")]
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(runtime.get("cuda_memory_fraction", 1.0)))

    train_path = resolve_relative(probe_cfg["train_conllu"], config_dir)
    test_path = resolve_relative(probe_cfg["test_conllu"], config_dir)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "UD English EWT files are missing. Download the official train/test "
            f"CoNLL-U files from {UD_SOURCE} into {train_path.parent}."
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"], cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[data] tokenizing UD English EWT", flush=True)
    train_records, train_stats = prepare_records(
        tokenizer, train_path, int(probe_cfg["max_length"]), int(probe_cfg["train_targets"])
    )
    test_records, test_stats = prepare_records(
        tokenizer, test_path, int(probe_cfg["max_length"]), int(probe_cfg["test_targets"])
    )
    layers = [int(layer) for layer in probe_cfg["layers"]]

    print("[teacher] extracting features", flush=True)
    teacher = load_model(cfg["teacher_model"], device, dtype, cache_dir=args.cache_dir)
    teacher_train, teacher_train_y = extract_features(
        teacher, train_records, layers, tokenizer.pad_token_id,
        int(probe_cfg["extraction_batch_size"]), device,
    )
    teacher_test, teacher_test_y = extract_features(
        teacher, test_records, layers, tokenizer.pad_token_id,
        int(probe_cfg["extraction_batch_size"]), device,
    )
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    teacher_states: dict[int, dict[str, torch.Tensor]] = {}
    teacher_metrics: dict[str, Any] = {}
    for layer in layers:
        print(f"[teacher] training probe layer {layer}", flush=True)
        state, metrics = train_probe(
            teacher_train[layer], teacher_train_y, teacher_test[layer], teacher_test_y,
            device, int(probe_cfg["epochs"]), int(probe_cfg["probe_batch_size"]),
            float(probe_cfg["learning_rate"]), float(probe_cfg["weight_decay"]),
            seed + layer,
        )
        teacher_states[layer] = state
        teacher_metrics[str(layer)] = metrics

    output_path = Path(args.output).resolve()
    if args.resume and output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        result = {
            "definition": {
                "task": "Predict the UPOS tag of the next word from the hidden state immediately before its first subtoken.",
                "source": UD_SOURCE,
                "classes": UPOS,
                "layers": layers,
                "train_targets": int(probe_cfg["train_targets"]),
                "test_targets": int(probe_cfg["test_targets"]),
                "train_data_sha256": sha256_file(train_path),
                "test_data_sha256": sha256_file(test_path),
                "train_stats": train_stats,
                "test_stats": test_stats,
            },
            "teacher": {"layers": teacher_metrics},
            "models": {},
        }

    selected = set(args.only or [])
    model_specs = [spec for spec in cfg["models"] if not selected or spec["id"] in selected]
    for model_index, spec in enumerate(model_specs):
        model_id = spec["id"]
        if args.resume and model_id in result["models"]:
            print(f"[resume] {model_id}", flush=True)
            continue
        print(f"[student {model_index + 1}/{len(model_specs)}] {model_id}", flush=True)
        student = load_model(resolve_relative(spec["path"], config_dir), device, dtype)
        student_train, student_train_y = extract_features(
            student, train_records, layers, tokenizer.pad_token_id,
            int(probe_cfg["extraction_batch_size"]), device,
        )
        student_test, student_test_y = extract_features(
            student, test_records, layers, tokenizer.pad_token_id,
            int(probe_cfg["extraction_batch_size"]), device,
        )
        assert torch.equal(student_train_y, teacher_train_y)
        assert torch.equal(student_test_y, teacher_test_y)
        layer_results = {}
        for layer in layers:
            print(f"  [probe] layer {layer}", flush=True)
            raw_cross = evaluate_probe(
                teacher_states[layer], student_test[layer], student_test_y, device
            )
            _, own_metrics = train_probe(
                student_train[layer], student_train_y, student_test[layer], student_test_y,
                device, int(probe_cfg["epochs"]), int(probe_cfg["probe_batch_size"]),
                float(probe_cfg["learning_rate"]), float(probe_cfg["weight_decay"]),
                seed + model_index * 100 + layer,
            )
            alignment = fit_scaled_procrustes(
                student_train[layer], teacher_train[layer],
                int(probe_cfg["procrustes_train_rows"]), device,
            )
            aligned_test = apply_alignment(student_test[layer], alignment, device)
            aligned_cross = evaluate_probe(
                teacher_states[layer], aligned_test, student_test_y, device
            )
            teacher_accuracy = teacher_metrics[str(layer)]["accuracy"]
            layer_results[str(layer)] = {
                "teacher_probe_on_teacher": teacher_metrics[str(layer)],
                "qwen_frozen_teacher_probe_on_raw_student": raw_cross,
                "separate_student_probe": own_metrics,
                "teacher_probe_on_procrustes_aligned_student": aligned_cross,
                "accuracy_drop_qwen_raw": teacher_accuracy - raw_cross["accuracy"],
                "accuracy_drop_separate_probe": teacher_accuracy - own_metrics["accuracy"],
                "accuracy_drop_aligned_cross_probe": teacher_accuracy - aligned_cross["accuracy"],
            }
            del aligned_test, alignment
        result["models"][model_id] = {
            "objective": spec["objective"],
            "alpha": float(spec["alpha"]),
            "layers": layer_results,
        }
        atomic_json_dump(result, output_path)
        del student, student_train, student_test, student_train_y, student_test_y
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["complete"] = len(result["models"]) == len(model_specs)
    atomic_json_dump(result, output_path)
    print(f"[complete] {output_path}", flush=True)


if __name__ == "__main__":
    main()
