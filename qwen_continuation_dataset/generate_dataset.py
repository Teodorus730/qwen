from __future__ import annotations

import argparse
import contextlib
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from qwen_continuation.config import load_config, with_overrides
from qwen_continuation.data import stream_documents
from qwen_continuation.generation import generate_continuation_batch
from qwen_continuation.io_utils import (
    ExtraFieldsWriter,
    HfShardWriter,
    IdLedger,
    JsonlWriter,
    load_completed_ids,
)
from qwen_continuation.model import load_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация пар продолжений с помощью модели Qwen Base."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--output",
        help=(
            "Путь к выходному JSONL. Только выбирает файл; "
            "существующие данные по умолчанию резюмируются."
        ),
    )
    parser.add_argument("--mode", choices=("fixed", "entropy"))
    parser.add_argument(
        "--dataset",
        dest="dataset_source",
        choices=("fineweb", "math", "mixed"),
        help="Выбрать FineWeb-Edu, FineMath или взвешенную смесь.",
    )
    parser.add_argument(
        "--math-ratio",
        type=float,
        help=(
            "Доля math для --dataset mixed, от 0.0 до 1.0. "
            "Например: 0.7 значит примерно 70%% FineMath."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Удалить выходной файл перед генерацией вместо резюмирования.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Печатать первые N сгенерированных примеров во время выполнения.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        help="Переопределить generation.prefix_tokens из config.yaml.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Переопределить generation.max_new_tokens из config.yaml.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        help="Переопределить generation.entropy_threshold из config.yaml.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Переопределить generation.batch_size из config.yaml. Документы "
            "генерируются вместе за один вызов модели; 1 отключает батчинг."
        ),
    )
    cycle_group = parser.add_mutually_exclusive_group()
    cycle_group.add_argument(
        "--cycle-detection",
        dest="cycle_enabled",
        action="store_true",
        default=None,
        help="Принудительно включить детекцию циклов по n-граммам на этот запуск.",
    )
    cycle_group.add_argument(
        "--no-cycle-detection",
        dest="cycle_enabled",
        action="store_false",
        default=None,
        help="Принудительно выключить детекцию циклов по n-граммам на этот запуск.",
    )
    parser.add_argument(
        "--cycle-window-chars",
        type=int,
        help="Переопределить generation.cycle_detection.window_chars из config.yaml.",
    )
    parser.add_argument(
        "--cycle-ngram-chars",
        type=int,
        help="Переопределить generation.cycle_detection.ngram_chars из config.yaml.",
    )
    parser.add_argument(
        "--cycle-min-chars",
        type=int,
        help="Переопределить generation.cycle_detection.min_chars из config.yaml.",
    )
    parser.add_argument(
        "--save-extra-fields",
        dest="save_extra_fields",
        action="store_true",
        default=None,
        help=(
            "Сохранять полные метаданные по примерам (источник, счётчики токенов, "
            "время, энтропия) в отдельный локальный файл, батчами. "
            "Без этого флага сохраняются только пары prefix/suffix."
        ),
    )
    parser.add_argument(
        "--extra-fields-path",
        help="Переопределить output.extra_fields.path из config.yaml.",
    )
    parser.add_argument(
        "--extra-fields-batch-size",
        type=int,
        help="Переопределить output.extra_fields.batch_size из config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверить конфигурацию, не загружая модель и датасет.",
    )
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help="Включить инкрементальную загрузку в датасет на Hugging Face (переопределяет huggingface.enabled).",
    )
    parser.add_argument(
        "--hf-repo-id",
        help="Целевой датасет на Hugging Face, например username/dataset-name.",
    )
    parser.add_argument(
        "--hf-token",
        help="Токен Hugging Face. Лучше 'huggingface-cli login' или переменная окружения HF_TOKEN, чем передача здесь.",
    )
    parser.add_argument(
        "--hf-shard-size",
        type=int,
        help="Строк на загружаемый шард.",
    )
    return parser.parse_args()


def print_environment(config: dict[str, Any]) -> None:
    print("Python:", sys.version.split()[0])
    print("Платформа:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA доступна:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Память GPU:",
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
            "GiB",
        )
    else:
        print("Внимание: CUDA недоступна; генерация будет очень медленной.")
    print("Teacher:", config["model"]["id"])
    print("Compile:", bool(config["model"].get("compile", False)))
    dataset_config = config["dataset"]
    selected_source = dataset_config.get("source", "fineweb")
    print("Режим датасета:", selected_source)
    if selected_source == "mixed":
        print(
            "Math ratio:",
            float(dataset_config.get("mixed", {}).get("math_ratio", 0.5)),
        )
        print("Датасет FineWeb:", dataset_config["sources"]["fineweb"]["id"])
        print("Датасет Math:", dataset_config["sources"]["math"]["id"])
    else:
        selected_config = dataset_config["sources"][selected_source]
        print("Датасет:", selected_config["id"])
        print("Подмножество датасета:", selected_config.get("subset"))
    print("Режим:", config["generation"]["mode"])
    print("Размер батча:", int(config["generation"].get("batch_size", 1)))
    print("Максимум примеров:", config["dataset"]["max_examples"])
    print("Вывод:", config["output"]["path"])

    cycle_cfg = config["generation"].get("cycle_detection", {})
    if cycle_cfg.get("enabled", False):
        print("Детекция циклов: включена")
        print("  window_chars:", cycle_cfg.get("window_chars", 100))
        print("  ngram_chars: ", cycle_cfg.get("ngram_chars", 20))
        print("  min_chars:   ", cycle_cfg.get("min_chars", 50))
    else:
        print("Детекция циклов: выключена")

    extra_cfg = config["output"].get("extra_fields", {})
    if extra_cfg.get("enabled", False):
        print("Extra fields: включены")
        print("  путь:      ", extra_cfg.get("path", "outputs/extra.jsonl"))
        print("  batch_size:", extra_cfg.get("batch_size", 100))
    else:
        print("Extra fields: выключены (только prefix/suffix)")

    hf_cfg = config.get("huggingface", {})
    if hf_cfg.get("enabled", False):
        print("Загрузка на HF: включена")
        print("Репозиторий HF:", hf_cfg.get("repo_id", "(не задан)"))
        print("Размер шарда:  ", hf_cfg.get("shard_size", 10000))
    else:
        print("Загрузка на HF: выключена")


def build_record(
    *,
    document: dict[str, Any],
    prefix_ids: list[int],
    real_continuation_ids: list[int],
    generated_ids: list[int],
    entropies: list[float],
    tokenizer: Any,
    config: dict[str, Any],
    dtype_name: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    generation = config["generation"]

    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    real_continuation = tokenizer.decode(
        real_continuation_ids, skip_special_tokens=True
    )
    teacher_continuation = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    synthetic_text = tokenizer.decode(
        prefix_ids + generated_ids, skip_special_tokens=True
    )

    return {
        "source_id": document["source_id"],
        "source_name": document["source_name"],
        "source_dataset": document["source_dataset"],
        "source_subset": document.get("source_subset"),
        "source_metadata": document.get("metadata", {}),
        "prefix_text": prefix_text,
        "real_continuation": real_continuation,
        "teacher_continuation": teacher_continuation,
        "synthetic_text": synthetic_text,
        "prefix_token_count": len(prefix_ids),
        "real_continuation_token_count": len(real_continuation_ids),
        "generated_token_count": len(generated_ids),
        "teacher_model": config["model"]["id"],
        "teacher_dtype": dtype_name,
        "generation_seconds": round(elapsed_seconds, 4),
        "generation": {
            "mode": generation["mode"],
            "temperature": float(generation.get("temperature", 0.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "top_k": int(generation.get("top_k", 0)),
            "max_new_tokens": int(generation["max_new_tokens"]),
            "entropy_threshold": (
                float(generation["entropy_threshold"])
                if generation["mode"] == "entropy"
                else None
            ),
            "token_entropies": [round(value, 6) for value in entropies],
        },
    }


def split_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix_suffix = {
        "prefix": record["prefix_text"],
        "suffix": record["teacher_continuation"],
    }
    extra = {
        key: value
        for key, value in record.items()
        if key not in ("prefix_text", "teacher_continuation")
    }
    return prefix_suffix, extra


def print_preview(record: dict[str, Any], index: int) -> None:
    entropies = record.get("generation", {}).get("token_entropies", [])
    entropy_mean = (
        sum(entropies) / len(entropies)
        if entropies
        else None
    )
    entropy_max = max(entropies) if entropies else None

    lines = [
        "",
        "=" * 88,
        f"ПРЕВЬЮ {index}",
        f"Источник: {record.get('source_name')}",
        f"Датасет: {record.get('source_dataset')}",
        f"Сгенерировано токенов: {record.get('generated_token_count')}",
        f"Время генерации, сек: {record.get('generation_seconds')}",
    ]

    if entropy_mean is not None:
        lines.append(f"Энтропия, среднее: {entropy_mean:.4f}")
        lines.append(f"Энтропия, максимум: {entropy_max:.4f}")

    lines.extend(
        [
            "",
            "ПРЕФИКС:",
            record.get("prefix_text", ""),
            "",
            "НАСТОЯЩЕЕ ПРОДОЛЖЕНИЕ:",
            record.get("real_continuation", ""),
            "",
            "ПРОДОЛЖЕНИЕ QWEN:",
            record.get("teacher_continuation", ""),
            "=" * 88,
            "",
        ]
    )
    tqdm.write("\n".join(lines))


def _next_batch(
    document_iter: Any,
    *,
    tokenizer: Any,
    prefix_tokens: int,
    max_new_tokens: int,
    completed_ids: set[str],
    batch_size: int,
    counters: dict[str, int],
) -> tuple[list[dict[str, Any]], list[list[int]], list[list[int]]]:
    required_tokens = prefix_tokens + max_new_tokens
    docs: list[dict[str, Any]] = []
    prefix_batch: list[list[int]] = []
    real_continuation_batch: list[list[int]] = []

    for document in document_iter:
        if document["source_id"] in completed_ids:
            counters["skipped_completed"] += 1
            continue

        text = document["text"]
        token_ids = tokenizer.encode(
            text[: required_tokens * 10], add_special_tokens=False
        )
        if len(token_ids) < required_tokens:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < required_tokens:
            counters["skipped_short"] += 1
            continue

        docs.append(document)
        prefix_batch.append(token_ids[:prefix_tokens])
        real_continuation_batch.append(
            token_ids[prefix_tokens:prefix_tokens + max_new_tokens]
        )
        if len(docs) >= batch_size:
            break

    return docs, prefix_batch, real_continuation_batch


def _generate_with_backoff(
    teacher: Any,
    prefix_batch: list[list[int]],
    config: dict[str, Any],
) -> list[tuple[list[int], list[float]]]:
    try:
        return generate_continuation_batch(teacher, prefix_batch, config)
    except torch.cuda.OutOfMemoryError:
        if len(prefix_batch) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(prefix_batch) // 2
        tqdm.write(
            f"[OOM] батч из {len(prefix_batch)} не влез, "
            f"повтор как {mid} + {len(prefix_batch) - mid}"
        )
        first = _generate_with_backoff(teacher, prefix_batch[:mid], config)
        second = _generate_with_backoff(teacher, prefix_batch[mid:], config)
        return first + second


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = with_overrides(
        config,
        max_examples=args.max_examples,
        output_path=args.output,
        mode=args.mode,
        dataset_source=args.dataset_source,
        math_ratio=args.math_ratio,
        prefix_tokens=args.prefix_tokens,
        max_new_tokens=args.max_new_tokens,
        entropy_threshold=args.entropy_threshold,
        batch_size=args.batch_size,
        cycle_enabled=args.cycle_enabled,
        cycle_window_chars=args.cycle_window_chars,
        cycle_ngram_chars=args.cycle_ngram_chars,
        cycle_min_chars=args.cycle_min_chars,
        save_extra_fields=args.save_extra_fields,
        extra_fields_path=args.extra_fields_path,
        extra_fields_batch_size=args.extra_fields_batch_size,
        hf_upload=args.hf_upload or None,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        hf_shard_size=args.hf_shard_size,
    )

    print_environment(config)

    if args.dry_run:
        print("Dry run завершён: конфигурация корректна.")
        return

    if not torch.cuda.is_available():
        answer = input(
            "CUDA недоступна. Продолжить на CPU? Это может быть очень медленно. [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Остановлено. Включите GPU и запустите снова.")

    output_path = Path(config["output"]["path"])
    resume = bool(config["output"].get("resume", True))
    hf_cfg = config.get("huggingface", {})
    hf_enabled = hf_cfg.get("enabled", False)
    extra_cfg = config["output"].get("extra_fields", {})
    extra_enabled = extra_cfg.get("enabled", False)
    extra_path = Path(extra_cfg.get("path", "outputs/extra.jsonl"))
    ledger_path = output_path.parent / "ids.txt"

    if args.overwrite or not resume:
        if hf_enabled:
            removed = 0
            for shard in output_path.parent.glob("train-*.jsonl"):
                shard.unlink()
                removed += 1
            state_file = output_path.parent / "state.json"
            if state_file.exists():
                state_file.unlink()
            if removed:
                print(f"Удалено шардов: {removed}, из: {output_path.parent}")
        else:
            if output_path.exists():
                output_path.unlink()
                print(f"Удалён существующий вывод: {output_path}")
        if ledger_path.exists():
            ledger_path.unlink()
        if extra_path.exists():
            extra_path.unlink()
        completed_ids: set[str] = set()
    else:
        completed_ids = load_completed_ids(ledger_path)
        if completed_ids:
            print(
                f"Резюмирование включено: найдено уже обработанных "
                f"source_id: {len(completed_ids)}"
            )

    teacher = load_teacher(config)
    tokenizer = teacher.tokenizer

    prefix_tokens = int(config["generation"]["prefix_tokens"])
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    max_examples = int(config["dataset"]["max_examples"])
    batch_size = max(1, int(config["generation"].get("batch_size", 1)))
    flush_every = int(config["output"].get("flush_every", 1))

    saved = 0
    counters = {"skipped_short": 0, "skipped_completed": 0}
    source_counts: dict[str, int] = {"fineweb": 0, "math": 0}
    started = time.perf_counter()

    progress = tqdm(total=max_examples, desc="Сохранено примеров")

    if hf_enabled:
        writer_cm = HfShardWriter(
            output_dir=output_path.parent,
            repo_id=hf_cfg["repo_id"],
            shard_size=hf_cfg.get("shard_size", 10000),
            token=hf_cfg.get("token"),
            flush_every=flush_every,
        )
    else:
        writer_cm = JsonlWriter(output_path, flush_every=flush_every)

    ledger_cm = IdLedger(ledger_path, flush_every=flush_every)
    extra_cm = (
        ExtraFieldsWriter(
            path=extra_path,
            batch_size=extra_cfg.get("batch_size", 100),
        )
        if extra_enabled
        else contextlib.nullcontext()
    )

    document_iter = iter(stream_documents(config))

    with writer_cm as writer, ledger_cm as ledger, extra_cm as extra_writer:
        while saved < max_examples:
            current_batch_size = min(batch_size, max_examples - saved)
            docs, prefix_batch, real_continuation_batch = _next_batch(
                document_iter,
                tokenizer=tokenizer,
                prefix_tokens=prefix_tokens,
                max_new_tokens=max_new_tokens,
                completed_ids=completed_ids,
                batch_size=current_batch_size,
                counters=counters,
            )
            if not docs:
                break

            generation_started = time.perf_counter()
            batch_results = _generate_with_backoff(teacher, prefix_batch, config)
            generation_elapsed = time.perf_counter() - generation_started
            per_doc_elapsed = generation_elapsed / len(docs)

            for document, prefix_ids, real_continuation_ids, (generated_ids, entropies) in zip(
                docs, prefix_batch, real_continuation_batch, batch_results
            ):
                record = build_record(
                    document=document,
                    prefix_ids=prefix_ids,
                    real_continuation_ids=real_continuation_ids,
                    generated_ids=generated_ids,
                    entropies=entropies,
                    tokenizer=tokenizer,
                    config=config,
                    dtype_name=teacher.dtype_name,
                    elapsed_seconds=per_doc_elapsed,
                )
                prefix_suffix, extra_record = split_record(record)
                writer.write(prefix_suffix)
                ledger.add(document["source_id"])
                if extra_writer is not None:
                    extra_writer.write(extra_record)
                completed_ids.add(document["source_id"])
                saved += 1

                if args.preview > 0 and saved <= args.preview:
                    print_preview(record, saved)

                source_name = document["source_name"]
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
                progress.update(1)
                progress.set_postfix(
                    {
                        "источник": source_name,
                        "сгенерировано": len(generated_ids),
                        "сек": round(per_doc_elapsed, 2),
                    }
                )

    progress.close()
    total_elapsed = time.perf_counter() - started

    print()
    print("Готово.")
    print("Сохранено:", saved)
    print("Сохранено по источникам:", source_counts)
    print("Пропущено коротких документов:", counters["skipped_short"])
    print("Пропущено уже обработанных документов:", counters["skipped_completed"])
    print("Затрачено секунд:", round(total_elapsed, 2))
    if hf_enabled:
        print("Репозиторий HF:", hf_cfg["repo_id"])
    else:
        print("Вывод:", output_path.resolve())
    if extra_enabled:
        print("Extra fields:", extra_path.resolve())


if __name__ == "__main__":
    main()
