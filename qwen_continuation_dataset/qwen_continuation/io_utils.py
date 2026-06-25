from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo


def load_completed_ids(path: str | Path) -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping invalid JSON at "
                    f"{output_path}:{line_number}"
                )
                continue
            source_id = record.get("source_id")
            if source_id is not None:
                completed.add(str(source_id))
    return completed


def load_completed_ids_from_dir(output_dir: str | Path) -> set[str]:
    """Scan all train-*.jsonl shards in a directory and return their source IDs."""
    output_dir = Path(output_dir)
    completed: set[str] = set()
    shard_files = sorted(output_dir.glob("train-*.jsonl"))
    for shard_path in shard_files:
        completed |= load_completed_ids(shard_path)
    if completed:
        print(
            f"[HF] Resume: found {len(completed):,} completed IDs "
            f"across {len(shard_files)} local shard(s)"
        )
    return completed


class JsonlWriter:
    def __init__(self, path: str | Path, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self._file = None
        self._pending = 0

    def __enter__(self) -> "JsonlWriter":
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("JsonlWriter must be used as a context manager.")
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._pending += 1
        if self._pending >= self.flush_every:
            self._file.flush()
            self._pending = 0

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()


class HfShardWriter:
    """
    Context manager: пишет JSONL-записи шардами фиксированного размера
    и загружает каждый завершённый шард в датасет Hugging Face.

    - Каждые shard_size примеров текущий файл заливается на HF и открывается новый.
      Уже загруженные шарды никогда не перезаписываются.
    - После каждой загрузки обновляется README.md и state.json.
    - При перезапуске читается state.json (или сканируется HF-репо),
      чтобы продолжить с нужного номера шарда.
    """

    _STATE_FILE = "state.json"

    def __init__(
        self,
        output_dir: str | Path,
        repo_id: str,
        shard_size: int = 10000,
        token: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.shard_size = shard_size

        token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=token)
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            token=token,
        )

        # current_shard = индекс следующего шарда для записи (0 при старте с нуля)
        self.current_shard, self.total_examples = self._load_state()
        self.current_examples = 0
        self.current_path: Path | None = None
        self.file = None

    # ------------------------------------------------------------------ state

    def _load_state(self) -> tuple[int, int]:
        """
        Возвращает (следующий_шард, всего_примеров).
        Источники по приоритету: state.json → список файлов на HF → с нуля.
        """
        state_path = self.output_dir / self._STATE_FILE
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                shard = int(state["current_shard"])
                total = int(state["total_examples"])
                print(
                    f"[HF] state.json найден: продолжаем с шарда {shard}, "
                    f"уже сгенерировано {total:,} примеров"
                )
                return shard, total
            except Exception as exc:
                print(f"[HF] state.json нечитаем ({exc}), сканируем HF-репо…")

        # Fallback: сканируем HF-репо
        try:
            files = list(self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
            ))
            shard_files = [
                f for f in files
                if f.startswith("data/train-") and f.endswith(".jsonl")
            ]
            if shard_files:
                indices = []
                for f in shard_files:
                    try:
                        idx = int(Path(f).stem.split("-")[1])
                        indices.append(idx)
                    except (IndexError, ValueError):
                        pass
                if indices:
                    next_shard = max(indices) + 1
                    total = next_shard * self.shard_size  # приблизительно
                    print(
                        f"[HF] На HF найдено {len(shard_files)} шард(ов), "
                        f"продолжаем с шарда {next_shard}"
                    )
                    return next_shard, total
        except Exception as exc:
            print(f"[HF] Не удалось просканировать HF-репо ({exc}), начинаем с нуля")

        return 0, 0

    def _save_state(self) -> None:
        """Сохраняет прогресс в state.json."""
        state = {
            "current_shard": self.current_shard,
            "total_examples": self.total_examples,
            "last_uploaded": (
                self.current_path.name if self.current_path else None
            ),
            "updated": datetime.utcnow().isoformat(),
        }
        (self.output_dir / self._STATE_FILE).write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------ shard logic

    def __enter__(self) -> "HfShardWriter":
        self._open_shard()
        return self

    def _open_shard(self) -> None:
        if self.file is not None:
            self.file.close()
        self.current_path = (
            self.output_dir / f"train-{self.current_shard:05d}.jsonl"
        )
        self.file = open(self.current_path, "w", encoding="utf-8")
        self.current_examples = 0

    def write(self, record: dict[str, Any]) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.current_examples += 1
        self.total_examples += 1
        if self.current_examples >= self.shard_size:
            self._finish_shard()

    def _finish_shard(self) -> None:
        """Загружает завершённый шард, обновляет README и state.json, открывает следующий."""
        self.file.flush()
        self.api.upload_file(
            path_or_fileobj=str(self.current_path),
            path_in_repo=f"data/{self.current_path.name}",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        self.current_shard += 1
        print(
            f"[HF] ✓ {self.current_path.name} загружен "
            f"— всего {self.total_examples:,} примеров"
        )
        self._update_readme()
        self._save_state()
        self._open_shard()

    def __exit__(self, *args) -> None:
        if self.file is None:
            return
        if self.current_examples > 0:
            # Загружаем последний неполный шард
            self.file.flush()
            self.api.upload_file(
                path_or_fileobj=str(self.current_path),
                path_in_repo=f"data/{self.current_path.name}",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self.current_shard += 1
            print(
                f"[HF] ✓ Финальный шард {self.current_path.name} загружен "
                f"— всего {self.total_examples:,} примеров"
            )
            self._update_readme()
            self._save_state()
        self.file.close()

    # ----------------------------------------------------------- HF metadata

    def _update_readme(self) -> None:
        """Загружает обновлённый dataset card (README.md) в HF-репо."""
        text = (
            "---\n"
            "license: apache-2.0\n"
            "language:\n"
            "- en\n"
            "task_categories:\n"
            "- text-generation\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: data/*.jsonl\n"
            "---\n"
            "\n"
            "# Qwen Continuation Dataset\n"
            "\n"
            "Автоматически сгенерированный датасет продолжений на базе модели Qwen.\n"
            "Датасет загружается **инкрементально** во время генерации — каждый завершённый шард\n"
            "сразу публикуется на Hub, так что данные доступны частично в любой момент,\n"
            "а генерацию можно безопасно возобновить после прерывания.\n"
            "\n"
            "## Текущая статистика\n"
            "\n"
            "| Метрика | Значение |\n"
            "|---|---:|\n"
            f"| Шардов загружено | {self.current_shard} |\n"
            f"| Примеров | {self.total_examples:,} |\n"
            f"| Размер шарда | {self.shard_size:,} |\n"
            f"| Последнее обновление | {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC |\n"
            "\n"
            "## Поддержка возобновления (resume)\n"
            "\n"
            "Генерация полностью возобновляема. Если процесс остановился:\n"
            "\n"
            "1. Локальные файлы шардов сохраняются в папке `outputs/`.\n"
            "2. Уже загруженные шарды остаются на Hugging Face — **ничего не перезаписывается**.\n"
            "3. При следующем запуске скрипт читает `state.json` (или сканирует HF-репо\n"
            "   в поисках существующих шардов) и автоматически продолжает с нужного номера.\n"
            "4. Обработанные документы отслеживаются через `source_id`, поэтому дублей\n"
            "   не возникает даже после аварийной остановки.\n"
            "\n"
            "## Структура репозитория\n"
            "\n"
            "```\n"
            "README.md\n"
            "data/\n"
            "  train-00000.jsonl\n"
            "  train-00001.jsonl\n"
            "  ...\n"
            "```\n"
            "\n"
            "## Загрузка\n"
            "\n"
            "```python\n"
            "from datasets import load_dataset\n"
            "\n"
            f'ds = load_dataset("{self.repo_id}")\n'
            "# или в потоковом режиме:\n"
            f'ds = load_dataset("{self.repo_id}", streaming=True)\n'
            "```\n"
            "\n"
            "## Поля записи\n"
            "\n"
            "| Поле | Описание |\n"
            "|---|---|\n"
            "| `source_id` | ID исходного документа |\n"
            "| `source_name` | Название датасета-источника (`fineweb` / `math`) |\n"
            "| `prefix_text` | Текст-префикс (контекст) |\n"
            "| `real_continuation` | Настоящее продолжение из источника |\n"
            "| `teacher_continuation` | Продолжение, сгенерированное моделью |\n"
            "| `synthetic_text` | prefix + teacher continuation |\n"
            "| `generation` | Гиперпараметры генерации и энтропии по токенам |\n"
        )
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(text, encoding="utf-8")
        self.api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
