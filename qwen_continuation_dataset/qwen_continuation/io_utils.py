from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, hf_hub_download


def load_completed_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


class IdLedger:
    def __init__(self, path: str | Path, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self._file = None
        self._pending = 0

    def __enter__(self) -> "IdLedger":
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def add(self, source_id: str) -> None:
        self._file.write(f"{source_id}\n")
        self._pending += 1
        if self._pending >= self.flush_every:
            self._file.flush()
            self._pending = 0

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()


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


class ExtraFieldsWriter:
    def __init__(self, path: str | Path, batch_size: int = 100) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, int(batch_size))
        self._buffer: list[dict[str, Any]] = []
        self._file = None

    def __enter__(self) -> "ExtraFieldsWriter":
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def write(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        lines = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in self._buffer
        )
        self._file.write(lines)
        self._file.flush()
        self._buffer.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._flush()
        if self._file is not None:
            self._file.close()


class HfShardWriter:
    _STATE_FILE = "state.json"

    def __init__(
        self,
        output_dir: str | Path,
        repo_id: str,
        shard_size: int = 10000,
        token: str | None = None,
        flush_every: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.shard_size = shard_size
        self.flush_every = max(1, int(flush_every))
        self._pending = 0

        self._token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self._token)
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            token=self._token,
        )

        self.current_shard, self.total_examples = self._load_state()
        self.current_examples = 0
        self.current_path: Path | None = None
        self.file = None

    def _load_state(self) -> tuple[int, int]:
        state_path = self.output_dir / self._STATE_FILE
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                shard = int(state["current_shard"])
                total = int(state["total_examples"])
                print(
                    f"[HF] резюмируем с шарда {shard} "
                    f"(загружено примеров: {total:,})"
                )
                return shard, total
            except Exception as exc:
                print(f"[HF] state.json нечитаем ({exc}), сканируем репозиторий...")

        try:
            files = list(
                self.api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
            )
            shard_files = [
                f for f in files
                if f.startswith("data/train-") and f.endswith(".jsonl")
            ]
            if shard_files:
                indices = []
                for f in shard_files:
                    try:
                        indices.append(int(Path(f).stem.split("-")[1]))
                    except (IndexError, ValueError):
                        pass
                if indices:
                    indices.sort()
                    next_shard = indices[-1] + 1
                    last_shard_count = self._count_remote_shard(indices[-1])
                    total = (len(indices) - 1) * self.shard_size + last_shard_count
                    print(
                        f"[HF] найдено шардов на хабе: {len(shard_files)}, "
                        f"резюмируем с шарда {next_shard} (примеров: {total:,})"
                    )
                    return next_shard, total
        except Exception as exc:
            print(f"[HF] не удалось просканировать репозиторий ({exc}), начинаем с нуля")

        return 0, 0

    def _count_remote_shard(self, shard_index: int) -> int:
        filename = f"data/train-{shard_index:05d}.jsonl"
        try:
            local_copy = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                token=self._token,
            )
            with open(local_copy, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception as exc:
            print(
                f"[HF] не удалось скачать {filename} для точного подсчёта ({exc}), "
                f"предполагаем {self.shard_size} строк"
            )
            return self.shard_size

    def _save_state(self) -> None:
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

    def __enter__(self) -> "HfShardWriter":
        self._open_shard()
        return self

    def _open_shard(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

        while True:
            self.current_path = (
                self.output_dir / f"train-{self.current_shard:05d}.jsonl"
            )

            if not self.current_path.exists():
                self.current_examples = 0
                self.file = open(self.current_path, "w", encoding="utf-8")
                return

            with open(self.current_path, "r", encoding="utf-8") as f:
                existing = sum(1 for line in f if line.strip())

            if existing < self.shard_size:
                self.current_examples = existing
                self.total_examples += existing
                self.file = open(self.current_path, "a", encoding="utf-8")
                print(
                    f"[HF] частичный шард {self.current_path.name}: "
                    f"строк {existing}, дозаписываем "
                    f"(осталось {self.shard_size - existing})"
                )
                return

            print(
                f"[HF] {self.current_path.name} уже содержит строк: {existing}, "
                f"перезаливаем и переходим дальше"
            )
            self.api.upload_file(
                path_or_fileobj=str(self.current_path),
                path_in_repo=f"data/{self.current_path.name}",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self.total_examples += existing
            self.current_shard += 1
            self._update_readme()
            self._save_state()

    def write(self, record: dict[str, Any]) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.current_examples += 1
        self.total_examples += 1
        self._pending += 1
        if self._pending >= self.flush_every:
            self.file.flush()
            self._pending = 0
        if self.current_examples >= self.shard_size:
            self._finish_shard()

    def _finish_shard(self) -> None:
        self.file.flush()
        self._pending = 0
        self.api.upload_file(
            path_or_fileobj=str(self.current_path),
            path_in_repo=f"data/{self.current_path.name}",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        self.current_shard += 1
        print(
            f"[HF] загружен {self.current_path.name} "
            f"(всего примеров: {self.total_examples:,})"
        )
        self._update_readme()
        self._save_state()
        self._open_shard()

    def __exit__(self, *args) -> None:
        if self.file is None:
            return
        if self.current_examples > 0:
            self.file.flush()
            self.api.upload_file(
                path_or_fileobj=str(self.current_path),
                path_in_repo=f"data/{self.current_path.name}",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self.current_shard += 1
            print(
                f"[HF] загружен {self.current_path.name} "
                f"(всего примеров: {self.total_examples:,})"
            )
            self._update_readme()
            self._save_state()
        self.file.close()

    def _update_readme(self) -> None:
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
            "Generated with [qwen_continuation_dataset](https://github.com/Teodorus730/qwen).\n"
            "\n"
            "## Statistics\n"
            "\n"
            "| | |\n"
            "|---|---:|\n"
            f"| Shards | {self.current_shard} |\n"
            f"| Examples | {self.total_examples:,} |\n"
            f"| Shard size | {self.shard_size:,} |\n"
            f"| Updated | {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC |\n"
            "\n"
            "## Usage\n"
            "\n"
            "```python\n"
            "from datasets import load_dataset\n"
            "\n"
            f'ds = load_dataset("{self.repo_id}")\n'
            f'ds = load_dataset("{self.repo_id}", streaming=True)\n'
            "```\n"
            "\n"
            "## Fields\n"
            "\n"
            "| Field | Description |\n"
            "|---|---|\n"
            "| `prefix` | input prefix |\n"
            "| `suffix` | model-generated continuation |\n"
            "\n"
            "Only these two fields are published here. If the generator was run with\n"
            "`--save-extra-fields`, the source generation run also has a local-only\n"
            "file (not published) with per-example metadata: source id/dataset,\n"
            "original continuation, token counts, timing, and per-token entropy.\n"
        )
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(text, encoding="utf-8")
        self.api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
