from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, hf_hub_download


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
    output_dir = Path(output_dir)
    completed: set[str] = set()
    shard_files = sorted(output_dir.glob("train-*.jsonl"))
    for shard_path in shard_files:
        completed |= load_completed_ids(shard_path)
    if completed:
        print(
            f"[HF] resume: {len(completed):,} completed IDs "
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
                    f"[HF] resuming from shard {shard} "
                    f"({total:,} examples uploaded)"
                )
                return shard, total
            except Exception as exc:
                print(f"[HF] state.json unreadable ({exc}), scanning repo...")

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
                        f"[HF] found {len(shard_files)} shard(s) on hub, "
                        f"resuming from shard {next_shard} ({total:,} examples)"
                    )
                    return next_shard, total
        except Exception as exc:
            print(f"[HF] could not scan repo ({exc}), starting from scratch")

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
                f"[HF] could not fetch {filename} for an exact count ({exc}), "
                f"assuming {self.shard_size} rows"
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
                    f"[HF] partial shard {self.current_path.name}: "
                    f"{existing} rows, appending "
                    f"({self.shard_size - existing} left)"
                )
                return

            print(
                f"[HF] {self.current_path.name} already has {existing} rows, "
                f"re-uploading and moving on"
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
            f"[HF] uploaded {self.current_path.name} "
            f"({self.total_examples:,} total)"
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
                f"[HF] uploaded {self.current_path.name} "
                f"({self.total_examples:,} total)"
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
            "| `source_id` | source document ID |\n"
            "| `source_name` | source dataset (`fineweb` / `math`) |\n"
            "| `prefix_text` | input prefix |\n"
            "| `real_continuation` | original continuation from source |\n"
            "| `teacher_continuation` | model-generated continuation |\n"
            "| `synthetic_text` | prefix + teacher continuation |\n"
            "| `generation` | generation settings and per-token entropies |\n"
        )
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(text, encoding="utf-8")
        self.api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
