from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
