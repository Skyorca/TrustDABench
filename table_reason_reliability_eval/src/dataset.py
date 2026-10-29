from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils import iter_jsonl


REQUIRED_FIELDS = {"id", "question", "answer", "input_file", "file_root"}


def load_dataset(dataset_path: str | Path, limit: int | None = None) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(path):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"Sample {row.get('id', '<unknown>')} is missing fields: {missing}")
        file_root = Path(str(row["file_root"]))
        if not file_root.is_absolute():
            row = {**row, "file_root": str((path.resolve().parent / file_root).resolve())}
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def parse_input_files(row: Dict[str, Any]) -> List[str]:
    files = [part.strip() for part in str(row.get("input_file", "")).splitlines() if part.strip()]
    if not files:
        raise ValueError(f"Sample {row.get('id')} has empty input_file.")
    return files
