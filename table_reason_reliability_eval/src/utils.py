from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log_event(message: str, level: str = "info", log_level: str = "info") -> None:
    order = {"quiet": 0, "info": 1, "verbose": 2}
    if order.get(log_level, 1) < order.get(level, 1):
        return
    line = f"[{now_time()}] {message}"
    if tqdm is not None:
        tqdm.write(line)
    else:
        print(line, flush=True)


def expand_env(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value)
    match = re.fullmatch(r"\$\{([^}]+)\}", text.strip())
    if match:
        return os.environ.get(match.group(1), "")
    return text


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install it with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text or "sample"


def truncate_text(text: str, limit: Optional[int]) -> str:
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    half = max(1, limit // 2)
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n...[{omitted} chars truncated]...\n{text[-half:]}"


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(row) + "\n")


def load_completed_ids(path: Path) -> set[str]:
    """Return IDs with at least one successful, non-empty prediction."""
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and is_successful_prediction(row):
                completed.add(str(row["id"]))
    return completed


def is_successful_prediction(row: Dict[str, Any]) -> bool:
    if row.get("status") != "success":
        return False
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        for key in ("final_answer", "raw_final_answer"):
            value = prediction.get(key)
            if isinstance(value, str) and value.strip():
                return True
    for key in ("model_response", "response", "final_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
