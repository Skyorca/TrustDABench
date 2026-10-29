from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from judge_refusal import JudgeFatalError, canonical_prediction_rows, load_prediction_groups


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def attempts() -> list[dict]:
    return [
        {"id": "a", "status": "run_failed", "prediction": {"final_answer": ""}},
        {"id": "a", "status": "success", "prediction": {"final_answer": "a-ok"}},
        {"id": "b", "status": "success", "prediction": {"final_answer": "b-ok"}},
        {"id": "b", "status": "run_failed", "prediction": {"final_answer": ""}},
        {"id": "c", "status": "success", "prediction": {"final_answer": "c-old"}},
        {"id": "c", "status": "success", "prediction": {"final_answer": "c-new"}},
        {"id": "d", "status": "run_failed", "prediction": {"final_answer": "first"}},
        {"id": "d", "status": "run_failed", "prediction": {"final_answer": "last"}},
    ]


def test_canonical_attempts_prefer_last_nonempty_success() -> None:
    selected, stats = canonical_prediction_rows(attempts(), Path("predictions.jsonl"))
    assert [row["prediction"]["final_answer"] for row in selected] == ["a-ok", "b-ok", "c-new", "last"]
    assert [row["_input_line"] for row in selected] == [2, 3, 6, 8]
    assert stats == {"physical_rows": 8, "unique_tasks": 4, "collapsed_attempts": 4}


def test_limit_is_applied_after_deduplication(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, attempts())
    groups, stats = load_prediction_groups([path], limit=2)
    assert [row["id"] for row in groups[0][1]] == ["a", "b"]
    assert stats == {"physical_rows": 8, "unique_tasks": 4, "collapsed_attempts": 4}


def test_missing_id_reports_source_line() -> None:
    with pytest.raises(JudgeFatalError, match=r"predictions.jsonl:2"):
        canonical_prediction_rows([{"id": "ok"}, {"status": "run_failed"}], Path("predictions.jsonl"))
