from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from utils import append_jsonl, is_successful_prediction, load_completed_ids


def test_only_successful_nonempty_predictions_are_completed(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "api", "status": "api_failed", "prediction": {"final_answer": ""}},
        {"id": "rounds", "status": "run_failed", "prediction": {"final_answer": "partial"}},
        {"id": "workspace", "status": "workspace_failed", "prediction": {"final_answer": ""}},
        {"id": "empty", "status": "success", "prediction": {"final_answer": "   "}},
        {"id": "nested", "status": "success", "prediction": {"final_answer": "answer"}},
        {"id": "raw", "status": "success", "prediction": {"raw_final_answer": "answer"}},
        {"id": "eventual", "status": "api_failed", "prediction": {"final_answer": ""}},
        {"id": "eventual", "status": "success", "prediction": {"final_answer": "recovered"}},
        {"id": "eventual", "status": "api_failed", "prediction": {"final_answer": ""}},
    ]
    for row in rows:
        append_jsonl(output, row)

    assert load_completed_ids(output) == {"nested", "raw", "eventual"}
    assert is_successful_prediction(rows[4])
    assert not is_successful_prediction(rows[0])
    assert not is_successful_prediction(rows[3])


def test_legacy_top_level_answer_is_supported() -> None:
    assert is_successful_prediction({"id": "x", "status": "success", "model_response": "42"})
