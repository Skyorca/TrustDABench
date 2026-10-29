from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from scorer_dabench import DabenchClosedFormScorer, extract_format, is_equal, normalize_dabench_id


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_extract_format_and_numeric_tolerance() -> None:
    parsed = extract_format("answer: @mean_fare[34.6500000001], @significance[significant]")
    assert parsed == {"mean_fare": "34.6500000001", "significance": "significant"}
    assert is_equal(parsed["mean_fare"], "34.65")
    assert is_equal(parsed["significance"], "significant")
    assert not is_equal("Significant", "significant")


def test_normalize_dabench_ids() -> None:
    assert normalize_dabench_id("DA_109") == "109"
    assert normalize_dabench_id("DA_109__unit_scale_conversion__001") == "109"
    assert normalize_dabench_id(109) == "109"


def test_score_prediction_requires_all_fields(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    write_jsonl(labels_path, [{"id": 6, "common_answers": [["a", "1.00"], ["b", "ok"]]}])
    write_jsonl(questions_path, [{"id": 6, "concepts": ["Summary Statistics"]}])
    scorer = DabenchClosedFormScorer(labels_path, questions_path)

    passed = scorer.score_prediction(
        prediction_id="DA_6__attack__001",
        source_id="DA_6",
        response="@a[1.00000000001] @b[ok]",
        run_status="success",
    )
    assert passed["score"] == 1
    assert passed["is_correct"] is True
    assert passed["correctness"] == {"a": True, "b": True}

    failed = scorer.score_prediction(
        prediction_id="DA_6__attack__001",
        source_id="DA_6",
        response="@a[1.00]",
        run_status="success",
    )
    assert failed["score"] == 0
    assert failed["is_correct"] is False
    assert failed["correctness"] == {"a": True, "b": False}
    assert "missing fields: b" in failed["reason"]


def test_not_evaluable_without_label_or_response(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(labels_path, [{"id": 0, "common_answers": [["mean_fare", "34.65"]]}])
    scorer = DabenchClosedFormScorer(labels_path)

    no_response = scorer.score_prediction(
        prediction_id="DA_0",
        source_id=None,
        response="",
        run_status="success",
    )
    assert no_response["judge_status"] == "not_evaluable"
    assert no_response["score"] == 0

    no_label = scorer.score_prediction(
        prediction_id="DA_404",
        source_id=None,
        response="@mean_fare[34.65]",
        run_status="success",
    )
    assert no_label["judge_status"] == "not_evaluable"
    assert no_label["error"] == "label_not_found"
