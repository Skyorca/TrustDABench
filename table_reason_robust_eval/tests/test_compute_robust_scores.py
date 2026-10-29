from __future__ import annotations

import sys
from pathlib import Path


COMPUTE_DIR = Path(__file__).resolve().parents[1] / "compute-robust"
sys.path.insert(0, str(COMPUTE_DIR))

from compute_robust_scores import (
    canonical_clean_judgments,
    judge_outcome,
    summarize,
)


def test_canonical_clean_prefers_last_success_and_keeps_last_failure() -> None:
    rows = [
        {"id": "a", "judge_status": "success", "score": 0},
        {"id": "a", "judge_status": "success", "score": 1},
        {"id": "a", "judge_status": "judge_error", "score": 0},
        {"id": "b", "judge_status": "judge_error", "score": 0},
        {"id": "b", "judge_status": "skipped_infer_error", "score": 0},
        {"id": "__summary__", "judge_status": "summary"},
    ]

    selected, stats = canonical_clean_judgments(rows)

    assert list(selected) == ["a", "b"]
    assert judge_outcome(selected["a"]) is True
    assert selected["b"]["judge_status"] == "skipped_infer_error"
    assert judge_outcome(selected["b"]) is False
    assert stats == {
        "physical_rows": 6,
        "summary_rows": 1,
        "unique_tasks": 2,
        "collapsed_attempts": 3,
        "selected_successes": 1,
        "selected_failures": 1,
    }


def test_canonical_clean_accepts_legacy_rows_without_status() -> None:
    selected, _stats = canonical_clean_judgments(
        [
            {"id": 1, "score": 0},
            {"id": 1, "score": 1},
        ]
    )

    assert list(selected) == ["DA_1"]
    assert judge_outcome(selected["DA_1"]) is True


def test_all_failed_clean_is_not_excluded_by_unscored_policy() -> None:
    selected, _stats = canonical_clean_judgments(
        [{"id": "a", "judge_status": "judge_error", "score": 0}]
    )
    clean_outcome = judge_outcome(selected["a"])

    result = summarize(
        "overall",
        [("a", clean_outcome, True)],
        unmatched_clean=0,
        policy="exclude",
        clean_outcomes={"a": clean_outcome},
    )

    assert result["paired_records"] == 1
    assert result["clean_dataset_accuracy"] == 0.0
    assert result["clean_incorrect_attack_correct"] == 1
