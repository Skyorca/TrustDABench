from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils import iter_jsonl


ANSWER_PATTERN = re.compile(r"@(\w+)\[(.*?)\]", flags=re.S)
NUMERIC_TOLERANCE = 1e-6


class DabenchClosedFormScorer:
    """Dataset-specific scorer compatible with DABENCH eval_closed_form.py."""

    def __init__(self, labels_path: str | Path, questions_path: str | Path | None = None, id_prefix: str = "DA_"):
        self.labels_path = Path(labels_path)
        self.questions_path = Path(questions_path) if questions_path else None
        self.id_prefix = id_prefix
        self.labels = load_labels(self.labels_path)
        self.questions = load_questions(self.questions_path) if self.questions_path else {}

    def score_prediction(
        self,
        *,
        prediction_id: Any,
        source_id: Any,
        response: str,
        run_status: Any,
        error: Any = None,
    ) -> Dict[str, Any]:
        label_id = self.resolve_label_id(prediction_id=prediction_id, source_id=source_id)
        label_answers = self.labels.get(label_id)
        predicted_answers = extract_format(response)

        if run_status != "success" or not response.strip():
            return self._not_evaluable(
                label_id=label_id,
                label_answers=label_answers,
                predicted_answers=predicted_answers,
                reason="Model did not produce an evaluable final answer.",
                error=error or "prediction_not_evaluable",
            )
        if label_answers is None:
            return self._not_evaluable(
                label_id=label_id,
                label_answers=None,
                predicted_answers=predicted_answers,
                reason=f"No DABENCH label found for normalized id {label_id!r}.",
                error="label_not_found",
            )

        correctness = {
            name: is_equal(predicted_answers.get(name), expected)
            for name, expected in label_answers.items()
        }
        is_correct = all(correctness.values()) if correctness else False
        missing = [name for name in label_answers if name not in predicted_answers]
        wrong = [name for name, ok in correctness.items() if not ok and name not in missing]
        reason = build_reason(is_correct, missing, wrong)
        return {
            "judge_status": "success",
            "score": 1 if is_correct else 0,
            "label": "correct" if is_correct else "incorrect",
            "is_correct": is_correct,
            "reason": reason,
            "label_id": label_id,
            "label_answers": label_answers,
            "predicted_answers": predicted_answers,
            "correctness": correctness,
            "judge_method": "dabench_closed_form_v1",
            "raw_judge_response": "",
            "error": None,
        }

    def resolve_label_id(self, *, prediction_id: Any, source_id: Any) -> str:
        candidate = source_id if source_id not in (None, "") else prediction_id
        return normalize_dabench_id(candidate, id_prefix=self.id_prefix)

    def build_dataset_report(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        materialized = list(rows)
        by_label_id = {str(row.get("label_id")): row for row in materialized if row.get("label_id") is not None}
        by_concept: Dict[str, List[bool]] = defaultdict(list)
        by_question: Dict[str, bool] = {}
        sub_question_total = 0
        sub_question_correct = 0

        for label_id, label_answers in self.labels.items():
            row = by_label_id.get(label_id)
            question = self.questions.get(label_id, {})
            is_correct = bool(row and row.get("is_correct"))
            by_question[label_id] = is_correct
            for concept in question.get("concepts") or []:
                by_concept[str(concept)].append(is_correct)
            correctness = row.get("correctness") if row else {}
            if not isinstance(correctness, dict):
                correctness = {}
            for name in label_answers:
                sub_question_total += 1
                if correctness.get(name) is True:
                    sub_question_correct += 1

        concept_accuracy = {
            concept: {
                "accuracy": round(sum(values) / len(values), 4) if values else 0,
                "correct": sum(values),
                "total": len(values),
            }
            for concept, values in sorted(by_concept.items())
        }
        return {
            "labels_path": str(self.labels_path),
            "questions_path": str(self.questions_path) if self.questions_path else None,
            "official_label_count": len(self.labels),
            "matched_label_count": len(by_label_id),
            "question_accuracy": round(sum(by_question.values()) / len(by_question), 4) if by_question else 0,
            "sub_question_accuracy": round(sub_question_correct / sub_question_total, 4) if sub_question_total else 0,
            "sub_question_correct": sub_question_correct,
            "sub_question_total": sub_question_total,
            "concept_accuracy": concept_accuracy,
        }

    def _not_evaluable(
        self,
        *,
        label_id: str,
        label_answers: Optional[Dict[str, str]],
        predicted_answers: Dict[str, str],
        reason: str,
        error: Any,
    ) -> Dict[str, Any]:
        correctness = {name: False for name in (label_answers or {})}
        return {
            "judge_status": "not_evaluable",
            "score": 0,
            "label": "incorrect",
            "is_correct": False,
            "reason": reason,
            "label_id": label_id,
            "label_answers": label_answers,
            "predicted_answers": predicted_answers,
            "correctness": correctness,
            "judge_method": "dabench_closed_form_v1",
            "raw_judge_response": "",
            "error": error,
        }


def load_labels(path: Path) -> Dict[str, Dict[str, str]]:
    labels: Dict[str, Dict[str, str]] = {}
    for row in iter_jsonl(path):
        label_id = str(row.get("id"))
        answers = row.get("common_answers")
        if not isinstance(answers, list):
            raise ValueError(f"DABENCH label {label_id} has invalid common_answers.")
        labels[label_id] = {str(name): str(value) for name, value in answers}
    return labels


def load_questions(path: Path) -> Dict[str, Dict[str, Any]]:
    questions: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        questions[str(row.get("id"))] = row
    return questions


def extract_format(input_string: str) -> Dict[str, str]:
    return {name: value for name, value in ANSWER_PATTERN.findall(input_string or "")}


def is_equal(response: Any, label: Any) -> bool:
    if response is None:
        return False
    response_text = str(response)
    label_text = str(label)
    if response_text == label_text:
        return True
    try:
        return abs(float(response_text) - float(label_text)) < NUMERIC_TOLERANCE
    except Exception:
        return False


def normalize_dabench_id(value: Any, id_prefix: str = "DA_") -> str:
    text = str(value or "").strip()
    if "__" in text:
        text = text.split("__", 1)[0]
    if id_prefix and text.startswith(id_prefix):
        text = text[len(id_prefix) :]
    return text


def build_reason(is_correct: bool, missing: List[str], wrong: List[str]) -> str:
    if is_correct:
        return "Closed-form DABENCH answer fields all match the official labels."
    parts: List[str] = []
    if missing:
        parts.append("missing fields: " + ", ".join(missing[:20]))
    if wrong:
        parts.append("wrong fields: " + ", ".join(wrong[:20]))
    return "; ".join(parts) or "Closed-form DABENCH answer fields do not match the official labels."
