from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.agent import AgentFatalError
from src.construct_dataset import construct_and_validate, rebuild_attack_dataset
from src.dataset import profile_files
from src.operators import ROBUSTNESS_OPERATORS


class ConstructValidateAgent:
    def __init__(self, question: str):
        self.question = question
        self.stages = []

    def complete_json(self, _prompt, workspace=None, stage="construct", allow_tools=True):
        if "{{" in _prompt or "}}" in _prompt:
            raise AssertionError("prompt contains unresolved template placeholders")
        self.stages.append(stage)
        if stage == "construct":
            source = workspace.original / "data.csv"
            target = workspace.final / "data.csv"
            with source.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            with target.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows([rows[0], *reversed(rows[1:])])
            return (
                {
                    "status": "constructed",
                    "attack_type": "row_order_shuffle",
                    "new_question": self.question,
                    "file_edit_required": True,
                    "output_files": ["data.csv"],
                    "input_file": "data.csv",
                    "edit_summary": "reversed five data rows",
                    "transformation_record": {
                        "targets": ["data.csv"],
                        "parameters": {"strategy": "reverse"},
                        "mapping": {"old_positions": [1, 2, 3, 4, 5], "new_positions": [5, 4, 3, 2, 1]},
                        "verification": {"method": "row multiset", "result": "equal"},
                    },
                    "quality_check": {
                        "question_unchanged": True,
                        "attack_effective": True,
                        "necessary_evidence_preserved": True,
                        "unique_answer_preserved": True,
                        "answer_equivalent": True,
                        "no_new_ambiguity_or_conflict": True,
                        "files_readable": True,
                    },
                    "reject_reason": None,
                },
                [],
            )
        return (
            {
                "verdict": "passed",
                "attack_effective": True,
                "task_still_answerable": True,
                "unique_answer_preserved": True,
                "normalized_equivalent": True,
                "original_answer": "150",
                "attacked_answer": "150",
                "equivalence_evidence": "same row multiset and sum",
                "checked_evidence": {"transformation_checks": ["row multiset equal"]},
                "reference_comparison": {"matches": True, "method": "exact comparison", "differences": []},
                "failure_category": None,
                "failure_reason": None,
            },
            [],
        )


class PassingJudgeAgent:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0
        self.prompts = []

    def complete_json(self, prompt, workspace=None, stage="judge", allow_tools=True):
        self.calls += 1
        self.prompts.append(prompt)
        if stage != "judge":
            raise AssertionError(f"judge received unexpected stage: {stage}")
        return (
            {
                "verdict": "passed",
                "attack_effective": True,
                "task_still_answerable": True,
                "unique_answer_preserved": True,
                "normalized_equivalent": True,
                "original_answer": self.answer,
                "attacked_answer": self.answer,
                "equivalence_evidence": "same typed row multiset and reference",
                "checked_evidence": {"transformation_checks": ["row multiset equal"]},
                "reference_comparison": {"matches": True, "method": "exact comparison", "differences": []},
                "failure_category": None,
                "failure_reason": None,
            },
            [],
        )


class RepairingConstructAgent:
    def __init__(self, question: str):
        self.question = question
        self.construct_calls = 0
        self.prompts = []

    def complete_json(self, prompt, workspace=None, stage="construct", allow_tools=True):
        self.prompts.append(prompt)
        if stage == "validate":
            return (
                {
                    "verdict": "passed",
                    "attack_effective": True,
                    "task_still_answerable": True,
                    "unique_answer_preserved": True,
                    "normalized_equivalent": True,
                    "original_answer": "30",
                    "attacked_answer": "30",
                    "equivalence_evidence": "same typed row multiset",
                    "checked_evidence": {"transformation_checks": ["exact"]},
                    "reference_comparison": {"matches": True, "method": "exact comparison", "differences": []},
                    "failure_category": None,
                    "failure_reason": None,
                },
                [],
            )

        self.construct_calls += 1
        with (workspace.original / "data.csv").open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        output_rows = [rows[0], *reversed(rows[1:])]
        if self.construct_calls == 1:
            output_rows[1][1] = "2"
            output_rows[2][1] = "1"
        with (workspace.final / "data.csv").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(output_rows)
        return (
            {
                "status": "constructed",
                "attack_type": "row_order_shuffle",
                "new_question": self.question,
                "file_edit_required": True,
                "output_files": ["data.csv"],
                "input_file": "data.csv",
                "edit_summary": "reversed rows",
                "transformation_record": {"targets": ["data.csv"], "verification": {"result": "claimed"}},
                "quality_check": {
                    "question_unchanged": True,
                    "attack_effective": True,
                    "necessary_evidence_preserved": True,
                    "unique_answer_preserved": True,
                    "answer_equivalent": True,
                    "no_new_ambiguity_or_conflict": True,
                    "files_readable": True,
                },
                "reject_reason": None,
            },
            [],
        )


class SemanticRepairAgent(ConstructValidateAgent):
    def __init__(self, question: str):
        super().__init__(question)
        self.construct_calls = 0
        self.validate_calls = 0
        self.prompts = []

    def complete_json(self, prompt, workspace=None, stage="construct", allow_tools=True):
        self.prompts.append(prompt)
        if stage == "construct":
            self.construct_calls += 1
            return super().complete_json(prompt, workspace=workspace, stage=stage, allow_tools=allow_tools)
        self.validate_calls += 1
        if self.validate_calls == 1:
            return (
                {
                    "verdict": "failed",
                    "attack_effective": False,
                    "task_still_answerable": True,
                    "unique_answer_preserved": True,
                    "normalized_equivalent": True,
                    "original_answer": "150",
                    "attacked_answer": "150",
                    "equivalence_evidence": "rows preserved but perturbation judged too weak",
                    "checked_evidence": {"transformation_checks": ["insufficient semantic effect"]},
                    "reference_comparison": {"matches": True, "method": "exact comparison", "differences": []},
                    "failure_category": "attack_not_effective",
                    "failure_reason": "perturbation needs to be redesigned",
                },
                [],
            )
        return super().complete_json(prompt, workspace=workspace, stage=stage, allow_tools=allow_tools)


class SemanticRepairJudge(PassingJudgeAgent):
    def complete_json(self, prompt, workspace=None, stage="judge", allow_tools=True):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return (
                {
                    "verdict": "failed",
                    "attack_effective": False,
                    "task_still_answerable": True,
                    "unique_answer_preserved": True,
                    "normalized_equivalent": True,
                    "original_answer": self.answer,
                    "attacked_answer": self.answer,
                    "equivalence_evidence": "rows preserved but perturbation judged too weak",
                    "checked_evidence": {"transformation_checks": ["insufficient semantic effect"]},
                    "reference_comparison": {"matches": True, "method": "exact comparison", "differences": []},
                    "failure_category": "attack_not_effective",
                    "failure_reason": "perturbation needs to be redesigned",
                },
                [],
            )
        self.calls -= 1
        return super().complete_json(prompt, workspace=workspace, stage=stage, allow_tools=allow_tools)


class FailingJudge:
    def __init__(self):
        self.calls = 0

    def complete_json(self, *_args, **_kwargs):
        self.calls += 1
        raise AgentFatalError("judge endpoint unavailable")


class IntegrationTests(unittest.TestCase):
    def test_construct_validate_and_rebuild_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.csv"
            source.write_text("id,value\n1,10\n2,20\n3,30\n4,40\n5,50\n", encoding="utf-8")
            output = root / "outputs"
            for name in ["accepted", "rejected", "candidates", "logs", "workspaces"]:
                (output / name).mkdir(parents=True)

            question = "Compute the sum of the value column."
            row = {
                "id": "S1",
                "question": question,
                "reference": "150",
                "input_file": "data.csv",
            }
            operator = ROBUSTNESS_OPERATORS["row_order_shuffle"]
            constructor = ConstructValidateAgent(question)
            judge = PassingJudgeAgent("150")
            context = {
                "project_root": Path(__file__).resolve().parents[1],
                "output_dir": output,
                "agent": constructor,
                "judge_agent": judge,
                "judge_available": True,
                "config": {"runner": {"log_level": "quiet"}, "judge_model": {"name": "test-judge"}},
            }
            result = construct_and_validate(
                row=row,
                source_files=[source],
                source_profile=profile_files([source]),
                operator=operator,
                selection={"reason": "order-independent aggregate"},
                new_id="S1__row_order_shuffle__001",
                context=context,
            )
            self.assertEqual("accepted", result["status"])
            accepted = output / "accepted" / "S1__row_order_shuffle__001.json"
            self.assertTrue(accepted.exists())
            record = json.loads(accepted.read_text(encoding="utf-8"))
            self.assertEqual("150", record["answer"])
            self.assertEqual("invariant", record["answer_relation"])
            self.assertEqual(["data.csv"], record["construction"]["file_diff"]["changed"])
            self.assertEqual(["construct"], constructor.stages)
            self.assertEqual(1, judge.calls)
            self.assertNotIn("transformation_record", judge.prompts[0])

            rebuild_attack_dataset(output)
            lines = (output / "attack_dataset.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(lines))
            public = json.loads(lines[0])
            self.assertNotIn("construct_history", public["construction"])
            self.assertNotIn("validate_history", public["construction"])
            self.assertNotIn("judge_history", public["construction"])

    def test_host_check_feedback_repairs_constructed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.csv"
            source.write_text("id,category,value\n1,01,10\n2,02,20\n", encoding="utf-8")
            output = root / "outputs"
            for name in ["accepted", "rejected", "candidates", "logs", "workspaces"]:
                (output / name).mkdir(parents=True)

            question = "Sum value."
            row = {"id": "S2", "question": question, "reference": "30", "input_file": "data.csv"}
            operator = ROBUSTNESS_OPERATORS["row_order_shuffle"]
            agent = RepairingConstructAgent(question)
            judge = PassingJudgeAgent("30")
            context = {
                "project_root": Path(__file__).resolve().parents[1],
                "output_dir": output,
                "agent": agent,
                "judge_agent": judge,
                "judge_available": True,
                "config": {"runner": {"log_level": "quiet", "construct_repair_attempts": 2}, "judge_model": {"name": "test-judge"}},
            }
            result = construct_and_validate(
                row=row,
                source_files=[source],
                source_profile=profile_files([source]),
                operator=operator,
                selection={"reason": "order-independent"},
                new_id="S2__row_order_shuffle__001",
                context=context,
            )

            self.assertEqual("accepted", result["status"])
            self.assertEqual(2, agent.construct_calls)
            self.assertIn("Framework integrity validation failed", agent.prompts[1])
            candidate = json.loads(
                (output / "candidates" / "S2__row_order_shuffle__001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(candidate["construct_attempts"]))
            self.assertIn("data_row_multiset_changed", candidate["construct_attempts"][0]["host_check_error"])
            self.assertIsNone(candidate["construct_attempts"][1]["host_check_error"])
            self.assertEqual(1, candidate["construct"]["host_repair_attempts_used"])

    def test_semantic_failure_reconstructs_once_and_revalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.csv"
            source.write_text("id,value\n1,10\n2,20\n3,30\n4,40\n5,50\n", encoding="utf-8")
            output = root / "outputs"
            for name in ["accepted", "rejected", "candidates", "logs", "workspaces"]:
                (output / name).mkdir(parents=True)

            question = "Compute the sum of the value column."
            row = {"id": "S3", "question": question, "reference": "150", "input_file": "data.csv"}
            operator = ROBUSTNESS_OPERATORS["row_order_shuffle"]
            agent = SemanticRepairAgent(question)
            judge = SemanticRepairJudge("150")
            context = {
                "project_root": Path(__file__).resolve().parents[1],
                "output_dir": output,
                "agent": agent,
                "judge_agent": judge,
                "judge_available": True,
                "config": {
                    "runner": {
                        "log_level": "quiet",
                        "construct_repair_attempts": 1,
                        "semantic_repair_attempts": 1,
                    },
                    "judge_model": {"name": "test-judge"},
                },
            }
            result = construct_and_validate(
                row=row,
                source_files=[source],
                source_profile=profile_files([source]),
                operator=operator,
                selection={"reason": "order-independent aggregate"},
                new_id="S3__row_order_shuffle__001",
                context=context,
            )

            self.assertEqual("accepted", result["status"])
            self.assertEqual(2, agent.construct_calls)
            self.assertEqual(2, judge.calls)
            self.assertTrue(any("Independent semantic validation failed" in prompt for prompt in agent.prompts))
            candidate = json.loads(
                (output / "candidates" / "S3__row_order_shuffle__001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(candidate["construct_attempts"]))
            self.assertEqual(2, len(candidate["validation_attempts"]))

    def test_judge_error_rejects_without_semantic_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.csv"
            source.write_text("id,value\n1,10\n2,20\n3,30\n4,40\n", encoding="utf-8")
            output = root / "outputs"
            for name in ["accepted", "rejected", "candidates", "logs", "workspaces"]:
                (output / name).mkdir(parents=True)
            question = "Sum value."
            constructor = ConstructValidateAgent(question)
            judge = FailingJudge()
            result = construct_and_validate(
                row={"id": "S4", "question": question, "reference": "100", "input_file": "data.csv"},
                source_files=[source],
                source_profile=profile_files([source]),
                operator=ROBUSTNESS_OPERATORS["row_order_shuffle"],
                selection={"reason": "order-independent aggregate"},
                new_id="S4__row_order_shuffle__001",
                context={
                    "project_root": Path(__file__).resolve().parents[1],
                    "output_dir": output,
                    "agent": constructor,
                    "judge_agent": judge,
                    "judge_available": True,
                    "config": {"runner": {"semantic_repair_attempts": 1}, "judge_model": {"name": "test-judge"}},
                },
            )
            self.assertEqual("rejected", result["status"])
            self.assertEqual(1, constructor.stages.count("construct"))
            self.assertEqual(1, judge.calls)
            record = json.loads((output / "rejected" / "S4__row_order_shuffle__001.json").read_text(encoding="utf-8"))
            self.assertEqual("judge_error", record["construction"]["validation"]["failure_category"])


if __name__ == "__main__":
    unittest.main()
