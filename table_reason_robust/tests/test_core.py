from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.agent import AgentFatalError, OpenAIWorkspaceAgent
from src.construct_dataset import (
    REQUIRED_QUALITY_FLAGS,
    canonicalize_construct_question,
    constrain_l4_selection,
    minimal_judge_feedback,
    process_row,
    render_judge_validate_prompt,
    stable_attack_id,
    summarize_resume,
    validate_construct_payload,
    validate_judge_config,
    write_json,
    log_event,
)
from src.dataset import build_dabench_question, load_dabench_rows, profile_files
from src.operators import ROBUSTNESS_OPERATORS, get_enabled_operators
from src.runner import Runner
from src.validators import validate_robustness_payload, validation_failure_reason
from src.workspace import AttackWorkspace


class StaticAgent:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, *_args, **_kwargs):
        self.calls += 1
        return self.payload, []


class FatalAgent:
    def complete_json(self, *_args, **_kwargs):
        raise AgentFatalError("temporary service overload")


class CoreTests(unittest.TestCase):
    def test_all_operators_registered_and_prompts_exist(self):
        self.assertEqual(12, len(ROBUSTNESS_OPERATORS))
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            {"decoy_feature_pack_injection", "non_observation_row_injection"},
            {name for name in ROBUSTNESS_OPERATORS if name.endswith("injection")},
        )
        for operator in ROBUSTNESS_OPERATORS.values():
            self.assertEqual("robustness", operator.dimension)
            self.assertTrue((root / operator.construct_prompt).exists(), operator.name)
            self.assertTrue((root / operator.instruction_prompt).exists(), operator.name)

    def test_l4_prompt_contracts_are_distinct(self):
        root = Path(__file__).resolve().parents[1] / "prompts" / "robustness" / "attacks"
        feature_pack = (root / "decoy_feature_pack_injection.md").read_text(encoding="utf-8")
        non_observation = (root / "non_observation_row_injection.md").read_text(encoding="utf-8")
        self.assertIn("feature_pack", feature_pack)
        self.assertIn("2--5", feature_pack)
        self.assertIn("non_observation_rows", non_observation)
        self.assertIn("marker_column", non_observation)

    def test_selection_prompt_keeps_attack_names_injected_not_global(self):
        prompt = (Path(__file__).resolve().parents[1] / "prompts" / "robustness" / "select_attack.md").read_text(encoding="utf-8")
        self.assertIn("{{allowed_attack_names_json}}", prompt)
        self.assertNotIn("decoy_feature_pack_injection", prompt)
        self.assertNotIn("non_observation_row_injection", prompt)

    def test_common_prompts_do_not_embed_l4_rules(self):
        root = Path(__file__).resolve().parents[1] / "prompts" / "robustness"
        construct = (root / "construct_attack.md").read_text(encoding="utf-8")
        judge = (root / "judge_validate_robustness.md").read_text(encoding="utf-8")
        self.assertNotIn("L4", construct)
        self.assertNotIn("L4", judge)
        self.assertIn("{{judge_requirements}}", judge)
        self.assertFalse((root / "validate_robustness.md").exists())

    def test_construct_payload_and_question_canonicalization(self):
        question = "Calculate total value."
        quality = {name: True for name in REQUIRED_QUALITY_FLAGS}
        payload = {
            "status": "constructed", "new_question": question, "file_edit_required": True,
            "output_files": ["data.csv"], "transformation_record": {"targets": ["data.csv"]},
            "quality_check": quality,
        }
        self.assertIsNone(validate_construct_payload(payload, {"question": question}))
        canonicalize_construct_question(payload, {"question": "Exact question"})
        self.assertEqual("Exact question", payload["new_question"])
        self.assertTrue(payload["question_canonicalized_by_host"])

    def test_validator_l4_audits(self):
        valid = {
            "verdict": "passed", "attack_effective": True, "task_still_answerable": True,
            "unique_answer_preserved": True, "normalized_equivalent": True,
            "original_answer": "10", "attacked_answer": "10", "equivalence_evidence": "same total",
            "checked_evidence": {"fields": ["value"]},
            "reference_comparison": {"matches": True, "method": "exact", "differences": []},
            "failure_category": None, "failure_reason": None,
        }
        feature_audits = [
            {"feature_name": "value_forecast", "type_compatible": True, "uniquely_excludable": True, "misuse_result_differs": True, "evidence": "actual vs forecast"},
            {"feature_name": "value_rank", "type_compatible": True, "uniquely_excludable": True, "misuse_result_differs": True, "evidence": "rank vs raw"},
        ]
        feature_construct = {"transformation_record": {"semantic_contract": {"feature_pack": {"added_features": [{}, {}]}}}}
        self.assertTrue(validate_robustness_payload({**valid, "counterfactual_answer": "12", "decoy_feature_audit": feature_audits}, "decoy_feature_pack_injection", feature_construct))
        self.assertFalse(validate_robustness_payload({**valid, "counterfactual_answer": "", "decoy_feature_audit": feature_audits}, "decoy_feature_pack_injection", feature_construct))

        row_audits = [{"record_identifier": "rows 12-13", "marker_present": True, "non_observation_verified": True, "marker_uniquely_excludes": True, "misuse_result_differs": True, "evidence": "control marker"}]
        row_construct = {"transformation_record": {"semantic_contract": {"non_observation_rows": [{}]}}}
        self.assertTrue(validate_robustness_payload({**valid, "non_observation_row_audit": row_audits}, "non_observation_row_injection", row_construct))
        self.assertFalse(validate_robustness_payload({**valid, "non_observation_row_audit": [{**row_audits[0], "marker_present": False}]}, "non_observation_row_injection", row_construct))

    def test_validator_requires_interpretation_risk_for_l1_l2(self):
        valid = {
            "verdict": "passed", "attack_effective": True, "task_still_answerable": True,
            "unique_answer_preserved": True, "normalized_equivalent": True,
            "original_answer": "10", "attacked_answer": "10", "equivalence_evidence": "same total",
            "checked_evidence": {"fields": ["value"]},
            "reference_comparison": {"matches": True, "method": "exact", "differences": []},
            "failure_category": None, "failure_reason": None,
        }
        risk = {
            "correct_interpretation": "Use the semantically bound field.",
            "plausible_incorrect_interpretation": "Use a nearby field with the same value family.",
            "why_plausible": "Both fields share unit and entity granularity.",
            "incorrect_outcome": "12",
            "outcome_differs": True,
            "recoverability_evidence": "Question scope uniquely selects the correct field.",
            "output_difference_evidence": "The requested rounded output is 10 versus 12.",
        }
        synonym_audit = [{
            "old_header": "Amount", "new_header": "Value", "same_concept": True,
            "same_metric_scope": True, "same_granularity": True, "same_time_basis": True,
            "same_unit": True, "can_coexist_as_distinct": False, "evidence": "same metric",
        }]
        binding = [{
            "question_concept": "reported amount", "selected_field": "Value", "alternative_fields": [],
            "binding_unique": True, "exclusion_evidence": "scope is explicit",
        }]
        self.assertFalse(validate_robustness_payload({**valid, "field_binding_audit": binding, "synonym_audit": synonym_audit}, "header_synonym_substitution"))
        self.assertTrue(validate_robustness_payload({**valid, "field_binding_audit": binding, "synonym_audit": synonym_audit, "interpretation_risk_audit": risk}, "header_synonym_substitution"))
        self.assertTrue(validate_robustness_payload({**valid, "interpretation_risk_audit": risk}, "unit_scale_conversion"))
        self.assertFalse(validate_robustness_payload({**valid, "interpretation_risk_audit": {**risk, "outcome_differs": False}}, "unit_scale_conversion"))

    def test_validation_reason_reports_reference_difference(self):
        payload = {
            "verdict": "passed", "attack_effective": True, "task_still_answerable": True,
            "unique_answer_preserved": True, "normalized_equivalent": True,
            "original_answer": "10", "attacked_answer": "10", "equivalence_evidence": "same",
            "checked_evidence": {"fields": ["value"]},
            "reference_comparison": {"matches": True, "method": "exact", "differences": [{"field": "rounded total"}]},
            "failure_category": None, "failure_reason": None,
        }
        self.assertIn("reference_comparison", validation_failure_reason(payload))

    def test_workspace_prepare_retires_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            source.write_text("id,value\n1,10\n", encoding="utf-8")
            workspace = AttackWorkspace(root / "outputs", "sample__attack__001", "sample", "attack")
            workspace.prepare([source])
            (workspace.final / "stale.csv").write_text("stale", encoding="utf-8")
            workspace.prepare([source])
            self.assertFalse((workspace.final / "stale.csv").exists())
            self.assertTrue((workspace.final / "source.csv").exists())

    def test_workspace_mapping_and_file_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            source.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
            workspace = AttackWorkspace(root / "outputs", "sample__attack__001", "sample", "attack")
            workspace.prepare([source])
            self.assertEqual(workspace.original, workspace.mapping("validate")["/mnt/original"])
            (workspace.final / "source.csv").write_text("id,value\n2,20\n1,10\n", encoding="utf-8")
            self.assertEqual(["source.csv"], workspace.file_diff()["changed"])

    def test_judge_config_requires_a_distinct_identity(self):
        constructor = {"base_url": "https://example.test/v1/", "api_key": "a", "name": "Qwen"}
        with self.assertRaisesRegex(RuntimeError, "Missing required"):
            validate_judge_config(constructor, None)
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            validate_judge_config(constructor, {"base_url": "https://example.test/v1", "api_key": "b", "name": "qwen"})
        validate_judge_config(constructor, {"base_url": "https://example.test/v1", "api_key": "b", "name": "deepseek"})
        validate_judge_config(
            constructor,
            {"base_url": "https://example.test/v1", "api_key": "b", "name": "qwen"},
            allow_same=True,
        )

    def test_dabench_loader_merges_official_rows_and_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            labels = root / "labels.jsonl"
            tables = root / "tables"
            tables.mkdir()
            (tables / "source.csv").write_text("id,value\n01,10\n", encoding="utf-8-sig")
            questions.write_text(json.dumps({
                "id": 7, "question": "Find value.", "constraints": "Keep leading zeros.",
                "format": "@value[x]", "file_name": "source.csv", "concepts": ["lookup"], "level": "easy",
            }) + "\n", encoding="utf-8")
            labels.write_text(json.dumps({"id": 7, "common_answers": [["value", "10"]]}) + "\n", encoding="utf-8")
            rows = load_dabench_rows(questions, labels, tables)
            self.assertEqual("DA_7", rows[0]["id"])
            self.assertIn("Constraints:\nKeep leading zeros.", rows[0]["question"])
            self.assertIn("Required output format:\n@value[x]", rows[0]["question"])
            self.assertEqual({"common_answers": [["value", "10"]]}, json.loads(rows[0]["reference"]))
            self.assertEqual("DABENCH", rows[0]["metadata"]["source"])
            (tables / "source.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing CSV"):
                load_dabench_rows(questions, labels, tables)

    def test_dabench_loader_rejects_unmatched_labels_and_selected_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tables = root / "tables"; tables.mkdir()
            (tables / "source.csv").write_text("a\n1\n", encoding="utf-8")
            questions = root / "questions.jsonl"; labels = root / "labels.jsonl"
            questions.write_text('{"id": 1, "question": "Q", "file_name": "source.csv"}\n', encoding="utf-8")
            labels.write_text('{"id": 2, "common_answers": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no matching label"):
                load_dabench_rows(questions, labels, tables)
            labels.write_text('{"id": 1, "common_answers": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "selected sample ids"):
                load_dabench_rows(questions, labels, tables, selected_ids=["DA_2"])

    def test_dabench_question_omits_empty_sections(self):
        self.assertEqual("Question", build_dabench_question({"question": "Question", "constraints": "", "format": None}))

    def test_log_event_tolerates_closed_stdout(self):
        import src.construct_dataset as module
        original = module.tqdm.write
        try:
            module.tqdm.write = lambda _message: (_ for _ in ()).throw(OSError(22, "closed"))
            log_event("background completion", context={"config": {"runner": {"log_level": "verbose"}}})
        finally:
            module.tqdm.write = original

    def test_judge_prompt_is_blind_and_workspace_is_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.csv"
            source.write_text("id,value\n01,10\n02,20\n", encoding="utf-8")
            workspace = AttackWorkspace(root / "outputs", "sample__row_order_shuffle__001", "sample", "row_order_shuffle")
            workspace.prepare([source])
            final = workspace.final / "data.csv"
            final.write_text("id,value\n02,20\n01,10\n", encoding="utf-8")
            workspace.prepare_judge_snapshot(1)
            prompt = render_judge_validate_prompt(
                {"question": "Sum value", "reference": "30"},
                profile_files([workspace.original / "data.csv"], virtual_root="/mnt/original"),
                profile_files([final]),
                ROBUSTNESS_OPERATORS["row_order_shuffle"],
                workspace,
                {"project_root": Path(__file__).resolve().parents[1]},
                {"file_diff": workspace.file_diff()},
            )
            self.assertNotIn("construction_result_json", prompt)
            self.assertNotIn("semantic_contract", prompt)
            self.assertNotIn("edit_summary", prompt)
            self.assertIn("/mnt/original/data.csv", prompt)
            self.assertIn("/mnt/data/data.csv", prompt)
            snapshot_file = workspace.mapping("judge")["/mnt/data"] / "data.csv"
            snapshot_file.write_text("corrupted", encoding="utf-8")
            self.assertIn("02,20", final.read_text(encoding="utf-8"))

    def test_minimal_judge_feedback_does_not_leak_answers_or_audits(self):
        feedback = minimal_judge_feedback({
            "failure_category": "answer_changed",
            "failure_reason": "source=10, attacked=20",
            "original_answer": "10",
            "attacked_answer": "20",
            "counterfactual_answer": "30",
            "decoy_feature_audit": [{"feature_name": "value_forecast"}],
            "checked_evidence": {"fields": ["value"], "unknown": "hidden"},
        })
        self.assertEqual("answer_changed", feedback["failure_category"])
        self.assertEqual({"fields": ["value"]}, feedback["checked_evidence"])
        self.assertNotIn("original_answer", feedback)
        self.assertNotIn("decoy_feature_audit", feedback)

    def test_unlimited_tool_budget_keeps_tools_available(self):
        agent = OpenAIWorkspaceAgent(api_key="test-key", base_url="https://example.invalid/v1", model_name="test-model", max_tool_calls_per_stage=None)
        seen = []
        tool_call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="execute_code", arguments='{"code":"print(1)"}'))
        replies = [SimpleNamespace(content="", tool_calls=[tool_call]), SimpleNamespace(content='{"ok": true}', tool_calls=None)]
        agent._chat = lambda _messages, tools: (seen.append(tools), replies.pop(0))[1]
        agent._handle_tool_call = lambda *_args, **_kwargs: "Execution succeeded"
        payload, _ = agent.complete_json("test", workspace=object(), stage="construct", allow_tools=True)
        self.assertEqual({"ok": True}, payload)
        self.assertTrue(all(item is not None for item in seen))

    def test_resume_is_per_attack(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "logs").mkdir()
            first = ROBUSTNESS_OPERATORS["row_order_shuffle"]
            second = ROBUSTNESS_OPERATORS["column_order_shuffle"]
            (output / "logs" / "QA_1.json").write_text(json.dumps({"id": "QA_1", "attacks": {first.name: {"status": "accepted"}}}), encoding="utf-8")
            self.assertEqual(1, summarize_resume([{"id": "QA_1"}], output, [first])["completed"])
            self.assertEqual(0, summarize_resume([{"id": "QA_1"}], output, [first, second])["completed"])
            self.assertEqual("QA_1__column_order_shuffle__001", stable_attack_id("QA_1", second.name))

    def test_unknown_operator_is_rejected(self):
        with self.assertRaises(KeyError):
            get_enabled_operators(["not_an_attack"])

    def test_l4_selection_is_constrained_to_one_physical_target(self):
        selection = {"target": {"files": ["RA.xlsx", "NonRA.xlsx"], "sheets": ["Data", "Archive"]}}
        constrained = constrain_l4_selection(selection, "decoy_feature_pack_injection")
        self.assertEqual(["RA.xlsx"], constrained["target"]["files"])
        self.assertEqual(["Data"], constrained["target"]["sheets"])
        self.assertEqual(["RA.xlsx", "NonRA.xlsx"], selection["target"]["files"])

    def test_json_state_write_creates_parent_and_replaces_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "state.json"
            write_json(path, {"round": 1})
            write_json(path, {"round": 2})
            self.assertEqual({"round": 2}, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual([], list(path.parent.glob("*.tmp")))

    def test_runner_keeps_other_samples_after_worker_exception(self):
        def process(row):
            if row["id"] == "bad":
                raise OSError("transient lock")
            return {"id": row["id"], "status": "ok"}

        results = Runner(num_workers=2, stall_timeout_sec=2).run(
            [{"id": "good-1"}, {"id": "bad"}, {"id": "good-2"}], process
        )
        by_id = {item["id"]: item for item in results}
        self.assertEqual("ok", by_id["good-1"]["status"])
        self.assertEqual("runner_error", by_id["bad"]["status"])
        self.assertEqual("ok", by_id["good-2"]["status"])


if __name__ == "__main__":
    unittest.main()
