from __future__ import annotations

from typing import Any, Dict


REQUIRED_TRUE_FIELDS = (
    "attack_effective",
    "task_still_answerable",
    "unique_answer_preserved",
    "normalized_equivalent",
)
FAILURE_CATEGORIES = {
    "attack_not_effective",
    "answer_changed",
    "unanswerable",
    "ambiguity_introduced",
    "evidence_lost",
    "invalid_file",
    "unverifiable",
    "judge_error",
}

INTERPRETATION_RISK_ATTACKS = {
    "header_synonym_substitution",
    "semantic_distractor_column",
    "equivalent_value_reencoding",
    "unit_scale_conversion",
}

INTERPRETATION_RISK_TEXT_FIELDS = (
    "correct_interpretation",
    "plausible_incorrect_interpretation",
    "why_plausible",
    "incorrect_outcome",
    "recoverability_evidence",
    "output_difference_evidence",
)


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_answer(value: Any) -> bool:
    """Judges may return a numeric counterfactual for numeric table tasks."""
    return value is not None and bool(str(value).strip())


def validate_robustness_payload(
    payload: Dict[str, Any],
    attack_type: str = "",
    construct_payload: Dict[str, Any] | None = None,
) -> bool:
    if payload.get("verdict") != "passed" or not all(
        payload.get(field) is True for field in REQUIRED_TRUE_FIELDS
    ):
        return False
    if payload.get("failure_category") is not None or payload.get("failure_reason") is not None:
        return False
    if not str(payload.get("original_answer", "")).strip():
        return False
    if not str(payload.get("attacked_answer", "")).strip():
        return False
    if not str(payload.get("equivalence_evidence", "")).strip():
        return False
    if not isinstance(payload.get("checked_evidence"), dict) or not payload["checked_evidence"]:
        return False
    reference = payload.get("reference_comparison")
    if not isinstance(reference, dict) or reference.get("matches") is not True:
        return False
    if not _has_text(reference.get("method")):
        return False
    if not isinstance(reference.get("differences"), list) or reference["differences"]:
        return False
    if attack_type in {"header_synonym_substitution", "semantic_distractor_column"}:
        bindings = payload.get("field_binding_audit")
        if not isinstance(bindings, list) or not bindings:
            return False
        for binding in bindings:
            if not isinstance(binding, dict) or binding.get("binding_unique") is not True:
                return False
            if not _has_text(binding.get("question_concept")):
                return False
            if not _has_text(binding.get("selected_field")):
                return False
            if not isinstance(binding.get("alternative_fields"), list):
                return False
            if not _has_text(binding.get("exclusion_evidence")):
                return False
    if attack_type == "header_synonym_substitution":
        audits = payload.get("synonym_audit")
        if not isinstance(audits, list) or not audits:
            return False
        required = ("same_concept", "same_metric_scope", "same_granularity", "same_time_basis", "same_unit")
        for audit in audits:
            if not isinstance(audit, dict) or not all(audit.get(field) is True for field in required):
                return False
            if audit.get("can_coexist_as_distinct") is not False:
                return False
            if not _has_text(audit.get("old_header")) or not _has_text(audit.get("new_header")):
                return False
            if not _has_text(audit.get("evidence")):
                return False
    if attack_type == "semantic_distractor_column" and not _has_answer(payload.get("counterfactual_answer")):
        return False
    if attack_type in INTERPRETATION_RISK_ATTACKS and not _valid_interpretation_risk_audit(payload.get("interpretation_risk_audit")):
        return False
    if attack_type == "decoy_feature_pack_injection":
        audits = payload.get("decoy_feature_audit")
        required = ("type_compatible", "uniquely_excludable", "misuse_result_differs")
        if not _has_answer(payload.get("counterfactual_answer")) or not isinstance(audits, list) or not audits:
            return False
        if construct_payload is not None and len(audits) != _l4_declared_count(construct_payload, "feature_pack", "added_features"):
            return False
        for audit in audits:
            if not isinstance(audit, dict) or not _has_text(audit.get("feature_name")):
                return False
            if not all(audit.get(field) is True for field in required) or not _has_text(audit.get("evidence")):
                return False
    if attack_type == "non_observation_row_injection":
        audits = payload.get("non_observation_row_audit")
        required = ("marker_present", "non_observation_verified", "marker_uniquely_excludes", "misuse_result_differs")
        if not isinstance(audits, list) or not audits:
            return False
        if construct_payload is not None and len(audits) != _l4_declared_count(construct_payload, "non_observation_rows"):
            return False
        for audit in audits:
            if not isinstance(audit, dict) or not _has_text(audit.get("record_identifier")):
                return False
            if not all(audit.get(field) is True for field in required) or not _has_text(audit.get("evidence")):
                return False
    return True


def validation_failure_reason(
    payload: Dict[str, Any],
    attack_type: str = "",
    construct_payload: Dict[str, Any] | None = None,
) -> str:
    if not payload:
        return "empty validation payload"
    if payload.get("failure_reason"):
        return str(payload["failure_reason"])
    failed = [field for field in REQUIRED_TRUE_FIELDS if payload.get(field) is not True]
    if payload.get("verdict") != "passed":
        failed.insert(0, "verdict")
    if payload.get("verdict") == "passed":
        for field in ("original_answer", "attacked_answer", "equivalence_evidence", "checked_evidence", "reference_comparison"):
            if not payload.get(field):
                failed.append(field)
        reference = payload.get("reference_comparison")
        if isinstance(reference, dict) and (
            reference.get("matches") is not True or reference.get("differences")
        ):
            failed.append("reference_comparison")
        if payload.get("failure_category") is not None:
            failed.append("failure_category")
        if payload.get("failure_reason") is not None:
            failed.append("failure_reason")
        if attack_type in {"header_synonym_substitution", "semantic_distractor_column"}:
            if not payload.get("field_binding_audit"):
                failed.append("field_binding_audit")
        if attack_type == "header_synonym_substitution" and not payload.get("synonym_audit"):
            failed.append("synonym_audit")
        if attack_type == "semantic_distractor_column" and not _has_answer(payload.get("counterfactual_answer")):
            failed.append("counterfactual_answer")
        if attack_type in INTERPRETATION_RISK_ATTACKS and not _valid_interpretation_risk_audit(payload.get("interpretation_risk_audit")):
            failed.append("interpretation_risk_audit")
        if attack_type == "decoy_feature_pack_injection":
            if not payload.get("decoy_feature_audit"):
                failed.append("decoy_feature_audit")
            if not _has_answer(payload.get("counterfactual_answer")):
                failed.append("counterfactual_answer")
        if attack_type == "non_observation_row_injection" and not payload.get("non_observation_row_audit"):
            failed.append("non_observation_row_audit")
        expected_key = {
            "decoy_feature_pack_injection": ("decoy_feature_audit", "feature_pack"),
            "non_observation_row_injection": ("non_observation_row_audit", "non_observation_rows"),
        }.get(attack_type)
        if construct_payload is not None and expected_key:
            audit_key, declaration_key = expected_key
            audits = payload.get(audit_key)
            expected = _l4_declared_count(construct_payload, "feature_pack", "added_features") if declaration_key == "feature_pack" else _l4_declared_count(construct_payload, declaration_key)
            if isinstance(audits, list) and len(audits) != expected:
                failed.append(f"{audit_key}_count_mismatch")
    category = payload.get("failure_category")
    if category is not None and category not in FAILURE_CATEGORIES:
        failed.append("invalid_failure_category")
    return "validation did not pass: " + ", ".join(dict.fromkeys(failed))


def _valid_interpretation_risk_audit(audit: Any) -> bool:
    if not isinstance(audit, dict) or audit.get("outcome_differs") is not True:
        return False
    return all(_has_text(audit.get(field)) for field in INTERPRETATION_RISK_TEXT_FIELDS)


def _l4_declared_count(construct_payload: Dict[str, Any], key: str, nested_key: str | None = None) -> int:
    record = construct_payload.get("transformation_record") or {}
    contract = record.get("semantic_contract") if isinstance(record, dict) else None
    values = contract.get(key) if isinstance(contract, dict) else None
    if nested_key is not None and isinstance(values, dict):
        values = values.get(nested_key)
    return len(values) if isinstance(values, list) else 0


def _l4_record_declaration_count(construct_payload: Dict[str, Any]) -> int:
    return _l4_declared_count(construct_payload, "injected_records") + _l4_declared_count(construct_payload, "injected_record_groups")
