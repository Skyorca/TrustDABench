from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


HARD_ATTACKS = {
    "deep_analysis_missing",
    "structural_context_missing",
}

# 这些字段直接决定样本是否真实有效，仍作为硬门槛。
CORE_TRUE_FIELDS = (
    "independent_check_completed",
    "original_answerable",
    "attack_rule_valid",
    "question_preservation_valid",
    "file_edit_scope_valid",
    "unique_answer_impossible",
    "no_alternative_answer_path",
)


def _has_items(value: Any) -> bool:
    """判断验证器是否给出了实际检查对象，而不要求固定证明格式。"""
    if isinstance(value, (list, tuple, set, dict, str)):
        return bool(value)
    return value is not None


def _first_failed_true_field(
    payload: Dict[str, Any],
    fields: Iterable[str],
) -> Optional[str]:
    for field in fields:
        if payload.get(field) is not True:
            return field
    return None


def validate_unanswerable_payload(
    payload: Dict[str, Any],
    attack_type: Optional[str] = None,
) -> bool:
    """验证不可答攻击的核心质量，不因非关键证明细节拒绝样本。"""
    if not isinstance(payload, dict):
        return False

    if payload.get("verdict") != "passed":
        return False

    if _first_failed_true_field(payload, CORE_TRUE_FIELDS):
        return False

    # Hard攻击必须额外满足难度定义；Easy攻击不检查该字段。
    if attack_type in HARD_ATTACKS and payload.get("hardness_valid") is not True:
        return False

    if payload.get("can_answer") is not False:
        return False
    if payload.get("should_refuse") is not True:
        return False
    if payload.get("fabricated_answer") is True:
        return False

    checked = payload.get("checked_evidence")
    if not isinstance(checked, dict):
        return False

    # 独立验证至少应实际查看原始输入和扰动后输入。
    # 不再强制每个证据子列表都非空，也不要求穷举所有替代路径。
    if not _has_items(checked.get("original_files_checked")):
        return False
    if not _has_items(checked.get("final_files_checked")):
        return False

    return True


def validation_failure_reason(
    payload: Dict[str, Any],
    attack_type: Optional[str] = None,
) -> str:
    if not payload:
        return "empty validation payload"

    if payload.get("verdict") != "passed":
        return str(
            payload.get("failure_reason")
            or payload.get("refusal_reason")
            or f"validator verdict={payload.get('verdict')!r}"
        )

    failed = _first_failed_true_field(payload, CORE_TRUE_FIELDS)
    if failed:
        return f"validator field failed: {failed}={payload.get(failed)!r}"

    if attack_type in HARD_ATTACKS and payload.get("hardness_valid") is not True:
        return (
            "validator field failed: "
            f"hardness_valid={payload.get('hardness_valid')!r}"
        )

    if payload.get("can_answer") is not False:
        return f"validator field failed: can_answer={payload.get('can_answer')!r}"
    if payload.get("should_refuse") is not True:
        return (
            "validator field failed: "
            f"should_refuse={payload.get('should_refuse')!r}"
        )
    if payload.get("fabricated_answer") is True:
        return "validator field failed: fabricated_answer=True"

    checked = payload.get("checked_evidence")
    if not isinstance(checked, dict):
        return "validator evidence missing: checked_evidence"
    if not _has_items(checked.get("original_files_checked")):
        return "validator evidence missing: original_files_checked"
    if not _has_items(checked.get("final_files_checked")):
        return "validator evidence missing: final_files_checked"

    return str(
        payload.get("failure_reason")
        or payload.get("refusal_reason")
        or "validation did not pass"
    )
