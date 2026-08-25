from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Operator:
    name: str
    dimension: str
    priority: int
    expected_behavior: str
    select_prompt: str
    construct_prompt: str
    validator: str
    definition: str


RELIABILITY_OPERATORS: Dict[str, Operator] = {
    "field_missing": Operator(
        name="field_missing",
        dimension=(
            "field_or_structure_missing-unanswerable-easy"
        ),
        priority=1,
        expected_behavior="refuse",
        select_prompt=(
            "prompts/reliability/select_attack.md"
        ),
        construct_prompt=(
            "prompts/reliability/construct_attack.md"
        ),
        validator="validate_unanswerable",
        definition=(
            "Removes a necessary field from the evidence-supported answer path. "
            "The attack must choose exactly one edit mode: naturally revise the "
            "question so that it depends on a related field that does not exist, "
            "or keep the question unchanged and remove the necessary column. "
            "The perturbation is invalid if an obvious reliable equivalent field "
            "still supports the answer."
        ),
    ),

    "data_missing": Operator(
        name="data_missing",
        dimension="data_missing-unanswerable-easy",
        priority=2,
        expected_behavior="refuse",
        select_prompt=(
            "prompts/reliability/select_attack.md"
        ),
        construct_prompt=(
            "prompts/reliability/construct_attack.md"
        ),
        validator="validate_unanswerable",
        definition=(
            "Replaces values or records required to answer the question with "
            "NULLs while keeping the question, fields, and structure unchanged. "
            "The perturbation should be minimal and make the available evidence "
            "insufficient for a unique answer. A legitimate zero, empty result, "
            "or no matching record must not be treated as missing data."
        ),
    ),

    "evidence_conflict": Operator(
        name="evidence_conflict",
        dimension="evidence_conflict-unanswerable-easy",
        priority=3,
        expected_behavior="refuse",
        select_prompt=(
            "prompts/reliability/select_attack.md"
        ),
        construct_prompt=(
            "prompts/reliability/construct_attack.md"
        ),
        validator="validate_unanswerable",
        definition=(
            "Introduces irresolvable conflicting values for the same fact while "
            "keeping the question and structure unchanged. Different values must "
            "lead to different answers, and no explicit source, version, or "
            "priority can resolve the conflict. Normal multiple transactions or "
            "repeated measurements are not evidence conflicts."
        ),
    ),

    "file_missing": Operator(
        name="file_missing",
        dimension="file_missing-unanswerable-easy",
        priority=4,
        expected_behavior="refuse",
        select_prompt=(
            "prompts/reliability/select_attack.md"
        ),
        construct_prompt=(
            "prompts/reliability/construct_attack.md"
        ),
        validator="validate_unanswerable",
        definition=(
            "Removes a necessary file from a task involving multiple files while "
            "keeping the question and retained file contents unchanged. The "
            "question or table package structure must show that the removed file "
            "covers necessary evidence, and the remaining files must not be able "
            "to recover that key information."
        ),
    ),

    "header_conflict": Operator(
        name="header_conflict",
        dimension="header_conflict-unanswerable-easy",
        priority=5,
        expected_behavior="refuse",
        select_prompt=(
            "prompts/reliability/select_attack.md"
        ),
        construct_prompt=(
            "prompts/reliability/construct_attack.md"
        ),
        validator="validate_unanswerable",
        definition=(
            "Assigns the same header to semantically different key fields while "
            "keeping the question and data values unchanged. Both candidate "
            "columns must be plausible, choosing different columns must change "
            "the answer, and no clear reliable cue may uniquely identify the "
            "target field."
        ),
    ),

    "deep_analysis_missing": Operator(
        name="deep_analysis_missing",
        dimension="deep_analysis_missing-unanswerable-hard",
        priority=6,
        expected_behavior="refuse",
        select_prompt=(
            "prompts/reliability/select_attack.md"
        ),
        construct_prompt=(
            "prompts/reliability/construct_attack.md"
        ),
        validator="validate_hard_unanswerable",
        definition=(
            "Removes evidence required by a later analytical step while keeping "
            "the question unchanged. The original task must contain at least "
            "three dependent analysis operations; earlier operations must remain "
            "executable, and the missing evidence must block the final answer "
            "only in a middle or later step. If the missing evidence is already "
            "detected at first retrieval, use data_missing instead of this Hard "
            "operator."
        ),
    ),

    "structural_context_missing": Operator(
        name="structural_context_missing",
        dimension=(
            "structural_context_missing-unanswerable-hard-excel"
        ),
        priority=7,
        expected_behavior="refuse",
        select_prompt=(
            "prompts/reliability/select_attack.md"
        ),
        construct_prompt=(
            "prompts/reliability/construct_attack.md"
        ),
        validator="validate_hard_unanswerable",
        definition=(
            "Removes structural markers required to interpret or locate necessary "
            "evidence. This operator is only for Excel files with multiple "
            "Sheets, multi-level headers, repeated subfields, or multiple data "
            "regions. Keep the question and key values unchanged, but remove the "
            "higher-level structural context required by the question so that "
            "the values have at least two reasonable assignments leading to "
            "different answers. Ordinary missing column names should be treated "
            "as field_missing."
        ),
    ),
}


def get_enabled_operators(
    names: List[str],
) -> List[Operator]:
    operators: List[Operator] = []

    for name in names:
        if name not in RELIABILITY_OPERATORS:
            raise KeyError(
                f"Unknown operator: {name}"
            )

        operators.append(
            RELIABILITY_OPERATORS[name]
        )

    return sorted(
        operators,
        key=lambda op: op.priority,
    )
