from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Operator:
    name: str
    priority: int
    expected_behavior: str
    construct_prompt: str
    instruction_prompt: str
    definition: str
    supported_extensions: Tuple[str, ...]
    dimension: str = "robustness"
    validator: str = "validate_robustness"


COMMON_CONSTRUCT_PROMPT = "prompts/robustness/construct_attack.md"
TABLE_EXTENSIONS = (".xlsx", ".csv", ".tsv")
CSV_EXTENSIONS = (".csv", ".tsv")
EXCEL_EXTENSIONS = (".xlsx",)


def _operator(
    name: str,
    priority: int,
    definition: str,
    supported_extensions: Tuple[str, ...] = TABLE_EXTENSIONS,
) -> Operator:
    return Operator(
        name=name,
        priority=priority,
        expected_behavior="answer",
        construct_prompt=COMMON_CONSTRUCT_PROMPT,
        instruction_prompt=f"prompts/robustness/attacks/{name}.md",
        definition=definition,
        supported_extensions=supported_extensions,
    )


ROBUSTNESS_OPERATORS: Dict[str, Operator] = {
    "row_order_shuffle": _operator(
        "row_order_shuffle", 1,
        "Reorder detail rows while preserving the typed row multiset, field bindings, and normalized answer.",
    ),
    "column_order_shuffle": _operator(
        "column_order_shuffle", 2,
        "Reorder flat-table columns while preserving each named column, record relation, and answer.",
    ),
    "header_synonym_substitution": _operator(
        "header_synonym_substitution", 3,
        "Replace answer-relevant headers with strictly equivalent, uniquely resolvable business synonyms.",
    ),
    "semantic_distractor_column": _operator(
        "semantic_distractor_column", 4,
        "Add one nearby, semantically related but excludable distractor column whose misuse changes the answer.",
    ),
    "equivalent_value_reencoding": _operator(
        "equivalent_value_reencoding", 5,
        "Reencode key values in a reversible, unambiguous representation to test value normalization.",
    ),
    "unit_scale_conversion": _operator(
        "unit_scale_conversion", 6,
        "Apply an exact unit or magnitude conversion to a key numeric field and update the unit label consistently.",
    ),
    "csv_wide_long_reshape": _operator(
        "csv_wide_long_reshape", 7,
        "Perform a reversible, non-aggregating wide/long transformation for CSV or TSV data.", CSV_EXTENSIONS,
    ),
    "csv_relational_decomposition": _operator(
        "csv_relational_decomposition", 8,
        "Decompose one CSV into losslessly joinable relations using safe primary/foreign keys.", CSV_EXTENSIONS,
    ),
    "excel_hierarchical_header_relayout": _operator(
        "excel_hierarchical_header_relayout", 9,
        "Relayout a flat Excel header as a semantically complete multi-level or merged header.", EXCEL_EXTENSIONS,
    ),
    "excel_cross_sheet_relayout": _operator(
        "excel_cross_sheet_relayout", 10,
        "Split or merge Excel sheets along a natural dimension while preserving record evidence and field location.", EXCEL_EXTENSIONS,
    ),
    "decoy_feature_pack_injection": _operator(
        "decoy_feature_pack_injection", 11,
        "Append two to five related, type-compatible decoy features. The original evidence remains intact, but using the pack changes the result.",
    ),
    "non_observation_row_injection": _operator(
        "non_observation_row_injection", 12,
        "Append naturally marked summary, sample, simulated, check, or control records that are not observations; treating them as observations changes the result.",
    ),
}


def get_enabled_operators(names: List[str]) -> List[Operator]:
    operators: List[Operator] = []
    for name in names:
        if name not in ROBUSTNESS_OPERATORS:
            raise KeyError(f"Unknown robustness operator: {name}")
        operators.append(ROBUSTNESS_OPERATORS[name])
    return sorted(operators, key=lambda op: op.priority)
