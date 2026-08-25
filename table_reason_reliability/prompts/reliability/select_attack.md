You need to determine which unanswerable attacks are suitable for the original table-reasoning sample.

Currently enabled attack types:
{{enabled_attacks_json}}

Minimum confidence for each type:
{{min_confidence_by_attack_json}}

This stage is only for candidate discovery. Do not perform final validation, and do not call tools. The goal is to improve recall while keeping attack types correct: if the profile contains enough evidence to support a concrete and reasonable construction plan, mark it as eligible. Reject only when a hard condition is clearly violated. Do not reject merely because numeric computation has not been completed, all alternative paths have not been exhausted, or the profile is incomplete; those checks are left to construct and validator.

## Decision Principles

- First identify which fields, records, files, or structures the answer depends on, then match an attack.
- An eligible item must specify a concrete target and modification method, but reasonable estimates of numeric impact are allowed.
- Exclude only plans with an obvious reliable recovery path. You do not need to prove that all possible paths are absent.
- Do not select attacks merely to fill the quota if they do not match the format or definition.
- For Easy attacks, only judge whether the attack is effective. Do not impose Hard complexity requirements.

## Core Conditions By Type

1. field_missing: A necessary field is removed, or the question is naturally changed to depend on a related field that does not exist; modify either the question or the file, but not both. Not applicable if an obvious equivalent field exists.
2. data_missing: Keep the question and structure unchanged, and set only answer-dependent key data to NULL. A legitimate zero, empty result, or no matching record is not missing data.
3. evidence_conflict: Create conflicting values that would change the answer for the same object, time, metric, and business event. Normal multiple transactions or repeated measurements are not conflicts.
4. file_missing: Select this only when the profile clearly shows at least two input files. After removing a file that covers necessary information, at least one file must remain. The question, filenames, or package structure must show that the removed scope is necessary, and the remaining files cannot recover the information. Single-file samples must reject this type.
5. header_conflict: Two semantically different but plausible candidate columns are changed to the same header, and choosing different columns would change the answer. Not applicable if units, formats, parent headers, or other evidence clearly disambiguate the target.
6. deep_analysis_missing: The question contains at least three dependent operations; earlier steps remain executable; missing information blocks the final conclusion only in a middle or later step. If the failure occurs during first retrieval, use data_missing instead.
7. structural_context_missing: Only for Excel files with multiple Sheets, multi-level headers, repeated subfields, or multiple data regions. The question depends on high-level structure for localization. After that structure is removed, values remain but have multiple reasonable owners. Ordinary column removal should be treated as field_missing.

## Original Sample

ID: {{sample_id}}

Question: {{question}}

Reference answer: {{reference}}

File profile:
{{file_profile_json}}

## Output Format

Output only one JSON object:

{
  "sample_id": "{{sample_id}}",
  "eligible_attacks": [
    {
      "attack_type": "attack type",
      "confidence": 0.0,
      "reason": "why this sample has a reasonable construction path",
      "required_edit": "question_only | modify_file | file_list_only",
      "target": {
        "file": "file name or null",
        "sheet": "Sheet name or null",
        "field": "field name or null",
        "candidate_fields": ["candidate fields"],
        "condition": "object, time, category, primary key, or filter condition",
        "fact_key": ["conflicting fact key"],
        "reasoning_stage": "first retrieval, earlier, middle, later, or null",
        "reasoning_chain": ["step1", "step2", "step3"],
        "structure_dependency": "structure dependency or null"
      }
    }
  ],
  "rejected_attacks": [
    {
      "attack_type": "attack type",
      "reason": "core condition that is clearly not satisfied"
    }
  ]
}

Requirements:

- Each attack type may appear at most once and must not appear in both eligible and rejected.
- Use null or empty arrays for target fields that do not apply. Do not fabricate evidence.
- `confidence` should reflect how worthwhile it is to enter the construction stage. When there is a concrete feasible target and no obvious counterexample, it may reach the configured minimum threshold for that type.
- If profile evidence is limited but a reasonable plan exists, include it as eligible with medium confidence and explain what must be checked later.
- The final output must be a JSON object only.
