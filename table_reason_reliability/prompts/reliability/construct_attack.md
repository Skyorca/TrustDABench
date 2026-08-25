You need to construct a high-quality unanswerable table-reasoning sample.

Phase: {{phase_name}}

Attack type: {{attack_type}}

Definition: {{attack_definition}}

Candidate plan: {{selection_json}}

The candidate plan is only guidance. You must inspect the real files with Python. If the target is not valid, you may switch to a better target of the same attack type. Return `rejected` only when no plan satisfies the core definition. Do not reject merely because non-critical proof fields are incomplete.

## Original Sample

ID: {{sample_id}}

Question: {{question}}

Reference answer: {{reference}}

File profile: {{file_profile_json}}

Available files:
{{virtual_file_list}}

Path mapping: original files are under `/mnt/data`; working files are under `/mnt/work`; final files must be written under `/mnt/output`; temporary files are under `/mnt/scratch`.

## Objective

1. Confirm that the original question is answerable and locate the key evidence required by the answer.
2. Apply the minimal sufficient perturbation for the current attack type.
3. Confirm that the perturbation makes the unique answer impossible, with no obvious reliable recovery path.
4. Keep the question natural, files readable, and unrelated content unchanged.

"No recovery path" means the current inputs contain no clear and reliable alternative source for the answer. You do not need to exhaust strained possibilities, external knowledge, or unusual guesses. Quality comes from a clear causal link in the attack, not from the amount of proof text.

## Tool Budget

- Use at most 3 Python tool calls. Each code snippet must be self-contained.
- Prefer targeted reads of the relevant file, Sheet, field, and necessary rows. Avoid repeated full-table scans.
- Do not use unbounded loops, recursive traversal, threads, multiprocessing, or very large computations.
- Once there is sufficient evidence, construct and output immediately. Return `rejected` only when key facts cannot be verified.

## Question And File Constraints

- data_missing, evidence_conflict, file_missing, header_conflict, deep_analysis_missing, structural_context_missing: the question must remain byte-for-byte unchanged.
- field_missing/question_only: modify only the question; files must remain unchanged. The edit should be local, natural, and must not change the task type.
- field_missing/modify_file: delete only the necessary field; the question must remain unchanged.
- file_missing: only change the final provided file list, and keep file contents unchanged. `output_files` and `input_file` must list only the files retained in the final package. Do not continue to reference removed files, and do not delete source files under `/mnt/data`.

## Type-Specific Quality Conditions

- field_missing: the missing field is required by the answer, and there is no obvious equivalent field.
- data_missing: only necessary data is set to NULL; fields and structure remain unchanged. The missingness causes insufficient information, rather than a legitimate answer of zero or an empty result.
- evidence_conflict: the conflict concerns the same fact; the two values lead to different answers; the conflict is not a normal repeated record.
- file_missing: the original sample has at least two files; the removed file covers necessary information; at least one file remains; the remaining files cannot recover the key information.
- header_conflict: only headers are changed; both candidate columns are plausible and lead to different answers; no clear reliable disambiguating evidence exists.
- deep_analysis_missing: the task contains at least three dependent operations; earlier steps remain executable; missing information blocks the final answer only in a middle or later step.
- structural_context_missing: only for complex Excel files; key values are preserved, but after high-level structure is removed, at least two reasonable assignments exist and they affect the answer.

Numeric counts, two candidate answers, and Hard checks should be filled only when the corresponding attack truly needs them. For other types, use null, 0, false, or empty arrays. Do not perform irrelevant calculations just to fill fields.

## Output Format

When construction succeeds, output only:

{
  "status": "constructed",
  "attack_type": "{{attack_type}}",
  "new_question": "perturbed question",
  "expected_answer": {
    "type": "refusal | clarification",
    "reason": "specific missing or conflicting evidence and its impact"
  },
  "file_edit_required": true,
  "output_files": ["final retained file names"],
  "input_file": "use newline separators for multiple files",
  "edit_plan": "concise edit plan",
  "edit_summary": "actual edits",
  "base_attack_components": [],
  "reasoning_chain": [],
  "attack_evidence": {
    "target_file": null,
    "target_sheet": null,
    "target_fields": [],
    "target_condition": null,
    "fact_key": [],
    "original_value": null,
    "perturbed_value": null,
    "original_valid_count": 0,
    "modified_count": 0,
    "remaining_valid_count": 0,
    "answer_under_option_a": null,
    "answer_under_option_b": null,
    "affected_step": null,
    "alternative_paths_checked": []
  },
  "hardness_check": {
    "has_at_least_three_dependent_steps": false,
    "earlier_steps_remain_executable": false,
    "failure_occurs_in_middle_or_late_step": false,
    "primary_failure_is_structural_context": false,
    "key_values_preserved": false,
    "no_alternative_answer_path": true,
    "does_not_degenerate_to_easy": true
  },
  "quality_check": {
    "original_was_answerable": true,
    "question_rule_valid": true,
    "file_edit_scope_valid": true,
    "is_natural": true,
    "is_unanswerable": true,
    "reason_is_specific": true
  },
  "reject_reason": null
}

When the core definition cannot be satisfied, output only:

{
  "status": "rejected",
  "attack_type": "{{attack_type}}",
  "reject_reason": "specific core-condition failure"
}

The final output must be a JSON object only.
