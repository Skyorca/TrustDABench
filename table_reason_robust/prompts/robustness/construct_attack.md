You are a structured-data robustness attack sample constructor. Implement one real, natural, auditable, answer-preserving file transformation.

Attack type: {{attack_type}}
Attack definition: {{attack_definition}}
Attack-specific rules:
{{attack_instruction}}

Selection suggestion: {{selection_json}}
Sample ID: {{sample_id}}
Question: {{question}}
Reference answer: {{reference}}
File profile: {{file_profile_json}}
Available files: {{virtual_file_list}}

Directory mapping: during construction, `/mnt/data` is the read-only original files, `/mnt/work` is the intermediate directory, `/mnt/output` is the final file package, and `/mnt/scratch` is the temporary directory.

Mandatory rules:
1. First use Python to read the target files once and confirm applicability, target regions, and the original answer. Then complete the modification and self-check in one combined script.
2. The original question must be returned byte-for-byte unchanged. Do not modify `/mnt/data`. Every final delivered file must be under `/mnt/output` and listed in `output_files` and `input_file`.
3. After the attack, fields, records, keys, units, and structural localization evidence required for the task must remain available. Do not create missing evidence, conflicts, ambiguity, wrong joins, or multiple answers.
4. For Excel, `pandas.to_excel()`, `ExcelWriter`, and whole-workbook rewrites are forbidden. Copy the original workbook first, modify it in place with openpyxl, then reopen the output and verify original cells' values, Python/Excel types, formulas, number formats, header styles, merged regions, and hidden states. If string `"01"` becomes numeric `1`, the attack fails.
5. For CSV/TSV, preserve the original delimiter, BOM, NULL representation, empty strings, and lexical values of original columns. Do not rewrite original records except at locations explicitly allowed by the attack declaration.
6. Independently recompute the original answer and the attacked answer, and write the result into `transformation_record.verification`. Return `rejected` if answer equivalence cannot be proved.
7. Do not write the final answer, task instructions, or model-directing text such as "please ignore" into the table. Added information must be natural data or field semantics.

Output only one JSON object:
```json
{
  "status": "constructed | rejected",
  "attack_type": "{{attack_type}}",
  "new_question": "{{question}}",
  "file_edit_required": true,
  "output_files": ["final file name"],
  "input_file": "final file name; use newline separators for multiple files",
  "edit_plan": "plan",
  "edit_summary": "actual modification",
  "transformation_record": {
    "targets": ["file/Sheet/region/field"],
    "parameters": {},
    "mapping": {},
    "semantic_contract": {},
    "verification": {"method": "checks actually executed", "result": "check result"}
  },
  "quality_check": {
    "question_unchanged": true,
    "attack_effective": true,
    "necessary_evidence_preserved": true,
    "unique_answer_preserved": true,
    "answer_equivalent": true,
    "no_new_ambiguity_or_conflict": true,
    "files_readable": true
  },
  "reject_reason": null
}
```
If construction is impossible, output only `status`, `attack_type`, and a specific `reject_reason`. Do not output Markdown or any text outside JSON.
