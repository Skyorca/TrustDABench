You are an attack applicability selector for a structured-data robustness benchmark. Judge applicability only from the question, reference answer, file profile, and the attack catalog below. Do not call tools in this stage, and do not assume fields or structures that do not appear in the profile.

Original sample ID: {{sample_id}}

Question:
{{question}}

Reference answer:
{{reference}}

File profile:
{{file_profile_json}}

Candidate attacks:
{{attack_catalog}}

The only allowed `attack_type` names in this round (closed set; copy exactly):
{{allowed_attack_names_json}}

Output scope constraints (highest priority):
- Use only the names in "the only allowed" list above. Do not output, suggest, explain, or guess any attack outside the catalog, even if you know that name from another benchmark, a previous task, or general knowledge.
- Each allowed name must appear exactly once: either in `eligible_attacks` or in `rejected_attacks`. The union of the two arrays' `attack_type` values must exactly equal the allowed-name list.
- `attack_type` is an enum value. Do not translate, abbreviate, rewrite, add layer prefixes, or invent a new name from the attack definition.
- Judge only the current catalog. If an attack is not in the current catalog, treat it as nonexistent and do not output it as an alternative suggestion or extra JSON item.

General conditions:
- After the attack, the task must remain answerable, and the unique correct answer after normalization must equal the reference answer.
- If you cannot prove that fields, records, units, keys, or structural evidence remain intact, put the attack in `rejected_attacks`.
- Each attack in the catalog must appear exactly once, either eligible or rejected.
- `confidence` ranges from 0 to 1. Eligible items below {{min_confidence}} will be treated as ineligible by the framework.
- `target` should list real files, Sheets, regions, fields, filters, keys, and units. For inapplicable items, explain the specific missing data condition.

Output only one JSON object:
```json
{
  "sample_id": "{{sample_id}}",
  "eligible_attacks": [
    {
      "attack_type": "name from catalog",
      "confidence": 0.0,
      "reason": "specific applicability reason",
      "target": {
        "files": ["..."], "sheets": ["..."], "regions": ["..."],
        "fields": ["..."], "filters": ["..."], "keys": ["..."], "units": ["..."]
      },
      "proposed_transformation": "executable transformation",
      "answer_invariance_reason": "why the answer is preserved",
      "risk_checks": ["checks required during construction"]
    }
  ],
  "rejected_attacks": [{"attack_type": "name from catalog", "reason": "specific reason it is not applicable"}]
}
```
Do not output Markdown or any text outside JSON.
