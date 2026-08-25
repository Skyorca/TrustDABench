Goal: change key values to semantically equivalent representations that require explicit normalization.

- Select exactly one subtype for each sample: numbers as strings with thousands separators, decimals and percentage strings, unambiguous dates, or booleans and explicit yes/no representations.
- Change only the representation. Do not change mathematical values, dates, truth values, or missingness.
- Transform at least 30% of non-null relevant cells in answer-dependent fields.
- Ambiguous dates or numbers with unclear locale settings are forbidden. Do not confuse NULL, zero, and string 0.
- Perform the inverse conversion and prove per-value equality.
- Transformations that only change visual formatting, while ordinary reading can complete the original task without any explicit parsing or normalization, are not applicable.
- The attacked raw representation must require an explicit and unambiguous normalization, type recovery, or semantic parsing step before the filtering, joining, comparison, sorting, grouping, or computation required by the original question can be completed.
- Provide one business-plausible interpretation path that omits this step, and prove that it leads to a different output, an incorrect data operation, or an unanswerable task. Do not assume any specific model, programming language, library, or locale setting.
- transformation_record.semantic_contract must include interpretation_risk: correct_interpretation, plausible_incorrect_interpretation, why_plausible, incorrect_outcome, outcome_differs, recoverability_evidence, output_difference_evidence.
