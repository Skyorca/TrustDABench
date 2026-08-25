Goal: replace answer-dependent column names with strictly equivalent synonyms in the current business context.

- Modify only headers. Do not modify the question or data.
- Old and new names must have the same concept, metric scope, and granularity. Hypernyms, hyponyms, or near-synonyms with different scope are forbidden.
- The new name must not duplicate an existing column and must not create two equally plausible mappings.
- Replace at least one key column used for filtering, grouping, computation, or comparison.
- The replaced column must be bound by the business concept, computational role, unit, entity, or structural context in the question. Do not choose a column merely because a value happens to appear in it.
- After the attack, there must be a business-plausible but insufficient field interpretation path. This path must not rely only on fixed column position, a single value distribution, or exact string matching, and it must select the wrong field, fail the task, or produce a different final output. At the same time, the correct field must still be uniquely recoverable from the question and table evidence.
- By default, an atomic attack replaces only one key field. Replace 2-3 fields only when the question depends on multiple fields and every pair independently passes strict synonym auditing. Never replace more than 3.
- Run a "coexistence test" for every name pair: if the two names could naturally coexist as distinct columns in the same table, such as "supplier" and "service provider", they are not strict synonyms and must be rejected.
- Do not delete or add constraints such as time basis, statistical scope, entity scope, unit, tax inclusion, planned/actual status, or cumulative/current-period status.
- The final mapping must contain only actual old_header to new_header changes. If the actual target is not in selection.target.fields, fill the corresponding field_bindings item with a specific selection_deviation_reason.
- transformation_record.semantic_contract must include:
  - field_bindings: for each modified field, include question_concept, target_field, transformed_field, question_evidence, and selection_deviation_reason;
  - synonym_audit: for each old_header/new_header pair, include same_concept, same_metric_scope, same_granularity, same_time_basis, same_unit, can_coexist_as_distinct, and evidence.
  - interpretation_risk: correct_interpretation, plausible_incorrect_interpretation, why_plausible, incorrect_outcome, outcome_differs, recoverability_evidence, output_difference_evidence. The wrong interpretation should be a plausible semantic misreading in the business context, not a reference to a specific model, library, or fixed heuristic.
- The first six same_* fields in synonym_audit must be true, and can_coexist_as_distinct must be false. Reject if a high-confidence natural synonym cannot be provided.
