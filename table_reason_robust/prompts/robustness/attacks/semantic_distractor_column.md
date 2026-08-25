Goal: add a distractor column that is semantically close but has a different scope and should clearly not be used for the original question.

- Each sample attacks only one target concept. By default, add only one distractor column. For multi-file tasks, you may add one same-name, same-scope distractor column to each necessary file.
- The distractor should resemble pairs such as "net profit/operating profit" or "actual transport volume/planned transport volume", while the question and original table evidence must still uniquely identify the correct column.
- Before construction, perform a field-binding audit: quote the wording in the question that localizes the target concept, enumerate the original field and distractor field, and provide unique exclusion evidence that does not depend on the reference answer.
- If the question uses only generic terms such as "date", "amount", "quantity", or "cost", do not add another date or business-scope field that is equally plausible. The only exclusion evidence cannot be "the original field has no qualifier", "the original field appears first", or an industry convention.
- Insert the new column immediately next to the target column. Do not append it far away at the end of the table.
- Prefer deterministic derivation from existing related fields to create natural values. Reject if business-plausible values cannot be constructed.
- The distractor column and target column must have compatible type families. Non-null coverage must be at least 50% of the target column, and at least 30% of answer-related records must have distractor values different from target values.
- The distractor should also be close to the target in at least two observable dimensions such as unit, entity granularity, time range, business role, or position. It must not be made easy to exclude by only one obvious qualifier.
- You must compute a counterfactual_answer under one business-plausible misuse path, and ensure it differs observably from the correct answer in the required output precision, ordering, set membership, or answerability.
- Do not modify the target column or correct evidence. Do not make the distractor a valid substitute.
- transformation_record.semantic_contract must include target_concept, target_field, distractor_field, question_binding_quote, exclusion_evidence, selection_deviation_reason, counterfactual_method, counterfactual_answer, and interpretation_risk (correct_interpretation, plausible_incorrect_interpretation, why_plausible, incorrect_outcome, outcome_differs, recoverability_evidence, output_difference_evidence).
- exclusion_evidence must come from the question text and table structure. Reject if the distractor field cannot be uniquely excluded.
