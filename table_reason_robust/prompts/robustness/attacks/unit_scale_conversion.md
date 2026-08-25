Goal: apply an explicit, reversible unit or magnitude conversion to key numeric columns.

- Use only fields whose original unit is explicit and has an exact conversion relationship, such as kg/g, yuan/ten-thousand yuan, or seconds/minutes.
- Convert all non-null values in the full column uniformly, and update the header, unit row, or adjacent note accordingly.
- Keep the question unchanged. If the question specifies an output unit, the attacked data must be convertible back to that unit.
- Prefer a fixed multiplicative factor. Currency exchange rates, vague scopes, and fields with missing units are not applicable.
- Avoid rounding that changes ordering or thresholds. Record the conversion formula, factor, and precision, and verify by inverse conversion.
- The unit must be part of the computation semantics of the original task: if attacked values are interpreted using the pre-attack unit or the conversion is ignored, the final output must differ observably in the required precision, ordering, set membership, or answerability.
- If all answer-dependent operands are scaled by the same factor and the unit completely cancels out in the original operation, the sample is not applicable. Do not claim an attack is effective merely because the file was converted.
- Provide one business-plausible interpretation path that omits conversion and its output consequence. Do not assume any specific model, programming language, or library.
- transformation_record.semantic_contract must include interpretation_risk: correct_interpretation, plausible_incorrect_interpretation, why_plausible, incorrect_outcome, outcome_differs, recoverability_evidence, output_difference_evidence.
