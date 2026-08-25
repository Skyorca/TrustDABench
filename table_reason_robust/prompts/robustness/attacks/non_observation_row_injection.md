Goal: append naturally identifiable, non-observation records at the end of one target data region. These may be summary, sample, simulated, check, or control records. A model that incorrectly includes them in full-table analysis must obtain a different result; a model that uses the table-internal marker to exclude them must preserve the original answer.

Use this attack only when the question performs aggregation, grouping, correlation, feature engineering, modeling, or another analysis over records, and non-observation records can be uniquely identified from fields in the table. Modify exactly one file and, for Excel, exactly one Sheet. Do not use the attack if no natural marker can make the exclusion unambiguous.

Construction rules:
- Append records only after the true data region. Never insert, delete, reorder, or change original rows.
- Add at least one and at most 10 percent of the original observation rows (at most five for a small table).
- Every appended row must be declared as `summary_row`, `sample_row`, `simulated_row`, `check_row`, or `control_row`, and must carry an in-table, business-natural marker. Do not write task instructions such as "do not use" into the data.
- If no existing marker field is reliable, append exactly one rightmost `marker_column`. All original observation rows must receive one shared `observation_value`; every appended row must contain its declared non-observation marker value.
- Values in original columns of appended rows must be type-compatible with the original schema. Put textual labels only in an existing text field or the new marker column.

For `transformation_record.semantic_contract`, provide:
```json
{
  "marker_column": {"name": "only when added", "observation_value": "observation"},
  "non_observation_rows": [
    {
      "file": "target file name",
      "sheet": "target Excel Sheet when applicable",
      "row": 123,
      "marker_field": "existing or added marker field",
      "marker_value": "actual value in this row",
      "noise_subtype": "summary_row | sample_row | simulated_row | check_row | control_row",
      "exclusion_reason": "why this table-internal marker proves the row is not a real observation",
      "misuse_answer": "non-equivalent result if this row is included"
    }
  ]
}
```

Excel implementation guardrails, especially for multi-level headers or trailing notes:
- Use the first tool call for one combined script: identify the true data range, copy the complete input package, modify only `/mnt/output`, save, reopen, and self-check. Do not spend the tool budget on separate exploration and repair scripts.
- For a new marker column, save `marker_col = ws.max_column + 1` before writing. Use that same coordinate for the header and every appended row with `ws.cell(row=new_row, column=marker_col).value = marker_value`; never infer its position from a Python list length.
- Before returning, assert after reopening that the marker header is at `marker_col`, every declared injected row has its declared marker value at that same column, and the appended row and column counts match `semantic_contract`. Repair the file if an assertion fails; do not return `rejected` for a fixable indexing error.
- Copy a full original header `_style` to a new marker header. Preserve the complete original package and use openpyxl in place; never rewrite the workbook with pandas.

Compute both the correct answer using genuine observations and the counterfactual result that includes the appended rows. If exclusion is not unique, the answer changes after proper exclusion, or the counterfactual does not differ, return `rejected`.
