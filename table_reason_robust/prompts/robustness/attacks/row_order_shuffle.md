Goal: randomly reorder detail data rows to test whether the model incorrectly depends on physical row positions.

- Apply only when the question does not depend on the original display order. Reject if the task mentions the first row, adjacent records, or ordering relationships without an explicit sorting key.
- Move only complete detail rows. Do not move headers, titles, footnotes, blank separators, or independent aggregate regions.
- Use a fixed random seed. Preserve all field bindings within each row. The multiset of data rows before and after the attack must be exactly identical.
- For Excel, do not rewrite the workbook with pandas `to_excel`. Use openpyxl to move complete cells, preserving original value types, formulas, styles, number formats, comments, hyperlinks, freeze panes, and filter ranges.
- Validation must compare row multisets by "value + Python type + cell style". If textual encoding `"1"` becomes numeric `1`, the attack fails.
- Require at least 5 detail rows, and at least 80% of data rows must change position.
- Reject Excel files with cross-row formulas, merged data cells, or physical-row-number dependencies that cannot be handled safely.
