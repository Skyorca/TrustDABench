Goal: randomly reorder data columns to test whether the model depends on fixed column positions.

- Only handle rectangular regions with a clear single-level header. Complex multi-level headers are not applicable.
- Move headers, data, and column formats together as complete columns, preserving row-wise record relationships.
- For Excel, first use `shutil.copy2` to copy each original workbook from `/mnt/data` to `/mnt/output`; then open only that copy with openpyxl and move complete cells in place. Do not create a new workbook, rebuild the worksheet with pandas/`to_excel`/`ExcelWriter`, or read into a DataFrame and write it back.
- When moving each column, copy each cell's `value`, `data_type`, `number_format`, complete `_style`, comment, and hyperlink as the semantic cell object. After moving, reopen the output file and compare each mapped column by original header using "value + Python type + number format".
- Validation must compare complete column sequences by header using "value + Python type + cell style". If textual encoding `"1"` becomes numeric `1`, the attack fails.
- Use a fixed random seed and record the original and new column orders.
- At least one answer-dependent column must be moved, and at least 60% of columns must change position.
- Reject if positional references, merged headers, or cross-column formulas cannot be handled safely.
