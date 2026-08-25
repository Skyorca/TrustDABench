Goal: reorganize Excel flat headers and multi-level or merged headers in a semantically equivalent way.

- Select only fields with a natural parent-child relationship, such as `2024_sales` with parent `2024` and child `sales`.
- All parent labels, child labels, units, and time labels must be preserved, and the original fields must be uniquely recoverable after composition.
- Correctly update header row counts, data start rows, and merged ranges. Do not overwrite data or break formula references.
- The new structure should match realistic Excel usage. Do not add meaningless hierarchy levels.
- Parse the new headers and compare them one by one with the original fields. Reject when complex dependencies cannot be moved safely.
- transformation_record.parameters must provide a sheet_specs array. Each item must contain file, sheet, original_header_rows, final_header_rows, original_data_start_row, and final_data_start_row. The framework will compare all data cells from the declared data start row, including text, NULLs, formulas, and number formats.
- Only header rows may be moved. Data column order must remain unchanged. If columns must be reordered to form the hierarchy, reject and use column_order_shuffle instead.
