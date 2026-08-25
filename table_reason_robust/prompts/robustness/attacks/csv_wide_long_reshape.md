Goal: perform a reversible, non-aggregating equivalent transformation between wide and long CSV/TSV tables.

- Identify entity keys, dimensions encoded in column names, and metric values. Field names must be natural and clear.
- The transformation must not rely on aggregation to handle duplicate keys. Reject if key combinations are not unique or information would be lost.
- Preserve all records, NULL positions, and type semantics, and cover at least one question-dependent field.
- Record direction, id_vars, dimension columns, value columns, and before/after shapes.
- Perform the inverse transformation and prove per-value equivalence with the original table after sorting by keys.
- transformation_record.parameters must provide a reshape_specs array, one item for each modified file, containing file, direction, id_vars, value_vars, dimension_column, and value_column. For long-to-wide transformations, value_vars should contain the complete value set of the dimension column.
- id_vars combinations in a wide table, and id_vars + dimension_column combinations in a long table, must be unique. Do not resolve duplicate keys through groupby, first, sum, or deduplication.
- Multi-file tasks must declare and inverse-transform each file separately. One vague parameter set cannot represent different schemas.
