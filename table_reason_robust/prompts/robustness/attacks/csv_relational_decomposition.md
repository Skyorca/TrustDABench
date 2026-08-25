Goal: split one CSV/TSV into two or three files along natural entity boundaries that require joining.

- There must be a stable, non-null, unique or provably safe join key. Do not invent arbitrary row numbers as fake business keys.
- Distribute fields required by the answer across at least two files. No single file should be able to complete the original question independently.
- Each file must retain the join key, and field allocation must follow natural entity boundaries.
- Do not allow many-to-many expansion, duplicate keys, NULL keys, lost records, or the original full table remaining in the final package.
- Rejoin the output files and prove that the original table can be recovered losslessly. Record field allocation and join cardinality.
- "Lossless recovery" means that after the join, all fields of the original CSV can be recovered row by row and string by string in the original column order, including derivable columns. Do not drop fields merely because they can be recomputed or do not affect the current answer.
- transformation_record.parameters must declare join_key or keys and record the field list for each output file. Every original field other than the join key must appear in exactly one output table.
