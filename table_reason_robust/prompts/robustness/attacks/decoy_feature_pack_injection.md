Goal: append 2--5 semantically related, type-compatible, high-coverage redundant features to the right side of an existing data region, testing whether the model keeps using the original fields and computation path specified by the question. Each attack may modify only one file; for Excel, it may modify only one Sheet.

Applicability conditions:
- The question depends on at least one localizable original field, and that field has enough data in the target CSV/TSV or Excel Sheet.
- It is possible to construct more than two business-natural derived, predicted, historical, normalized, rounded, ranked, or alternative-scope features.
- The original field remains uniquely identifiable from the question text and table structure. If a new feature is used as the original field or intermediate evidence, the final result must truly differ.

Construction rules:
- Append columns only to the far right of the target data region. Do not modify, reorder, rename, overwrite, or delete any existing cell. Row count must remain unchanged.
- Each new column must have at least 60% coverage among non-null original data rows, and its value type must be compatible with the declared `source_field`. Do not append constant columns, answer columns, or model-facing instruction text.
- For Excel, copy the existing header style. For CSV/TSV, preserve the original delimiter, BOM, lexical values of original columns, and NULL representation.
- Do not exclude a new feature merely because it is on the right, has a longer name, or the constructor declares it irrelevant. `exclusion_reason` must come from verifiable facts in the question text and table, such as field semantics, time basis, actual/predicted status, or original/derived status.

`transformation_record.semantic_contract` must be:
```json
{
  "feature_pack": {
    "file": "target file name",
    "sheet": "target Excel Sheet; omit for CSV/TSV",
    "header_row": 1,
    "target_fields": ["original fields truly required by the question"],
    "added_features": [
      {
        "name": "new column name",
        "source_field": "original field used for generation or approximation",
        "noise_subtype": "derived_feature_pack | historical_feature_pack | normalized_feature_pack | forecast_feature_pack | rounded_feature_pack | ranking_feature_pack",
        "exclusion_reason": "why the question and table evidence require not using this column"
      }
    ],
    "misuse_answer": "non-equivalent result when at least one new column is misused"
  }
}
```

First use one combined Python script to read, construct, and recompute both the correct answer and the misuse answer. Return `rejected` if you cannot prove both "the original answer is preserved" and "the misuse result differs".
