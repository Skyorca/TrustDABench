#!/usr/bin/env python3
"""Build per-operator ASR tables from clean/attacked judgment pairs.

The input manifest has the same schema as batch_compute_robust_scores.py:

dataset,model,clean,attacked

ASR follows compute_robust_scores.py exactly:

    P(attacked incorrect | clean correct)

The script writes a machine-readable long table and human-readable matrix
tables. Matrix cells report ``ASR (clean-correct support)``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any

import compute_robust_scores as core


MANIFEST_COLUMNS = ["dataset", "model", "clean", "attacked"]
LONG_COLUMNS = [
    "dataset",
    "model",
    "level",
    "attack_type",
    "ASR",
    "Support",
    "attack_successes",
    "CRA",
    "paired_records",
    "unmatched_records",
]

ROBUSTNESS_LEVELS = {
    "row_order_shuffle": "L0",
    "column_order_shuffle": "L0",
    "header_synonym_substitution": "L0",
    "semantic_distractor_column": "L0",
    "equivalent_value_reencoding": "L1",
    "unit_scale_conversion": "L1",
    "csv_wide_long_reshape": "L2",
    "csv_relational_decomposition": "L2",
    "excel_hierarchical_header_relayout": "L2",
    "excel_cross_sheet_relayout": "L2",
    "decoy_feature_pack_injection": "L3",
    "non_observation_row_injection": "L3",
}
OPERATOR_ORDER = list(ROBUSTNESS_LEVELS)

MODEL_DISPLAY_NAMES = {
    "qwen3.6-27b": "qwen3.6-27b",
    "qwen3-30b-a3b-thinking-2507": "qwen3-30b-a3b",
}


def display_model_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV containing dataset, model, clean, and attacked columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <manifest_dir>/outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="operator_asr",
        help="Output filename prefix (default: operator_asr).",
    )
    parser.add_argument(
        "--unscored-policy",
        choices=("incorrect", "exclude"),
        default="incorrect",
        help="Match compute_robust_scores.py unscored handling.",
    )
    parser.add_argument(
        "--scale",
        choices=("percent", "fraction"),
        default="percent",
        help="Write ASR/CRA as percentages or fractions.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places in matrix tables (default: 2).",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [name for name in MANIFEST_COLUMNS if name not in fields]
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {', '.join(missing)}"
            )

        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for line_number, raw in enumerate(reader, start=2):
            row = {
                name: str(raw.get(name) or "").strip()
                for name in MANIFEST_COLUMNS
            }
            empty = [name for name, value in row.items() if not value]
            if empty:
                raise ValueError(
                    f"Manifest line {line_number} has empty fields: "
                    f"{', '.join(empty)}"
                )
            key = (row["dataset"], row["model"])
            if key in seen:
                raise ValueError(
                    f"Duplicate dataset/model in manifest line {line_number}: {key}"
                )
            seen.add(key)
            rows.append(row)

    if not rows:
        raise ValueError("Manifest contains no rows")
    return rows


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def operator_sort_key(name: str) -> tuple[int, str]:
    try:
        return OPERATOR_ORDER.index(name), name
    except ValueError:
        return len(OPERATOR_ORDER), name


def score_pair(
    manifest_row: dict[str, str],
    manifest_dir: Path,
    unscored_policy: str,
) -> list[dict[str, Any]]:
    clean_path = resolve_path(manifest_row["clean"], manifest_dir)
    attacked_path = resolve_path(manifest_row["attacked"], manifest_dir)
    clean_rows = core.load_jsonl(clean_path)
    attacked_rows = core.load_jsonl(attacked_path)

    clean_by_id, _clean_canonicalization = core.canonical_clean_judgments(
        clean_rows
    )
    reported_model = core.output_model_name(list(clean_by_id.values()))
    if reported_model != manifest_row["model"]:
        raise ValueError(
            "Manifest model does not match clean judgments: "
            f"manifest={manifest_row['model']!r}, reported={reported_model!r}"
        )

    clean_outcomes = {
        sample_id: core.judge_outcome(row)
        for sample_id, row in clean_by_id.items()
    }
    pairs_by_operator: dict[
        str, list[tuple[str, bool | None, bool | None]]
    ] = defaultdict(list)
    unmatched_by_operator: dict[str, int] = defaultdict(int)

    for row in attacked_rows:
        attack_type = row.get("attack_type")
        operator = (
            attack_type
            if isinstance(attack_type, str) and attack_type
            else "__missing_attack_type__"
        )
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            unmatched_by_operator[operator] += 1
            continue
        clean = clean_by_id.get(source_id)
        if clean is None:
            unmatched_by_operator[operator] += 1
            continue
        pairs_by_operator[operator].append(
            (
                source_id,
                core.judge_outcome(clean),
                core.judge_outcome(row),
            )
        )

    operators = sorted(
        set(pairs_by_operator) | set(unmatched_by_operator),
        key=operator_sort_key,
    )
    return [
        core.summarize(
            operator,
            pairs_by_operator[operator],
            unmatched_by_operator[operator],
            unscored_policy,
            clean_outcomes,
        )
        for operator in operators
    ]


def scaled(value: Any, scale: str) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    multiplier = 100.0 if scale == "percent" else 1.0
    return round(float(value) * multiplier, 6)


def build_long_rows(
    manifest_rows: list[dict[str, str]],
    manifest_dir: Path,
    unscored_policy: str,
    scale: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest_rows, start=1):
        print(
            f"[{index}/{len(manifest_rows)}] "
            f"dataset={manifest_row['dataset']} model={manifest_row['model']}"
        )
        for summary in score_pair(
            manifest_row,
            manifest_dir,
            unscored_policy,
        ):
            operator = summary["group"]
            rows.append(
                {
                    "dataset": manifest_row["dataset"],
                    "model": display_model_name(manifest_row["model"]),
                    "level": ROBUSTNESS_LEVELS.get(operator, "Unknown"),
                    "attack_type": operator,
                    "ASR": scaled(summary["attack_success_rate"], scale),
                    "Support": summary["support_clean_correct"],
                    "attack_successes": summary["attack_successes"],
                    "CRA": scaled(
                        summary["conditional_robust_accuracy"],
                        scale,
                    ),
                    "paired_records": summary["paired_records"],
                    "unmatched_records": summary["unmatched_clean"],
                }
            )
    return rows


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def format_matrix_cell(
    row: dict[str, Any] | None,
    scale: str,
    digits: int,
    markdown: bool = False,
) -> str:
    if row is None:
        return "--"
    support = int(row["Support"])
    value = row["ASR"]
    if value is None:
        rendered = "N/A"
    else:
        suffix = "%" if scale == "percent" else ""
        rendered = f"{float(value):.{digits}f}{suffix}"
    if markdown:
        return f"{rendered} (n={support})"
    return f"{rendered} (n={support})"


def matrix_rows(
    long_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    scale: str,
    digits: int,
) -> tuple[list[str], list[dict[str, str]]]:
    models = list(
        dict.fromkeys(display_model_name(row["model"]) for row in manifest_rows)
    )
    datasets = list(dict.fromkeys(row["dataset"] for row in manifest_rows))
    lookup = {
        (row["dataset"], row["model"], row["attack_type"]): row
        for row in long_rows
    }
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        operators = sorted(
            {
                row["attack_type"]
                for row in long_rows
                if row["dataset"] == dataset
            },
            key=operator_sort_key,
        )
        for operator in operators:
            output = {
                "dataset": dataset,
                "level": ROBUSTNESS_LEVELS.get(operator, "Unknown"),
                "attack_type": operator,
            }
            for model in models:
                output[model] = format_matrix_cell(
                    lookup.get((dataset, model, operator)),
                    scale,
                    digits,
                )
            rows.append(output)
    return models, rows


def markdown_text(
    matrix: list[dict[str, str]],
    models: list[str],
    scale: str,
) -> str:
    unit = "%" if scale == "percent" else "fraction"
    lines = [
        "# Per-operator Attack Success Rate",
        "",
        f"Each model cell reports ASR ({unit}) with clean-correct support `n`.",
        "",
    ]
    for dataset in dict.fromkeys(row["dataset"] for row in matrix):
        rows = [row for row in matrix if row["dataset"] == dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Level | Attack operator | "
                + " | ".join(models)
                + " |",
                "|---|---|" + "|".join("---:" for _ in models) + "|",
            ]
        )
        for row in rows:
            cells = [row[model] for model in models]
            lines.append(
                f"| {row['level']} | `{row['attack_type']}` | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_matrix_support(value: str) -> int:
    marker = "(n="
    if marker not in value:
        return 0
    return int(value.split(marker, 1)[1].rstrip(")"))


def _parse_matrix_value(value: str, scale: str) -> float | None:
    head = value.split(" ", 1)[0]
    if head == "N/A":
        return None
    if scale == "percent":
        head = head.rstrip("%")
    return float(head)


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def latex_text(
    matrix: list[dict[str, str]],
    models: list[str],
    scale: str,
    digits: int,
) -> str:
    columns = "lll" + ("r" * len(models))
    metric = r"ASR (\%)" if scale == "percent" else "ASR"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        (
            r"\caption{Per-operator attack success rate. "
            + f"Each model cell reports {metric} with clean-correct support "
            + r"in parentheses.}"
        ),
        r"\label{tab:operator_asr}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Dataset & Level & Attack operator & "
        + " & ".join(tex_escape(model) for model in models)
        + r" \\",
        r"\midrule",
    ]
    previous_dataset: str | None = None
    for row in matrix:
        dataset = row["dataset"]
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\midrule")
        rendered_cells = []
        for model in models:
            value = row[model]
            if value == "--":
                rendered_cells.append("--")
                continue
            support = _parse_matrix_support(value)
            metric_value = _parse_matrix_value(value, scale)
            rendered = (
                "N/A"
                if metric_value is None
                else f"{metric_value:.{digits}f}"
            )
            rendered_cells.append(f"{rendered} ({support})")
        lines.append(
            " & ".join(
                [
                    tex_escape(dataset) if dataset != previous_dataset else "",
                    tex_escape(row["level"]),
                    tex_escape(row["attack_type"]),
                    *rendered_cells,
                ]
            )
            + r" \\"
        )
        previous_dataset = dataset
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.digits < 0 or args.digits > 8:
        raise ValueError("--digits must be between 0 and 8")

    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else manifest.parent / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(manifest)
    long_rows = build_long_rows(
        manifest_rows,
        manifest.parent,
        args.unscored_policy,
        args.scale,
    )
    models, matrix = matrix_rows(
        long_rows,
        manifest_rows,
        args.scale,
        args.digits,
    )

    long_path = output_dir / f"{args.prefix}_long.csv"
    matrix_path = output_dir / f"{args.prefix}_matrix.csv"
    markdown_path = output_dir / f"{args.prefix}_table.md"
    latex_path = output_dir / f"{args.prefix}_table.tex"
    matrix_fields = ["dataset", "level", "attack_type", *models]

    core.atomic_write(long_path, csv_text(long_rows, LONG_COLUMNS))
    core.atomic_write(matrix_path, csv_text(matrix, matrix_fields))
    core.atomic_write(
        markdown_path,
        markdown_text(matrix, models, args.scale),
    )
    core.atomic_write(
        latex_path,
        latex_text(matrix, models, args.scale, args.digits),
    )

    print(f"Wrote {long_path}")
    print(f"Wrote {matrix_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
