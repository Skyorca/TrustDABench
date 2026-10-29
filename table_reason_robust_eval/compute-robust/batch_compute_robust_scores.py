#!/usr/bin/env python3
"""Batch robustness scoring from a CSV manifest.

The manifest must contain four columns:
dataset, model, clean, attacked

Each row is evaluated by invoking compute_robust_scores.py. The final CSV
contains exactly: dataset, model, ASR, RAD.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = ["dataset", "model", "ASR", "RAD"]
MANIFEST_COLUMNS = ["dataset", "model", "clean", "attacked"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV with dataset, model, clean, and attacked columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Aggregated CSV path. Defaults to <manifest_stem>_scores.csv.",
    )
    parser.add_argument(
        "--compute-script",
        type=Path,
        default=Path(__file__).with_name("compute_robust_scores.py"),
        help="Path to compute_robust_scores.py.",
    )
    parser.add_argument(
        "--unscored-policy",
        choices=("incorrect", "exclude"),
        default="incorrect",
    )
    parser.add_argument(
        "--scale",
        choices=("percent", "fraction"),
        default="percent",
        help="Write rates as percentages (15.78) or fractions (0.1578).",
    )
    return parser.parse_args()


def resolve_input_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value.strip())
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in MANIFEST_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")

        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            normalized = {
                name: str(row.get(name) or "").strip()
                for name in MANIFEST_COLUMNS
            }
            empty = [name for name, value in normalized.items() if not value]
            if empty:
                raise ValueError(
                    f"Manifest line {line_number} has empty fields: {', '.join(empty)}"
                )
            rows.append(normalized)

    if not rows:
        raise ValueError("Manifest contains no scoring rows")
    return rows


def find_overall(report: dict[str, Any]) -> dict[str, Any]:
    for row in report.get("summary", []):
        if row.get("group") == "overall":
            return row
    raise ValueError("compute_robust_scores.py output has no overall summary")


def format_metric(value: Any, scale: str) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Expected numeric metric, got {value!r}")
    scaled = float(value) * (100.0 if scale == "percent" else 1.0)
    return f"{scaled:.6f}".rstrip("0").rstrip(".")


def score_one(
    row: dict[str, str],
    manifest_dir: Path,
    compute_script: Path,
    temporary_dir: Path,
    unscored_policy: str,
    scale: str,
    index: int,
) -> dict[str, str]:
    clean_path = resolve_input_path(row["clean"], manifest_dir)
    attacked_path = resolve_input_path(row["attacked"], manifest_dir)
    for label, path in (("clean", clean_path), ("attacked", attacked_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    detail_json = temporary_dir / f"{index:04d}.json"
    detail_csv = temporary_dir / f"{index:04d}.csv"
    command = [
        sys.executable,
        str(compute_script),
        "--clean",
        str(clean_path),
        "--attacked",
        str(attacked_path),
        "--output-json",
        str(detail_json),
        "--output-csv",
        str(detail_csv),
        "--unscored-policy",
        unscored_policy,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"Scoring failed for dataset={row['dataset']}, model={row['model']}: {detail}"
        )

    report = json.loads(detail_json.read_text(encoding="utf-8"))
    reported_model = str(report.get("model") or "").strip()
    if reported_model != row["model"]:
        raise ValueError(
            "Manifest model does not match the clean result: "
            f"manifest={row['model']!r}, reported={reported_model!r}"
        )

    overall = find_overall(report)
    return {
        "dataset": row["dataset"],
        "model": row["model"],
        "ASR": format_metric(overall.get("attack_success_rate"), scale),
        "RAD": format_metric(overall.get("robustness_accuracy_drop"), scale),
    }


def write_csv_atomically(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    compute_script = args.compute_script.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    if not compute_script.is_file():
        raise FileNotFoundError(f"Compute script does not exist: {compute_script}")

    output = (
        args.output.resolve()
        if args.output is not None
        else manifest.with_name(f"{manifest.stem}_scores.csv")
    )
    manifest_rows = load_manifest(manifest)

    results: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="batch-robust-scores-") as temp:
        temporary_dir = Path(temp)
        for index, row in enumerate(manifest_rows, start=1):
            print(
                f"[{index}/{len(manifest_rows)}] "
                f"dataset={row['dataset']} model={row['model']}"
            )
            results.append(
                score_one(
                    row=row,
                    manifest_dir=manifest.parent,
                    compute_script=compute_script,
                    temporary_dir=temporary_dir,
                    unscored_policy=args.unscored_policy,
                    scale=args.scale,
                    index=index,
                )
            )

    write_csv_atomically(output, results)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
