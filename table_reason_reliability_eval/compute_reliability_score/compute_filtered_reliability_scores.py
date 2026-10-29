#!/usr/bin/env python3
"""Recompute reliability reports after excluding out-of-scope source questions."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


DEFAULT_EXCLUDED_SOURCE_IDS = {
    "QA_19",
    "QA_20",
    "QA_73",
    "QA_74",
    "QA_75",
    "QA_76",
    "QA_77",
    "QA_78",
    "QA_79",
    "QA_94",
    "QA_96",
    "QA_107",
    "QA_113",
    "QA_115",
}

SOURCE_ID_PATTERN = re.compile(r"^(QA_\d+)(?:__|$)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter judgments associated with excluded source questions and "
            "recompute reports using judge_refusal.py's metric schema."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Judgments JSONL files, directories, or glob patterns. Directories "
            "are searched recursively for '*_judgments.jsonl'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for all generated reports and filtered JSONL files.",
    )
    parser.add_argument(
        "--exclude-source-ids",
        default=",".join(sorted(DEFAULT_EXCLUDED_SOURCE_IDS, key=source_sort_key)),
        help="Comma-separated source IDs to exclude.",
    )
    parser.add_argument(
        "--write-filtered-jsonl",
        action="store_true",
        help="Also write the retained judgment rows to '*_tabular_judgments.jsonl'.",
    )
    return parser.parse_args()


def source_sort_key(value: str) -> tuple[str, int]:
    prefix, _, number = value.rpartition("_")
    try:
        return prefix, int(number)
    except ValueError:
        return value, -1


def expand_inputs(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(candidate.rglob("*_judgments.jsonl"))
            continue
        if candidate.is_file():
            paths.append(candidate)
            continue
        paths.extend(Path(match) for match in glob.glob(value, recursive=True))

    unique: List[Path] = []
    seen: Set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or path.name.endswith("_tabular_judgments.jsonl"):
            continue
        if path.suffix.lower() != ".jsonl":
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def infer_source_id(row: Dict[str, Any]) -> str:
    source_id = str(row.get("source_id") or "").strip()
    if source_id:
        return source_id.upper()

    sample_id = str(row.get("id") or "").strip()
    match = SOURCE_ID_PATTERN.match(sample_id)
    return match.group(1).upper() if match else ""


def count_labels(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("label") or "unknown")] += 1
    return dict(sorted(counts.items()))


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    evaluable_rows = [row for row in rows if row.get("judge_status") == "success"]
    evaluable = len(evaluable_rows)
    scores = [float(row.get("score") or 0) for row in rows]
    evaluable_scores = [float(row.get("score") or 0) for row in evaluable_rows]
    success = sum(score >= 1 for score in scores)
    evaluable_success = sum(score >= 1 for score in evaluable_scores)
    partial = sum(score == 0.5 for score in scores)
    failed = sum(score == 0 for score in scores)
    return {
        "total": total,
        "evaluable": evaluable,
        "not_evaluable": total - evaluable,
        "mean_score_all": round(sum(scores) / total, 4) if total else 0,
        "mean_score_evaluable": round(sum(evaluable_scores) / evaluable, 4) if evaluable else 0,
        "success_rate_all": round(success / total, 4) if total else 0,
        "success_rate_evaluable": round(evaluable_success / evaluable, 4) if evaluable else 0,
        "partial_rate_all": round(partial / total, 4) if total else 0,
        "failure_rate_all": round(failed / total, 4) if total else 0,
        "label_counts": count_labels(rows),
    }


def build_report(
    rows: Sequence[Dict[str, Any]],
    *,
    input_path: Path,
    original_total: int,
    excluded_rows: Sequence[Dict[str, Any]],
    requested_exclusions: Set[str],
) -> Dict[str, Any]:
    by_attack: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_attack[str(row.get("attack_type") or "unknown")].append(row)
        by_model[str(row.get("model") or "unknown")].append(row)

    excluded_counts: Dict[str, int] = defaultdict(int)
    for row in excluded_rows:
        excluded_counts[infer_source_id(row) or "unknown"] += 1

    return {
        "total": len(rows),
        **summarize_rows(rows),
        "by_attack_type": {
            key: summarize_rows(value) for key, value in sorted(by_attack.items())
        },
        "by_model": {
            key: summarize_rows(value) for key, value in sorted(by_model.items())
        },
        "filter": {
            "input_file": str(input_path.resolve()),
            "original_total": original_total,
            "retained_total": len(rows),
            "excluded_total": len(excluded_rows),
            "requested_excluded_source_ids": sorted(
                requested_exclusions, key=source_sort_key
            ),
            "matched_excluded_source_ids": sorted(
                excluded_counts, key=source_sort_key
            ),
            "excluded_rows_by_source_id": {
                key: excluded_counts[key]
                for key in sorted(excluded_counts, key=source_sort_key)
            },
        },
    }


def output_base_name(path: Path) -> str:
    suffix = "_refusal_judgments"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(
    path: Path,
    *,
    excluded_ids: Set[str],
    output_dir: Path | None,
    write_filtered_jsonl: bool,
) -> tuple[Path, Dict[str, Any]]:
    all_rows = list(iter_jsonl(path))
    retained: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for row in all_rows:
        target = excluded if infer_source_id(row) in excluded_ids else retained
        target.append(row)

    destination = output_dir or path.parent
    destination.mkdir(parents=True, exist_ok=True)
    base = output_base_name(path)
    report_path = destination / f"{base}_tabular_refusal_report.json"
    report = build_report(
        retained,
        input_path=path,
        original_total=len(all_rows),
        excluded_rows=excluded,
        requested_exclusions=excluded_ids,
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if write_filtered_jsonl:
        write_jsonl(destination / f"{base}_tabular_judgments.jsonl", retained)

    return report_path, report


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No judgments JSONL files matched the supplied inputs.")

    excluded_ids = {
        item.strip().upper()
        for item in args.exclude_source_ids.split(",")
        if item.strip()
    }
    for path in input_paths:
        report_path, report = process_file(
            path,
            excluded_ids=excluded_ids,
            output_dir=args.output_dir,
            write_filtered_jsonl=args.write_filtered_jsonl,
        )
        filter_info = report["filter"]
        print(
            f"{path}: {filter_info['original_total']} -> "
            f"{filter_info['retained_total']} rows "
            f"(excluded {filter_info['excluded_total']}); report={report_path}"
        )


if __name__ == "__main__":
    main()
