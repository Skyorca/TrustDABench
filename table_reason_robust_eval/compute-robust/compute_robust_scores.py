#!/usr/bin/env python3
"""Compute paired robustness metrics from clean and attacked judgment JSONL files.

Attacked records are paired to clean sources through
``attacked.source_id == clean.id``. Existing micro metrics remain
attack-record weighted. Source-level metrics first average all attacks of the
same source, so each original question contributes at most once.

字段含义：
attack records: 扰动样本总数
support_clean_correct 原题答对数量
stable correct: 原题答对 且 扰动后也答对的数量
conditional_robust_accuracy： stable correct/support_clean_correct
attack_successes：扰动攻击造成错误的样本数
attack_success_rate ASR（核心指标）：attack_successes/support_clean_correct
robustness_accuracy_drop RAD（核心指标）：以原始题总数为分母，只累计
原题正确样本在攻击后的平均正确率损失；原题错误->攻击后正确不会抵消该损失
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRIC_COLUMNS = [
    "group",
    "attack_records",
    "paired_records",
    "original_correct_rate",
    "attacked_correct_rate",
    "accuracy_drop",
    "conditional_robust_accuracy",
    "attack_success_rate",
    "clean_dataset_accuracy",
    "robust_accuracy",
    "robustness_accuracy_drop",
    "unique_clean_records",
    "clean_dataset_correct",
    "attacked_unique_sources",
    "attack_coverage",
    "source_clean_correct_support",
    "source_balanced_cra",
    "source_balanced_asr",
    "support_clean_correct",
    "stable_correct",
    "attack_successes",
    "clean_incorrect_attack_correct",
    "clean_incorrect_attack_incorrect",
    "clean_unscored",
    "attacked_unscored",
    "unmatched_clean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True, help="Clean/original judgment JSONL.")
    parser.add_argument("--attacked", type=Path, required=True, help="Attacked judgment JSONL.")
    parser.add_argument("--output-json", type=Path, help="Summary JSON output path.")
    parser.add_argument("--output-csv", type=Path, help="Summary CSV output path.")
    parser.add_argument(
        "--unscored-policy",
        choices=("incorrect", "exclude"),
        default="incorrect",
        help="How missing/failed judge outcomes affect paired metrics (default: incorrect).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def judge_outcome(row: dict[str, Any]) -> bool | None:
    """Return True/False for a scored judgment and None for an unavailable one."""
    if row.get("_canonical_all_attempts_failed") is True:
        return False
    status = row.get("judge_status")
    if status not in (None, "success"):
        return None
    if isinstance(row.get("is_correct"), bool):
        return row["is_correct"]
    score = row.get("score")
    if isinstance(score, bool):
        return score
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
        return score == 1
    return None


def canonical_clean_judgments(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Collapse repeated clean judgments using the last successful attempt.

    A successful attempt is any row for which ``judge_outcome`` returns a
    usable boolean result. Later failed attempts therefore never overwrite an
    earlier successful judgment. If an ID has no successful attempt, its last
    physical row is retained and explicitly treated as a failed clean sample,
    including when attacked unscored outcomes are otherwise excluded.
    """
    ordered_ids: list[str] = []
    latest: dict[str, dict[str, Any]] = {}
    latest_success: dict[str, dict[str, Any]] = {}
    physical_rows = 0
    summary_rows = 0

    for row in rows:
        physical_rows += 1
        sample_id = normalize_clean_id(row.get("id"))
        if sample_id == "__summary__":
            summary_rows += 1
            continue
        if sample_id not in latest:
            ordered_ids.append(sample_id)
        latest[sample_id] = row
        if judge_outcome(row) is not None:
            latest_success[sample_id] = row

    selected: dict[str, dict[str, Any]] = {}
    for sample_id in ordered_ids:
        successful = latest_success.get(sample_id)
        if successful is not None:
            selected[sample_id] = successful
            continue
        failed = dict(latest[sample_id])
        failed["_canonical_all_attempts_failed"] = True
        selected[sample_id] = failed
    selected_successes = len(latest_success)
    return selected, {
        "physical_rows": physical_rows,
        "summary_rows": summary_rows,
        "unique_tasks": len(selected),
        "collapsed_attempts": physical_rows - summary_rows - len(selected),
        "selected_successes": selected_successes,
        "selected_failures": len(selected) - selected_successes,
    }


def format_rate(numerator: float, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def output_model_name(clean_rows: list[dict[str, Any]]) -> str:
    """Return the single declared clean-run model name for default output names."""
    models = {str(row["model"]).strip() for row in clean_rows if str(row.get("model", "")).strip()}
    if len(models) != 1:
        raise ValueError(
            "Default output names require exactly one nonempty clean model value; "
            f"found {sorted(models)!r}. Pass --output-json and --output-csv explicitly for mixed inputs."
        )
    return next(iter(models))


def normalize_clean_id(value: Any) -> str:
    """Normalize clean IDs to the source_id convention used by attacked rows."""
    if isinstance(value, bool) or value is None:
        raise ValueError("Every clean row must contain a nonempty string or integer id")
    if isinstance(value, int):
        return f"DA_{value}"
    if isinstance(value, str) and value:
        return value
    raise ValueError("Every clean row must contain a nonempty string or integer id")


def source_level_metrics(
    records: list[tuple[str, bool | None, bool | None]],
    clean_outcomes: dict[str, bool | None],
    policy: str,
) -> dict[str, Any]:
    if policy == "exclude":
        evaluated_clean = {
            source_id: outcome
            for source_id, outcome in clean_outcomes.items()
            if outcome is not None
        }
    else:
        evaluated_clean = {
            source_id: False if outcome is None else outcome
            for source_id, outcome in clean_outcomes.items()
        }

    attacked_by_source: dict[str, list[bool]] = defaultdict(list)
    attacked_sources = set()
    for source_id, _clean, attacked in records:
        if source_id not in evaluated_clean:
            continue
        attacked_sources.add(source_id)
        if policy == "exclude" and attacked is None:
            continue
        attacked_by_source[source_id].append(False if attacked is None else attacked)

    unique_clean_records = len(evaluated_clean)
    clean_dataset_correct = sum(outcome is True for outcome in evaluated_clean.values())
    clean_dataset_accuracy = format_rate(clean_dataset_correct, unique_clean_records)

    source_correct_rates: list[float] = []
    for source_id, attacked_values in attacked_by_source.items():
        if evaluated_clean[source_id] is not True or not attacked_values:
            continue
        source_correct_rates.append(sum(attacked_values) / len(attacked_values))

    source_clean_correct_support = len(source_correct_rates)
    source_stable_mass = sum(source_correct_rates)
    source_failure_mass = sum(1.0 - value for value in source_correct_rates)
    source_balanced_cra = format_rate(source_stable_mass, source_clean_correct_support)
    source_balanced_asr = format_rate(source_failure_mass, source_clean_correct_support)
    robustness_accuracy_drop = format_rate(source_failure_mass, unique_clean_records)
    robust_accuracy = (
        round(clean_dataset_accuracy - robustness_accuracy_drop, 6)
        if clean_dataset_accuracy is not None and robustness_accuracy_drop is not None
        else None
    )

    return {
        "clean_dataset_accuracy": clean_dataset_accuracy,
        "robust_accuracy": robust_accuracy,
        "robustness_accuracy_drop": robustness_accuracy_drop,
        "unique_clean_records": unique_clean_records,
        "clean_dataset_correct": clean_dataset_correct,
        "attacked_unique_sources": len(attacked_sources),
        "attack_coverage": format_rate(len(attacked_sources), unique_clean_records),
        "source_clean_correct_support": source_clean_correct_support,
        "source_balanced_cra": source_balanced_cra,
        "source_balanced_asr": source_balanced_asr,
    }


def summarize(
    group: str,
    records: Iterable[tuple[str, bool | None, bool | None]],
    unmatched_clean: int,
    policy: str,
    clean_outcomes: dict[str, bool | None],
) -> dict[str, Any]:
    materialized = list(records)
    pairs = [(clean, attacked) for _source_id, clean, attacked in materialized]
    clean_unscored = sum(clean is None for clean, _ in pairs)
    attacked_unscored = sum(attacked is None for _, attacked in pairs)

    if policy == "exclude":
        evaluated = [(clean, attacked) for clean, attacked in pairs if clean is not None and attacked is not None]
    else:
        evaluated = [(False if clean is None else clean, False if attacked is None else attacked) for clean, attacked in pairs]

    original_correct = sum(clean is True for clean, _ in evaluated)
    attacked_correct = sum(attacked is True for _, attacked in evaluated)
    stable_correct = sum(clean is True and attacked is True for clean, attacked in evaluated)
    attack_successes = sum(clean is True and attacked is False for clean, attacked in evaluated)
    improved = sum(clean is False and attacked is True for clean, attacked in evaluated)
    base_failure = sum(clean is False and attacked is False for clean, attacked in evaluated)
    original_correct_rate = format_rate(original_correct, len(evaluated))
    attacked_correct_rate = format_rate(attacked_correct, len(evaluated))
    accuracy_drop = (
        round(original_correct_rate - attacked_correct_rate, 6)
        if original_correct_rate is not None and attacked_correct_rate is not None
        else None
    )

    result = {
        "group": group,
        "attack_records": len(materialized) + unmatched_clean,
        "paired_records": len(materialized),
        "original_correct_rate": original_correct_rate,
        "attacked_correct_rate": attacked_correct_rate,
        "accuracy_drop": accuracy_drop,
        "conditional_robust_accuracy": format_rate(stable_correct, original_correct),
        "attack_success_rate": format_rate(attack_successes, original_correct),
        "support_clean_correct": original_correct,
        "stable_correct": stable_correct,
        "attack_successes": attack_successes,
        "clean_incorrect_attack_correct": improved,
        "clean_incorrect_attack_incorrect": base_failure,
        "clean_unscored": clean_unscored,
        "attacked_unscored": attacked_unscored,
        "unmatched_clean": unmatched_clean,
    }
    result.update(source_level_metrics(materialized, clean_outcomes, policy))
    return result


def csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=METRIC_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    args = parse_args()
    clean_rows = load_jsonl(args.clean)
    attacked_rows = load_jsonl(args.attacked)
    clean_by_id, clean_canonicalization = canonical_clean_judgments(clean_rows)
    model_name = output_model_name(list(clean_by_id.values()))

    clean_outcomes = {
        sample_id: judge_outcome(row)
        for sample_id, row in clean_by_id.items()
    }
    pairs_by_group: dict[str, list[tuple[str, bool | None, bool | None]]] = defaultdict(list)
    unmatched_by_group: dict[str, int] = defaultdict(int)
    unmatched_records: list[dict[str, Any]] = []
    missing_source_id: list[str] = []

    for row in attacked_rows:
        attack_id = row.get("id")
        group = row.get("attack_type")
        group = group if isinstance(group, str) and group else "__missing_attack_type__"
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            missing_source_id.append(str(attack_id))
            unmatched_by_group[group] += 1
            unmatched_records.append({"id": attack_id, "source_id": source_id, "attack_type": group, "reason": "missing_source_id"})
            continue
        clean = clean_by_id.get(source_id)
        if clean is None:
            unmatched_by_group[group] += 1
            unmatched_records.append({"id": attack_id, "source_id": source_id, "attack_type": group, "reason": "clean_source_not_found"})
            continue
        pairs_by_group[group].append((source_id, judge_outcome(clean), judge_outcome(row)))

    groups = sorted(set(pairs_by_group) | set(unmatched_by_group))
    per_attack = [
        summarize(
            group,
            pairs_by_group[group],
            unmatched_by_group[group],
            args.unscored_policy,
            clean_outcomes,
        )
        for group in groups
    ]
    overall = summarize(
        "overall",
        [pair for group_pairs in pairs_by_group.values() for pair in group_pairs],
        sum(unmatched_by_group.values()),
        args.unscored_policy,
        clean_outcomes,
    )
    report_rows = [overall, *per_attack]

    output_json = args.output_json or args.attacked.with_name(f"{model_name}_robustness_summary.json")
    output_csv = args.output_csv or args.attacked.with_name(f"{model_name}_robustness_summary.csv")
    report = {
        "metric_definition": {
            "pairing": "attacked.source_id == normalized clean.id",
            "core_metrics": ["attack_success_rate", "robustness_accuracy_drop"],
            "aggregation": {
                "micro_metrics": "attack-record weighted",
                "source_metrics": "average attacks within each source before aggregating unique original questions",
            },
            "original_correct_rate": "P(clean correct) on the paired attack-record population",
            "attacked_correct_rate": "P(attacked correct) on the paired attack-record population",
            "accuracy_drop": "original_correct_rate - attacked_correct_rate",
            "conditional_robust_accuracy": "P(attacked correct | clean correct)",
            "attack_success_rate": "P(attacked incorrect | clean correct)",
            "support_clean_correct": "number of paired records with a clean-correct outcome",
            "clean_dataset_accuracy": "clean-correct unique original questions / evaluable unique original questions",
            "source_balanced_cra": "mean attacked correctness per clean-correct source, then mean across supported sources",
            "source_balanced_asr": "1 - source_balanced_cra",
            "robustness_accuracy_drop": (
                "sum over supported clean-correct sources of "
                "(1 - mean attacked correctness for that source), divided by "
                "the number of evaluable unique original questions; "
                "clean-incorrect to attacked-correct transitions never offset the loss"
            ),
            "robust_accuracy": "clean_dataset_accuracy - robustness_accuracy_drop",
            "attack_coverage": "unique original questions with at least one matched attack / unique original questions",
            "unscored_policy": args.unscored_policy,
        },
        "model": model_name,
        "inputs": {"clean": str(args.clean), "attacked": str(args.attacked)},
        "clean_canonicalization": clean_canonicalization,
        "clean_rows": len(clean_rows),
        "attacked_rows": len(attacked_rows),
        "summary": report_rows,
        "unmatched_records": unmatched_records,
        "missing_source_id": missing_source_id,
    }
    atomic_write(output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output_csv, csv_text(report_rows))

    print("Robustness summary (micro ASR + unique-source robustness accuracy drop):")
    for row in report_rows:
        def pct(value: float | None) -> str:
            return "N/A" if value is None else f"{value * 100:.2f}%"

        print(
            f"{row['group']}: clean={pct(row['original_correct_rate'])}, "
            f"attacked={pct(row['attacked_correct_rate'])}, "
            f"drop={pct(row['accuracy_drop'])}, "
            f"CRA={pct(row['conditional_robust_accuracy'])}, "
            f"ASR={pct(row['attack_success_rate'])}, "
            f"clean_dataset={pct(row['clean_dataset_accuracy'])}, "
            f"robust_acc={pct(row['robust_accuracy'])}, "
            f"RAD={pct(row['robustness_accuracy_drop'])}, "
            f"coverage={pct(row['attack_coverage'])}, "
            f"support={row['support_clean_correct']}, paired={row['paired_records']}, "
            f"unmatched={row['unmatched_clean']}"
        )
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
