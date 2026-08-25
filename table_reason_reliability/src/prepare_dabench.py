from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path} 第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc

    return rows


def build_full_question(row: Dict[str, Any]) -> str:
    parts = [str(row.get("question") or "").strip()]

    constraints = str(row.get("constraints") or "").strip()
    output_format = str(row.get("format") or "").strip()

    if constraints:
        parts.append(f"Constraints:\n{constraints}")

    if output_format:
        parts.append(f"Required output format:\n{output_format}")

    return "\n\n".join(part for part in parts if part)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 DABench questions 和 labels 合并为项目输入格式"
    )

    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--table-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "skip"],
        default="error",
        help="表格缺失时中止或跳过，正式运行建议先使用 error",
    )

    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    labels = load_jsonl(args.labels)

    labels_by_id: Dict[str, Dict[str, Any]] = {}

    for label in labels:
        label_id = str(label.get("id"))

        if label_id in labels_by_id:
            raise RuntimeError(f"labels 中存在重复 ID：{label_id}")

        labels_by_id[label_id] = label

    converted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    seen_question_ids = set()

    for question_row in questions:
        original_id = str(question_row.get("id"))

        if original_id in seen_question_ids:
            raise RuntimeError(
                f"questions 中存在重复 ID：{original_id}"
            )

        seen_question_ids.add(original_id)

        label_row = labels_by_id.get(original_id)

        if label_row is None:
            raise RuntimeError(
                f"问题 ID={original_id} 找不到对应 label"
            )

        file_name = str(
            question_row.get("file_name") or ""
        ).strip()

        if not file_name:
            raise RuntimeError(
                f"问题 ID={original_id} 缺少 file_name"
            )

        table_path = args.table_root / file_name

        if not table_path.is_file():
            missing_record = {
                "id": original_id,
                "file_name": file_name,
                "reason": "table_not_found",
            }

            if args.missing_policy == "error":
                raise FileNotFoundError(
                    f"问题 ID={original_id} 引用的表格不存在："
                    f"{table_path}"
                )

            skipped.append(missing_record)
            continue

        common_answers = label_row.get("common_answers", [])

        converted.append(
            {
                "id": f"DA_{original_id}",
                "question": build_full_question(question_row),
                "input_file": file_name,
                "reference": json.dumps(
                    {"common_answers": common_answers},
                    ensure_ascii=False,
                ),
                "output_type": "data_analysis",
                "metadata": {
                    "source": "DABench",
                    "original_id": question_row.get("id"),
                    "concepts": question_row.get("concepts", []),
                    "level": question_row.get("level"),
                    "original_question": question_row.get(
                        "question"
                    ),
                    "constraints": question_row.get(
                        "constraints"
                    ),
                    "required_format": question_row.get(
                        "format"
                    ),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        for row in converted:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    missing_report = args.output.with_name(
        args.output.stem + "_missing.json"
    )

    missing_report.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    label_ids = set(labels_by_id)
    question_ids = {
        str(row.get("id")) for row in questions
    }

    labels_without_questions = sorted(label_ids - question_ids)

    print(f"questions={len(questions)}")
    print(f"labels={len(labels)}")
    print(f"converted={len(converted)}")
    print(f"skipped_missing_tables={len(skipped)}")
    print(
        f"labels_without_questions="
        f"{len(labels_without_questions)}"
    )
    print(f"output={args.output}")
    print(f"missing_report={missing_report}")


if __name__ == "__main__":
    main()