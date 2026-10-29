from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


COMPUTE_DIR = Path(__file__).resolve().parents[1] / "compute-robust"
sys.path.insert(0, str(COMPUTE_DIR))

from build_operator_asr_table import matrix_rows, score_pair


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


class OperatorAsrTableTest(unittest.TestCase):
    def test_operator_asr_is_conditioned_on_clean_correct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.jsonl"
            attacked = root / "attacked.jsonl"
            write_jsonl(
                clean,
                [
                    {"id": 1, "model": "test-model", "is_correct": True},
                    {"id": 2, "model": "test-model", "is_correct": False},
                ],
            )
            write_jsonl(
                attacked,
                [
                    {
                        "id": "attack-1",
                        "source_id": "DA_1",
                        "attack_type": "row_order_shuffle",
                        "is_correct": False,
                    },
                    {
                        "id": "attack-2",
                        "source_id": "DA_2",
                        "attack_type": "row_order_shuffle",
                        "is_correct": True,
                    },
                    {
                        "id": "attack-3",
                        "source_id": "DA_1",
                        "attack_type": "column_order_shuffle",
                        "is_correct": True,
                    },
                    {
                        "id": "attack-4",
                        "source_id": "DA_404",
                        "attack_type": "column_order_shuffle",
                        "is_correct": False,
                    },
                ],
            )
            manifest_row = {
                "dataset": "Toy",
                "model": "test-model",
                "clean": clean.name,
                "attacked": attacked.name,
            }

            summaries = {
                row["group"]: row
                for row in score_pair(manifest_row, root, "incorrect")
            }

        row_shuffle = summaries["row_order_shuffle"]
        self.assertEqual(row_shuffle["support_clean_correct"], 1)
        self.assertEqual(row_shuffle["attack_successes"], 1)
        self.assertEqual(row_shuffle["attack_success_rate"], 1.0)

        column_shuffle = summaries["column_order_shuffle"]
        self.assertEqual(column_shuffle["support_clean_correct"], 1)
        self.assertEqual(column_shuffle["attack_successes"], 0)
        self.assertEqual(column_shuffle["attack_success_rate"], 0.0)
        self.assertEqual(column_shuffle["unmatched_clean"], 1)

    def test_matrix_cells_include_support(self) -> None:
        manifest = [
            {
                "dataset": "Toy",
                "model": "model-a",
                "clean": "clean.jsonl",
                "attacked": "attacked.jsonl",
            }
        ]
        long_rows = [
            {
                "dataset": "Toy",
                "model": "model-a",
                "level": "L0",
                "attack_type": "row_order_shuffle",
                "ASR": 25.0,
                "Support": 8,
            }
        ]

        models, rows = matrix_rows(long_rows, manifest, "percent", 2)

        self.assertEqual(models, ["model-a"])
        self.assertEqual(rows[0]["model-a"], "25.00% (n=8)")

    def test_duplicate_clean_ids_use_last_success_or_last_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.jsonl"
            attacked = root / "attacked.jsonl"
            write_jsonl(
                clean,
                [
                    {
                        "id": 1,
                        "model": "test-model",
                        "judge_status": "judge_error",
                        "score": 0,
                    },
                    {
                        "id": 1,
                        "model": "test-model",
                        "judge_status": "success",
                        "score": 1,
                    },
                    {
                        "id": 1,
                        "model": "test-model",
                        "judge_status": "judge_error",
                        "score": 0,
                    },
                    {
                        "id": 2,
                        "model": "test-model",
                        "judge_status": "judge_error",
                        "score": 0,
                    },
                    {
                        "id": 2,
                        "model": "test-model",
                        "judge_status": "skipped_infer_error",
                        "score": 0,
                    },
                ],
            )
            write_jsonl(
                attacked,
                [
                    {
                        "id": "attack-1",
                        "source_id": "DA_1",
                        "attack_type": "row_order_shuffle",
                        "is_correct": False,
                    },
                    {
                        "id": "attack-2",
                        "source_id": "DA_2",
                        "attack_type": "row_order_shuffle",
                        "is_correct": True,
                    },
                ],
            )
            summaries = score_pair(
                {
                    "dataset": "Toy",
                    "model": "test-model",
                    "clean": clean.name,
                    "attacked": attacked.name,
                },
                root,
                "incorrect",
            )

        summary = summaries[0]
        self.assertEqual(summary["support_clean_correct"], 1)
        self.assertEqual(summary["attack_successes"], 1)
        self.assertEqual(summary["attack_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
