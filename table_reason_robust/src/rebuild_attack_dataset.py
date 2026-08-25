from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.construct_dataset import rebuild_attack_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild attack_dataset.jsonl from accepted attack records without running any models."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Generation output directory containing accepted/*.json.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    accepted_dir = output_dir / "accepted"
    if not accepted_dir.is_dir():
        raise RuntimeError(f"accepted directory does not exist: {accepted_dir}")
    rebuild_attack_dataset(output_dir)
    count = sum(1 for _ in accepted_dir.glob("*.json"))
    print(f"Rebuilt {output_dir / 'attack_dataset.jsonl'} from {count} accepted records.")


if __name__ == "__main__":
    main()
