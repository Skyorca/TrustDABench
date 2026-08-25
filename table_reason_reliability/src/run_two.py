from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

import yaml


TARGET_ATTACKS = (
    "file_missing",
    "deep_analysis_missing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fully rebuild file_missing and deep_analysis_missing "
            "for AIDA-QA and DABench under outputs/two."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "config_aida.yaml",
            "config_dabench.yaml",
        ],
        help="Base dataset configurations, run sequentially.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for testing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional worker override.",
    )
    parser.add_argument(
        "--keep-generated-config",
        action="store_true",
        help="Keep generated YAML files for inspection.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(
            f"Invalid YAML object: {path}"
        )
    return value


def dataset_name(
    config: Dict[str, Any],
    config_path: Path,
) -> str:
    combined = config.get("combined_output_path")
    if combined:
        stem = PurePosixPath(str(combined)).stem
        if "_combined" in stem:
            return stem.split("_combined", 1)[0]
        return stem

    name = config_path.stem
    if name.startswith("config_"):
        name = name[len("config_") :]
    return name


def original_output_root(
    config: Dict[str, Any],
    project_root: Path,
) -> PurePosixPath:
    combined = config.get("combined_output_path")
    if combined:
        return PurePosixPath(str(combined)).parent
    return PurePosixPath(
        str(project_root / "outputs")
    )


def find_source_phase(
    phases: List[Dict[str, Any]],
    attack_type: str,
) -> Dict[str, Any]:
    for phase in phases:
        enabled = (
            phase.get("attacks", {})
            .get("enabled", [])
        )
        if attack_type in enabled:
            return phase
    raise ValueError(
        f"No phase enables attack '{attack_type}'"
    )


def build_two_config(
    base_config: Dict[str, Any],
    base_path: Path,
    project_root: Path,
) -> tuple[Dict[str, Any], PurePosixPath]:
    config = copy.deepcopy(base_config)
    source_phases = config.get("phases")
    if (
        not isinstance(source_phases, list)
        or not source_phases
    ):
        raise ValueError(
            f"No phases configured in {base_path}"
        )

    name = dataset_name(config, base_path)
    dataset_root = (
        original_output_root(config, project_root)
        / "two"
        / name
    )

    targeted_phases: List[Dict[str, Any]] = []

    for attack_type in TARGET_ATTACKS:
        phase = copy.deepcopy(
            find_source_phase(
                source_phases,
                attack_type,
            )
        )
        phase["name"] = attack_type
        phase["output_dir"] = str(
            dataset_root / attack_type
        )

        attack_config = phase.setdefault(
            "attacks",
            {},
        )
        attack_config["enabled"] = [
            attack_type
        ]
        attack_config[
            "max_attacks_per_sample"
        ] = 1

        thresholds = attack_config.get(
            "min_confidence_by_attack"
        )
        if isinstance(thresholds, dict):
            attack_config[
                "min_confidence_by_attack"
            ] = {
                attack_type: thresholds.get(
                    attack_type,
                    attack_config.get(
                        "min_confidence",
                        0.0,
                    ),
                )
            }

        targeted_phases.append(phase)

    config["phases"] = targeted_phases
    config["combined_output_path"] = str(
        dataset_root / "attack_dataset.jsonl"
    )

    runner = config.setdefault("runner", {})
    runner.pop("sample_log_dir_name", None)
    runner.pop("retry_existing_rejected", None)

    return config, dataset_root


def generated_config_path(
    base_path: Path,
) -> Path:
    return base_path.parent / (
        f".{base_path.stem}_run_two_generated.yaml"
    )


def remove_previous_dataset_output(
    dataset_root: PurePosixPath,
) -> None:
    path = Path(str(dataset_root))
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def run_one_dataset(
    project_root: Path,
    base_path: Path,
    limit: int | None,
    num_workers: int | None,
    keep_generated_config: bool,
) -> None:
    base_config = load_yaml(base_path)
    targeted_config, dataset_root = (
        build_two_config(
            base_config=base_config,
            base_path=base_path,
            project_root=project_root,
        )
    )

    # A run_two invocation is always a clean full run.
    # It never resumes from logs or existing attack results.
    remove_previous_dataset_output(
        dataset_root
    )

    generated_path = generated_config_path(
        base_path
    )
    with generated_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            targeted_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    command = [
        sys.executable,
        "-u",
        str(
            project_root
            / "src"
            / "construct_dataset.py"
        ),
        "--config",
        str(generated_path),
        "--phase",
        "all",
    ]
    if limit is not None:
        command.extend(
            ["--limit", str(limit)]
        )
    if num_workers is not None:
        command.extend(
            [
                "--num-workers",
                str(num_workers),
            ]
        )

    print(
        f"[RUN_TWO] start dataset={dataset_name(base_config, base_path)} "
        f"output={dataset_root}",
        flush=True,
    )

    try:
        subprocess.run(
            command,
            cwd=project_root,
            env=os.environ.copy(),
            check=True,
        )
    finally:
        if (
            not keep_generated_config
            and generated_path.exists()
        ):
            generated_path.unlink()

    print(
        f"[RUN_TWO] done dataset={dataset_name(base_config, base_path)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    for config_value in args.configs:
        base_path = Path(config_value)
        if not base_path.is_absolute():
            base_path = (
                project_root / base_path
            )
        base_path = base_path.resolve()

        if not base_path.exists():
            raise FileNotFoundError(
                f"Config not found: {base_path}"
            )

        run_one_dataset(
            project_root=project_root,
            base_path=base_path,
            limit=args.limit,
            num_workers=args.num_workers,
            keep_generated_config=(
                args.keep_generated_config
            ),
        )


if __name__ == "__main__":
    main()
