from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent import (
    AgentFatalError,
    AgentOperationTimeout,
    AgentOutputError,
    AgentTransientError,
    OpenAIWorkspaceAgent,
    expand_env,
)
from src.dataset import compact_json, load_jsonl, profile_files, resolve_input_files, rows_to_limit
from src.operators import Operator, get_enabled_operators
from src.runner import Runner
# from src.validators import validate_unanswerable_payload, validation_failure_reason
from src.workspace import AttackWorkspace
from src.validators import (
    validate_unanswerable_payload,
    validation_failure_reason,
)

QUESTION_MUST_BE_UNCHANGED = {
    "data_missing",
    "evidence_conflict",
    "file_missing",
    "header_conflict",
    "deep_analysis_missing",
    "structural_context_missing",
}

class AttackTimeoutError(Exception):
    def __init__(
        self,
        attack_id: str,
        stage: str,
        elapsed_sec: float,
        timeout_sec: float,
    ):
        self.attack_id = attack_id
        self.stage = stage
        self.elapsed_sec = elapsed_sec
        self.timeout_sec = timeout_sec
        super().__init__(
            f"attack timeout: id={attack_id}, "
            f"stage={stage}, elapsed={elapsed_sec:.1f}s, "
            f"limit={timeout_sec:.1f}s"
        )

def main() -> None:
    args = parse_args()

    config_path = Path(
        args.config
    ).resolve()

    project_root = config_path.parent
    base_config = load_config(config_path)

    phases = base_config.get(
        "phases",
        [],
    )

    if not phases:
        raise ValueError(
            "config.yaml中没有配置phases。"
        )

    phase_output_dirs = [
        Path(phase["output_dir"])
        for phase in phases
    ]

    requested_phase = str(
        getattr(args, "phase", "all")
    ).strip().lower()
    active_phase_pairs = [
        (phase, output_dir)
        for phase, output_dir in zip(phases, phase_output_dirs)
        if requested_phase == "all"
        or str(phase.get("name", "")).strip().lower() == requested_phase
    ]
    if not active_phase_pairs:
        raise ValueError(
            f"Requested phase '{requested_phase}' is not configured."
        )

    combined_output_value = (
        base_config.get(
            "combined_output_path"
        )
    )

    combined_output_path = (
        Path(combined_output_value)
        if combined_output_value
        else None
    )

    # Easy和Hard的所有worker共用一把锁。
    rebuild_lock = threading.RLock()

    # 只重建，不加载数据，不调用模型。
    if args.rebuild_only:
        with rebuild_lock:
            for output_dir in (
                phase_output_dirs
            ):
                ensure_output_dirs(
                    output_dir
                )

                rebuild_attack_dataset(
                    output_dir
                )

                log_event(
                    "REBUILD ONLY phase done "
                    f"dataset="
                    f"{output_dir / 'attack_dataset.jsonl'}",
                    level="quiet",
                    context={
                        "config": base_config
                    },
                )

            if combined_output_path:
                rebuild_combined_attack_dataset(
                    phase_output_dirs=(
                        phase_output_dirs
                    ),
                    combined_output_path=(
                        combined_output_path
                    ),
                )

                log_event(
                    "REBUILD ONLY combined done "
                    f"dataset="
                    f"{combined_output_path}",
                    level="quiet",
                    context={
                        "config": base_config
                    },
                )

        return

    all_phases_completed = False

    try:
        all_rows = load_jsonl(
            Path(
                base_config[
                    "dataset_path"
                ]
            )
        )

        # # 两个阶段共用模型客户端。
        # agent = build_agent(
        #     base_config
        # )
        construct_agent = build_agent(
            base_config,
            model_key="model",
        )

        validator_agent = build_agent(
            base_config,
            model_key="validator_model",
        )

        # if base_config.get(
        #     "runner",
        #     {},
        # ).get(
        #     "preflight",
        #     True,
        # ):
        #     log_event(
        #         "PREFLIGHT start",
        #         context={
        #             "config": base_config
        #         },
        #     )

        #     try:
        #         agent.preflight()

        #     except Exception as exc:
        #         log_event(
        #             "PREFLIGHT fatal "
        #             f"type="
        #             f"{type(exc).__name__} "
        #             f"reason="
        #             f"{short_reason(exc)}",
        #             level="quiet",
        #             context={
        #                 "config": base_config
        #             },
        #         )
        #         raise

        #     log_event(
        #         "PREFLIGHT ok",
        #         context={
        #             "config": base_config
        #         },
        #     )

        if base_config.get(
            "runner",
            {},
        ).get(
            "preflight",
            True,
        ):
            preflight_agents = [
                (
                    "construct",
                    construct_agent,
                    "model",
                ),
                (
                    "validator",
                    validator_agent,
                    "validator_model",
                ),
            ]

            for (
                role,
                current_agent,
                model_key,
            ) in preflight_agents:
                model_name = str(
                    base_config.get(
                        model_key,
                        {},
                    ).get(
                        "name",
                        "",
                    )
                )

                log_event(
                    "PREFLIGHT start "
                    f"role={role} "
                    f"model={model_name}",
                    context={
                        "config": base_config
                    },
                )

                try:
                    current_agent.preflight()

                except Exception as exc:
                    log_event(
                        "PREFLIGHT fatal "
                        f"role={role} "
                        f"model={model_name} "
                        f"type={type(exc).__name__} "
                        f"reason={short_reason(exc)}",
                        level="quiet",
                        context={
                            "config": base_config
                        },
                    )
                    raise

                log_event(
                    "PREFLIGHT ok "
                    f"role={role} "
                    f"model={model_name}",
                    context={
                        "config": base_config
                    },
                )

        log_event(
            f"RUN MODE phase={requested_phase} "
            f"active_phases={[str(item[0].get('name', '')) for item in active_phase_pairs]}",
            level="quiet",
            context={"config": base_config},
        )

        # Strictly sequential across the selected phases.
        for phase, output_dir in active_phase_pairs:
            phase_name = str(
                phase.get(
                    "name",
                    "",
                )
            ).strip()

            if not phase_name:
                raise ValueError(
                    "Each phase must "
                    "have a name."
                )

            phase_config = (
                build_phase_config(
                    base_config=base_config,
                    phase=phase,
                    args=args,
                )
            )

            rows = rows_to_limit(
                all_rows,
                phase_config.get(
                    "runner",
                    {},
                ).get("limit"),
            )

            # run_phase(
            #     phase_name=phase_name,
            #     rows=rows,
            #     project_root=project_root,
            #     config=phase_config,
            #     output_dir=output_dir,
            #     agent=agent,
            #     rebuild_lock=rebuild_lock,
            #     phase_output_dirs=(
            #         phase_output_dirs
            #     ),
            #     combined_output_path=(
            #         combined_output_path
            #     ),
            # )

            run_phase(
                phase_name=phase_name,
                rows=rows,
                project_root=project_root,
                config=phase_config,
                output_dir=output_dir,
                construct_agent=construct_agent,
                validator_agent=validator_agent,
                rebuild_lock=rebuild_lock,
                phase_output_dirs=(
                    phase_output_dirs
                ),
                combined_output_path=(
                    combined_output_path
                ),
            )

        all_phases_completed = True

    except KeyboardInterrupt as exc:
        log_event(
            "ALL PHASES INTERRUPTED "
            "type=KeyboardInterrupt "
            f"reason={short_reason(exc)}",
            level="quiet",
            context={
                "config": base_config
            },
        )
        raise

    except Exception as exc:
        log_event(
            "ALL PHASES FATAL "
            f"type={type(exc).__name__} "
            f"reason={short_reason(exc)}",
            level="quiet",
            context={
                "config": base_config
            },
        )
        raise

    finally:
        # 无论Easy或Hard在哪一步异常，
        # 都重建所有已存在的阶段结果。
        try:
            with rebuild_lock:
                for output_dir in (
                    phase_output_dirs
                ):
                    ensure_output_dirs(
                        output_dir
                    )

                    rebuild_attack_dataset(
                        output_dir
                    )

                if combined_output_path:
                    rebuild_combined_attack_dataset(
                        phase_output_dirs=(
                            phase_output_dirs
                        ),
                        combined_output_path=(
                            combined_output_path
                        ),
                    )

            log_event(
                "ALL PHASES FINAL REBUILD done",
                level="quiet",
                context={
                    "config": base_config
                },
            )

        except Exception as rebuild_exc:
            log_event(
                "ALL PHASES FINAL REBUILD "
                "failed "
                f"type="
                f"{type(rebuild_exc).__name__} "
                f"reason="
                f"{short_reason(rebuild_exc)}",
                level="quiet",
                context={
                    "config": base_config
                },
            )

    if all_phases_completed:
        log_event(
            "ALL PHASES DONE "
            f"combined="
            f"{combined_output_path}",
            level="quiet",
            context={
                "config": base_config
            },
        )

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def directory_file_state(
    directory: Path,
) -> Dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(
            directory.iterdir()
        )
        if path.is_file()
    }


def apply_question_policy(
    row: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    original_question = str(
        row.get("question", "")
    )

    if operator.name in (
        QUESTION_MUST_BE_UNCHANGED
    ):
        payload["new_question"] = (
            original_question
        )

    if operator.name == "file_missing":
        payload["file_edit_required"] = (
            False
        )

    if operator.name == "field_missing":
        required_edit = str(
            selection.get(
                "required_edit",
                "",
            )
        )

        if required_edit == "modify_file":
            payload["new_question"] = (
                original_question
            )

        if required_edit == "question_only":
            payload["file_edit_required"] = (
                False
            )


def static_invariant_errors(
    row: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    payload: Dict[str, Any],
    workspace: AttackWorkspace,
) -> List[str]:
    errors: List[str] = []

    original_question = str(
        row.get("question", "")
    )
    new_question = str(
        payload.get("new_question", "")
    )

    original_state = directory_file_state(
        workspace.original
    )
    final_state = directory_file_state(
        workspace.final
    )

    original_names = set(original_state)
    final_names = set(final_state)

    if operator.name in (
        QUESTION_MUST_BE_UNCHANGED
    ):
        if new_question != original_question:
            errors.append(
                "question must remain exactly unchanged"
            )

    if operator.name == "file_missing":
        if not final_names < original_names:
            errors.append(
                "file_missing requires a strict subset "
                "of original files"
            )

        for name in final_names:
            if (
                final_state[name]
                != original_state[name]
            ):
                errors.append(
                    f"retained file was modified: {name}"
                )

    elif operator.name == "field_missing":
        required_edit = str(
            selection.get(
                "required_edit",
                "",
            )
        )

        if required_edit == "question_only":
            if new_question == original_question:
                errors.append(
                    "question_only did not modify question"
                )

            if final_state != original_state:
                errors.append(
                    "question_only modified files"
                )

        elif required_edit == "modify_file":
            if new_question != original_question:
                errors.append(
                    "modify_file changed question"
                )

            if final_names != original_names:
                errors.append(
                    "field_missing modify_file changed "
                    "file package"
                )

            if final_state == original_state:
                errors.append(
                    "field_missing modify_file did not "
                    "modify any file"
                )

    else:
        if final_names != original_names:
            errors.append(
                f"{operator.name} changed file package"
            )

        if final_state == original_state:
            errors.append(
                f"{operator.name} did not modify files"
            )

    return errors

def build_phase_config(
    base_config: Dict[str, Any],
    phase: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    phase_config = dict(base_config)

    phase_config["output_dir"] = phase[
        "output_dir"
    ]

    phase_config["runner"] = {
        **base_config.get("runner", {}),
        **phase.get("runner", {}),
    }

    phase_config["attacks"] = dict(
        phase.get("attacks", {})
    )

    # 命令行参数同时覆盖Easy和Hard。
    if args.limit is not None:
        phase_config["runner"]["limit"] = (
            args.limit
        )

    if args.num_workers is not None:
        phase_config["runner"][
            "num_workers"
        ] = args.num_workers

    return phase_config


def run_phase(
    phase_name: str,
    rows: List[Dict[str, Any]],
    project_root: Path,
    config: Dict[str, Any],
    output_dir: Path,
    construct_agent: OpenAIWorkspaceAgent,
    validator_agent: OpenAIWorkspaceAgent,
    rebuild_lock: threading.RLock,
    phase_output_dirs: List[Path],
    combined_output_path: Optional[Path],
) -> None:
    ensure_output_dirs(
        output_dir
    )

    runner_config = config.get(
        "runner",
        {},
    )

    operators = get_enabled_operators(
        config.get(
            "attacks",
            {},
        ).get(
            "enabled",
            [],
        )
    )

    resume_summary = summarize_resume(
        rows,
        output_dir,
    )

    context = {
        "project_root": project_root,
        "config": config,
        "output_dir": output_dir,
        "data_root": Path(
            config["data_root"]
        ),

        # 当前阶段启用的攻击算子
        "operators": operators,

        # 攻击选择和攻击构造模型
        "agent": construct_agent,

        # 独立验证模型
        "validator_agent": validator_agent,

        "phase_name": phase_name,

        # 增量重建所需信息
        "dataset_rebuild_lock": (
            rebuild_lock
        ),
        "phase_output_dirs": (
            phase_output_dirs
        ),
        "combined_output_path": (
            combined_output_path
        ),
        "attack_timeout_sec": float(
            runner_config.get("attack_timeout_sec", 900)
        ),
        "selection_timeout_sec": float(
            runner_config.get("selection_timeout_sec", 300)
        ),
    }

    # accepted/*.json is the source of truth. Rebuild once before workers
    # start, then each newly accepted record can be appended in O(1).
    with rebuild_lock:
        rebuild_attack_dataset(output_dir)
        if combined_output_path:
            rebuild_combined_attack_dataset(
                phase_output_dirs=phase_output_dirs,
                combined_output_path=combined_output_path,
            )

    log_event(
        "PHASE START "
        f"name={phase_name} "
        f"loaded="
        f"{resume_summary['loaded']} "
        f"completed="
        f"{resume_summary['completed']} "
        f"remaining="
        f"{resume_summary['remaining']} "
        f"attacks="
        f"{[op.name for op in operators]} "
        f"num_workers="
        f"{runner_config.get('num_workers', 4)} "
        f"output_dir={output_dir}",
        level="quiet",
        context=context,
    )

    runner = Runner(
        num_workers=runner_config.get(
            "num_workers",
            4,
        ),
        stall_timeout_sec=(
            runner_config.get(
                "stall_timeout_sec",
                600,
            )
        ),
    )

    try:
        runner.run(
            rows,
            lambda row: process_row(
                row,
                context,
            ),
            on_stall=lambda pending_ids, elapsed: log_event(
                "PHASE STALL diagnostic "
                f"name={phase_name} "
                f"elapsed={elapsed:.1f}s "
                f"pending_count={len(pending_ids)} "
                f"pending_sample_ids={pending_ids[:20]}",
                level="quiet",
                context=context,
            ),
        )

    except KeyboardInterrupt as exc:
        log_event(
            "PHASE INTERRUPTED "
            f"name={phase_name} "
            "type=KeyboardInterrupt "
            f"reason={short_reason(exc)}",
            level="quiet",
            context=context,
        )
        raise

    except Exception as exc:
        log_event(
            "PHASE FATAL "
            f"name={phase_name} "
            f"type={type(exc).__name__} "
            f"reason={short_reason(exc)}",
            level="quiet",
            context=context,
        )
        raise

    finally:
        # 当前phase无论如何结束，
        # 都重建当前phase和combined。
        try:
            with rebuild_lock:
                rebuild_attack_dataset(
                    output_dir
                )

                if combined_output_path:
                    rebuild_combined_attack_dataset(
                        phase_output_dirs=(
                            phase_output_dirs
                        ),
                        combined_output_path=(
                            combined_output_path
                        ),
                    )

            log_event(
                "PHASE FINAL REBUILD done "
                f"name={phase_name} "
                f"dataset="
                f"{output_dir / 'attack_dataset.jsonl'}",
                level="quiet",
                context=context,
            )

        except Exception as rebuild_exc:
            log_event(
                "PHASE FINAL REBUILD failed "
                f"name={phase_name} "
                f"type="
                f"{type(rebuild_exc).__name__} "
                f"reason="
                f"{short_reason(rebuild_exc)}",
                level="quiet",
                context=context,
            )

    log_event(
        "PHASE DONE "
        f"name={phase_name} "
        f"dataset="
        f"{output_dir / 'attack_dataset.jsonl'}",
        level="quiet",
        context=context,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct table reason "
            "reliability attack samples."
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help=(
            "Do not call the LLM. Rebuild "
            "all phase datasets and the "
            "combined dataset from accepted files."
        ),
    )

    parser.add_argument(
        "--phase",
        choices=["all", "easy", "hard"],
        default="all",
        help="Run all phases or only easy/hard.",
    )

    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for config.yaml. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(
    config: Dict[str, Any],
    model_key: str = "model",
) -> OpenAIWorkspaceAgent:
    model = config.get(model_key)

    if not isinstance(model, dict):
        raise RuntimeError(
            f"Missing required model configuration: {model_key}"
        )

    if not model.get("name"):
        raise RuntimeError(
            f"Missing model name: {model_key}.name"
        )

    return OpenAIWorkspaceAgent(
        api_key=expand_env(model.get("api_key")),
        base_url=expand_env(model.get("base_url")),
        model_name=str(model["name"]),
        max_rounds=model.get("max_rounds", 12),
        temperature=model.get("temperature", 0.0),
        max_tokens=model.get("max_tokens", 8192),
        request_timeout=model.get(
            "request_timeout",
            120,
        ),
        max_retries=model.get("max_retries", 2),
        tool_timeout_sec=config.get(
            "runner",
            {},
        ).get(
            "tool_timeout_sec",
            300,
        ),
    )

def ensure_output_dirs(output_dir: Path) -> None:
    for name in ["accepted", "rejected", "candidates", "logs", "workspaces"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def log_event(message: str, level: str = "info", context: Optional[Dict[str, Any]] = None) -> None:
    config = (context or {}).get("config", {}) if context else {}
    if not should_log(level, config):
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"[{timestamp}] {message}")


def should_log(level: str, config: Dict[str, Any]) -> bool:
    order = {"quiet": 0, "info": 1, "verbose": 2}
    configured = str(config.get("runner", {}).get("log_level", "info")).strip().lower()
    if configured not in order:
        configured = "info"
    requested = order.get(level, order["info"])
    return requested <= order[configured]


def summarize_resume(rows: List[Dict[str, Any]], output_dir: Path) -> Dict[str, int]:
    completed = 0
    logs_dir = output_dir / "logs"
    for row in rows:
        sample_id = str(row.get("id", "unknown"))
        log_path = logs_dir / f"{sample_id}.json"
        if not log_path.exists():
            continue
        try:
            existing_log = json.loads(log_path.read_text(encoding="utf-8"))
            if existing_log.get("completed") is True:
                completed += 1
        except Exception:
            continue
    return {
        "loaded": len(rows),
        "completed": completed,
        "remaining": len(rows) - completed,
    }


def short_reason(value: Any, limit: int = 240) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def process_row(
    row: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    sample_id = str(row.get("id", "unknown"))
    output_dir: Path = context["output_dir"]
    log_path = output_dir / "logs" / f"{sample_id}.json"

    if log_path.exists():
        try:
            existing_log = json.loads(log_path.read_text(encoding="utf-8"))
            if existing_log.get("completed") is True:
                log_event(
                    f"SAMPLE {sample_id} skipped completed_log",
                    context=context,
                )
                return {"id": sample_id, "status": "skipped_existing_log"}
        except Exception:
            pass

    row_log: Dict[str, Any] = {
        "id": sample_id,
        "phase": context.get("phase_name"),
        "selected": [],
        "selection_rejected": [],
        "results": [],
        "completed": False,
    }

    try:
        source_files = resolve_input_files(row, context["data_root"])
        source_profile = profile_files(source_files, virtual_root="/mnt/data")
        selected, selection_rejected = select_attacks(
            row,
            source_profile,
            context,
        )
        row_log["selected"] = selected
        row_log["selection_rejected"] = selection_rejected
        write_json(log_path, row_log)
    except AgentFatalError as exc:
        row_log["fatal_error"] = exception_payload(exc)
        write_json(log_path, row_log)
        log_event(
            f"SAMPLE {sample_id} fatal {short_reason(exc)}",
            level="quiet",
            context=context,
        )
        raise
    except (AgentTransientError, AgentOperationTimeout) as exc:
        row_log["retryable_error"] = exception_payload(exc)
        row_log["completed"] = False
        write_json(log_path, row_log)
        log_event(
            f"SAMPLE {sample_id} retryable select_error "
            f"{type(exc).__name__}: {short_reason(exc)}",
            level="quiet",
            context=context,
        )
        return {
            "id": sample_id,
            "status": "retryable_select_error",
            "error": str(exc),
        }
    except AgentOutputError as exc:
        row_log["error"] = exception_payload(exc)
        row_log["completed"] = True
        write_json(log_path, row_log)
        log_event(
            f"SAMPLE {sample_id} select_failed {short_reason(exc)}",
            context=context,
        )
        return {
            "id": sample_id,
            "status": "select_failed",
            "error": str(exc),
        }
    except Exception as exc:
        row_log["fatal_error"] = exception_payload(exc)
        write_json(log_path, row_log)
        log_event(
            f"SAMPLE {sample_id} unexpected_fatal "
            f"{type(exc).__name__}: {short_reason(exc)}",
            level="quiet",
            context=context,
        )
        raise

    if not selected:
        log_event(f"SAMPLE {sample_id} no_attack", context=context)

    has_retryable_error = False

    for idx, attack in enumerate(selected, start=1):
        attack_type = str(attack.get("attack_type", ""))
        operator = next(
            (op for op in context["operators"] if op.name == attack_type),
            None,
        )
        if operator is None:
            result = {
                "attack_type": attack_type,
                "status": "unknown_operator",
            }
            row_log["results"].append(result)
            write_json(log_path, row_log)
            continue

        new_id = f"{sample_id}__{attack_type}__{idx:03d}"

        try:
            result = construct_and_validate(
                row=row,
                source_files=source_files,
                source_profile=source_profile,
                operator=operator,
                new_id=new_id,
                context=context,
                selection=attack,
            )
        except AttackTimeoutError as exc:
            has_retryable_error = True
            result = {
                "id": new_id,
                "source_id": sample_id,
                "attack_type": operator.name,
                "status": "retryable_error",
                "stage": "attack_timeout",
                "timeout_stage": exc.stage,
                "elapsed_sec": round(exc.elapsed_sec, 2),
                "timeout_sec": exc.timeout_sec,
                "reason": (
                    f"single attack exceeded {exc.timeout_sec:.0f} seconds; "
                    f"stopped at stage={exc.stage}"
                ),
                "completed": False,
            }
            log_event(
                f"ATTACK {new_id} retryable timeout "
                f"stage={exc.stage} elapsed={exc.elapsed_sec:.1f}s "
                f"limit={exc.timeout_sec:.1f}s",
                level="quiet",
                context=context,
            )
        except AgentFatalError as exc:
            result = {
                "id": new_id,
                "status": "fatal_error",
                "error": exception_payload(exc),
            }
            row_log["results"].append(result)
            write_json(log_path, row_log)
            log_event(
                f"ATTACK {new_id} fatal {short_reason(exc)}",
                level="quiet",
                context=context,
            )
            raise
        except AgentTransientError as exc:
            has_retryable_error = True
            result = {
                "id": new_id,
                "source_id": sample_id,
                "attack_type": operator.name,
                "status": "retryable_error",
                "error": exception_payload(exc),
                "reason": str(exc),
            }
            log_event(
                f"ATTACK {new_id} retryable_error "
                f"{type(exc).__name__}: {short_reason(exc)}",
                level="quiet",
                context=context,
            )
        except AgentOutputError as exc:
            result = write_exception_rejected(
                new_id,
                row,
                operator,
                context,
                exc,
            )
            log_event(
                f"ATTACK {new_id} rejected "
                f"{short_reason(result.get('reason'))}",
                context=context,
            )
        except Exception as exc:
            result = {
                "id": new_id,
                "status": "fatal_error",
                "error": exception_payload(exc),
            }
            row_log["results"].append(result)
            write_json(log_path, row_log)
            log_event(
                f"ATTACK {new_id} unexpected_fatal "
                f"{type(exc).__name__}: {short_reason(exc)}",
                level="quiet",
                context=context,
            )
            raise

        row_log["results"].append(result)
        write_json(log_path, row_log)

    row_log["completed"] = not has_retryable_error
    write_json(log_path, row_log)

    counts: Dict[str, int] = {}
    for item in row_log["results"]:
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    status_text = " ".join(
        f"{key}={value}" for key, value in sorted(counts.items())
    ) or "results=0"

    log_event(
        f"SAMPLE {sample_id} done completed={row_log['completed']} "
        f"{status_text}",
        context=context,
    )
    return {
        "id": sample_id,
        "status": "processed" if row_log["completed"] else "retryable",
        "num_results": len(row_log["results"]),
    }

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def write_json_atomic(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.name
        + f".{threading.get_ident()}.tmp"
    )

    temp_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp_path,
        path,
    )

def exception_payload(exc: Exception) -> Dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def write_exception_rejected(
    new_id: str,
    row: Dict[str, Any],
    operator: Operator,
    context: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    output_dir: Path = context["output_dir"]
    payload = {"exception": exception_payload(exc)}
    rejected = build_rejected(new_id, row, operator, payload, f"exception: {type(exc).__name__}: {exc}")
    rejected_path = output_dir / "rejected" / f"{new_id}.json"
    write_json(rejected_path, rejected)
    return {
        "id": new_id,
        "status": "rejected",
        "reason": rejected["rejected_reason"],
    }


def file_missing_static_ineligible(
    attack_type: str,
    profile: Dict[str, Any],
) -> bool:
    """file_missing cannot be constructed from a one-file package."""
    return (
        attack_type == "file_missing"
        and int(profile.get("file_count", 0)) < 2
    )


def select_attacks(
    row: Dict[str, Any],
    profile: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    config = context["config"]

    sample_id = str(
        row.get("id", "")
    )

    enabled_names = [
        op.name
        for op in context["operators"]
    ]

    enabled = set(enabled_names)

    attack_config = config.get(
        "attacks",
        {},
    )

    default_min_confidence = float(
        attack_config.get(
            "min_confidence",
            0.65,
        )
    )

    configured_thresholds = (
        attack_config.get(
            "min_confidence_by_attack",
            {},
        )
        or {}
    )

    # 为当前阶段每种攻击生成最终阈值。
    min_confidence_by_attack: Dict[
        str,
        float,
    ] = {}

    for attack_type in enabled_names:
        raw_threshold = (
            configured_thresholds.get(
                attack_type,
                default_min_confidence,
            )
        )

        try:
            threshold = float(
                raw_threshold
            )
        except (
            TypeError,
            ValueError,
        ):
            threshold = (
                default_min_confidence
            )

        min_confidence_by_attack[
            attack_type
        ] = threshold

    max_attacks = int(
        attack_config.get(
            "max_attacks_per_sample",
            len(enabled_names),
        )
    )

    prompt_path = (
        context["project_root"]
        / "prompts"
        / "reliability"
        / "select_attack.md"
    )

    prompt = render_template(
        prompt_path.read_text(
            encoding="utf-8"
        ),
        {
            "phase_name": str(
                context.get(
                    "phase_name",
                    "",
                )
            ),
            "enabled_attacks_json": (
                compact_json(
                    enabled_names
                )
            ),
            "min_confidence_by_attack_json": (
                compact_json(
                    min_confidence_by_attack
                )
            ),
            # 保留旧Prompt兼容字段。
            "min_confidence": str(
                default_min_confidence
            ),
            "sample_id": sample_id,
            "question": str(
                row.get(
                    "question",
                    "",
                )
            ),
            "reference": str(
                row.get(
                    "reference",
                    "",
                )
            ),
            "file_profile_json": (
                compact_json(profile)
            ),
        },
    )

    selection_deadline = time.monotonic() + float(
        context.get("selection_timeout_sec", 300)
    )

    payload, _history = (
        context["agent"].complete_json(
            prompt,
            workspace=None,
            allow_tools=False,
            stage="select",
            deadline_monotonic=selection_deadline,
        )
    )

    if not isinstance(payload, dict):
        raise AgentOutputError(
            "select_attack returned "
            "a non-object payload"
        )

    raw_eligible = (
        payload.get(
            "eligible_attacks",
            [],
        )
        or []
    )

    raw_rejected = (
        payload.get(
            "rejected_attacks",
            [],
        )
        or []
    )

    if not isinstance(
        raw_eligible,
        list,
    ):
        raw_eligible = []

    if not isinstance(
        raw_rejected,
        list,
    ):
        raw_rejected = []

    # -------------------------------------------------
    # 第一步：eligible按attack_type去重。
    # 同一类型只保留置信度最高的一条。
    # -------------------------------------------------
    best_eligible: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for raw_item in raw_eligible:
        if not isinstance(
            raw_item,
            dict,
        ):
            continue

        attack_type = str(
            raw_item.get(
                "attack_type",
                "",
            )
        ).strip()

        if attack_type not in enabled:
            continue

        try:
            confidence = float(
                raw_item.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        item = dict(raw_item)

        item["attack_type"] = (
            attack_type
        )
        item["confidence"] = confidence

        current = best_eligible.get(
            attack_type
        )

        if current is None:
            best_eligible[
                attack_type
            ] = item
            continue

        try:
            current_confidence = float(
                current.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            current_confidence = 0.0

        if (
            confidence
            > current_confidence
        ):
            best_eligible[
                attack_type
            ] = item

    eligible = list(
        best_eligible.values()
    )

    eligible_types = set(
        best_eligible
    )

    # -------------------------------------------------
    # 第二步：rejected也按attack_type去重。
    # 如果同一类型同时出现在eligible和rejected，
    # eligible优先，避免一类攻击同时被接受和拒绝。
    # -------------------------------------------------
    best_rejected: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for raw_item in raw_rejected:
        if not isinstance(
            raw_item,
            dict,
        ):
            continue

        attack_type = str(
            raw_item.get(
                "attack_type",
                "",
            )
        ).strip()

        if attack_type not in enabled:
            continue

        if attack_type in eligible_types:
            continue

        if attack_type in best_rejected:
            # 只保留模型给出的第一条明确拒绝原因。
            continue

        best_rejected[
            attack_type
        ] = dict(raw_item)

    model_rejected = list(
        best_rejected.values()
    )

    filtered: List[
        Dict[str, Any]
    ] = []

    selection_rejected: List[
        Dict[str, Any]
    ] = []

    reported_types = set()

    # -------------------------------------------------
    # 第三步：记录模型明确拒绝的攻击。
    # -------------------------------------------------
    for item in model_rejected:
        attack_type = str(
            item.get(
                "attack_type",
                "",
            )
        ).strip()

        if attack_type not in enabled:
            continue

        reported_types.add(
            attack_type
        )

        selection_rejected.append(
            {
                "attack_type": (
                    attack_type
                ),
                "stage": (
                    "model_rejected"
                ),
                "confidence": (
                    item.get(
                        "confidence"
                    )
                ),
                "reason": (
                    item.get(
                        "reason"
                    )
                    or (
                        "模型判断不适合"
                        "该攻击"
                    )
                ),
            }
        )

    # -------------------------------------------------
    # 第四步：使用每类独立阈值过滤eligible。
    # -------------------------------------------------
    for item in eligible:
        attack_type = str(
            item.get(
                "attack_type",
                "",
            )
        ).strip()

        if attack_type not in enabled:
            continue

        reported_types.add(
            attack_type
        )

        if file_missing_static_ineligible(
            attack_type,
            profile,
        ):
            selection_rejected.append(
                {
                    "attack_type": "file_missing",
                    "stage": "static_ineligible",
                    "confidence": item.get("confidence"),
                    "reason": (
                        "file_missing requires at least two "
                        "source files; profile.file_count < 2"
                    ),
                }
            )
            continue

        try:
            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        item["confidence"] = (
            confidence
        )

        attack_min_confidence = float(
            min_confidence_by_attack.get(
                attack_type,
                default_min_confidence,
            )
        )

        if (
            confidence
            < attack_min_confidence
        ):
            selection_rejected.append(
                {
                    "attack_type": (
                        attack_type
                    ),
                    "stage": (
                        "below_confidence_threshold"
                    ),
                    "confidence": confidence,
                    "reason": (
                        f"置信度{confidence}"
                        f"低于{attack_type}"
                        f"阈值"
                        f"{attack_min_confidence}"
                    ),
                }
            )
            continue

        # 确保selection至少有基本结构。
        if not isinstance(
            item.get("target"),
            dict,
        ):
            item["target"] = {}

        item[
            "applied_min_confidence"
        ] = attack_min_confidence

        filtered.append(item)

    # -------------------------------------------------
    # 第五步：模型完全遗漏的类型写入JSON日志。
    # -------------------------------------------------
    for attack_type in (
        enabled - reported_types
    ):
        selection_rejected.append(
            {
                "attack_type": (
                    attack_type
                ),
                "stage": (
                    "model_omitted"
                ),
                "confidence": None,
                "reason": (
                    "模型没有对本阶段"
                    "启用的攻击类型"
                    "给出判断"
                ),
            }
        )

    # -------------------------------------------------
    # 第六步：按operator优先级排序。
    # 同优先级时优先选择置信度更高的攻击。
    # -------------------------------------------------
    priority = {
        op.name: op.priority
        for op in context["operators"]
    }

    filtered.sort(
        key=lambda item: (
            priority.get(
                item.get(
                    "attack_type"
                ),
                999,
            ),
            -float(
                item.get(
                    "confidence",
                    0.0,
                )
            ),
        )
    )

    selected = filtered[
        :max_attacks
    ]

    # 超过单样本最大攻击数量的候选写入拒绝日志。
    for item in filtered[
        max_attacks:
    ]:
        selection_rejected.append(
            {
                "attack_type": (
                    item.get(
                        "attack_type"
                    )
                ),
                "stage": (
                    "max_attacks_limit"
                ),
                "confidence": (
                    item.get(
                        "confidence"
                    )
                ),
                "reason": (
                    "攻击适用，但超过"
                    "max_attacks_per_sample"
                    f"={max_attacks}"
                ),
            }
        )

    # 选择详情只写logs/QA_x.json，
    # 不通过log_event输出到控制台。
    return (
        selected,
        selection_rejected,
    )

def resolve_attack_runtime_limits(
    context: Dict[str, Any],
    attack_type: str,
) -> Dict[str, Optional[int]]:
    """Resolve per-attack limits without mutating shared Agent instances."""
    runner_config = context.get("config", {}).get("runner", {})
    per_attack = runner_config.get("attack_limits", {}).get(
        attack_type,
        {},
    )

    return {
        "attack_timeout_sec": int(
            per_attack.get(
                "attack_timeout_sec",
                context.get("attack_timeout_sec", 900),
            )
        ),
        "construct_max_rounds": (
            int(per_attack["construct_max_rounds"])
            if per_attack.get("construct_max_rounds") is not None
            else None
        ),
        "validate_max_rounds": (
            int(per_attack["validate_max_rounds"])
            if per_attack.get("validate_max_rounds") is not None
            else None
        ),
    }


def normalize_constructed_file_package(
    operator: Operator,
    construct_payload: Dict[str, Any],
    workspace: AttackWorkspace,
) -> str:
    """
    Normalize the final package and return its canonical input_file value.

    For file_missing, the final directory is the source of truth after the
    requested subset has been applied. This prevents a stale model-generated
    input_file field from referring to the file that was intentionally removed.
    """
    requested_output = normalize_file_names(
        construct_payload.get("output_files") or []
    )

    if operator.name == "file_missing" and not requested_output:
        requested_output = normalize_file_names(
            construct_payload.get("input_file") or ""
        )

    workspace.normalize_final_files(requested_output)
    final_names = workspace.final_file_names()

    if operator.name == "file_missing":
        canonical_input = "\n".join(final_names)
        construct_payload["output_files"] = final_names
        construct_payload["input_file"] = canonical_input
        construct_payload["file_edit_required"] = False
        return canonical_input

    canonical_input = normalize_input_file(
        construct_payload.get("input_file")
        or "\n".join(final_names)
    )
    construct_payload["output_files"] = final_names
    construct_payload["input_file"] = canonical_input
    return canonical_input


def construct_and_validate(
    row: Dict[str, Any],
    source_files: List[Path],
    source_profile: Dict[str, Any],
    operator: Operator,
    new_id: str,
    context: Dict[str, Any],
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir: Path = context["output_dir"]

    accepted_path = (
        output_dir
        / "accepted"
        / f"{new_id}.json"
    )
    rejected_path = (
        output_dir
        / "rejected"
        / f"{new_id}.json"
    )
    candidate_path = (
        output_dir
        / "candidates"
        / f"{new_id}.json"
    )

    if accepted_path.exists() or rejected_path.exists():
        log_event(
            f"ATTACK {new_id} skipped existing_result",
            context=context,
        )
        return {
            "id": new_id,
            "status": "skipped_existing_result",
        }

    attack_limits = resolve_attack_runtime_limits(
        context,
        operator.name,
    )
    timeout_sec = float(
        attack_limits["attack_timeout_sec"]
        or context.get("attack_timeout_sec", 900)
    )
    started_at = time.monotonic()
    deadline = started_at + timeout_sec

    runtime: Dict[str, Any] = {
        "attack_id": new_id,
        "stage": "initializing",
        "started_at": started_at,
        "elapsed_sec": 0.0,
        "timeout_sec": timeout_sec,
    }

    def set_stage(stage: str) -> None:
        runtime["stage"] = stage
        runtime["elapsed_sec"] = (
            time.monotonic() - started_at
        )
        if time.monotonic() >= deadline:
            raise AttackTimeoutError(
                attack_id=new_id,
                stage=stage,
                elapsed_sec=runtime["elapsed_sec"],
                timeout_sec=timeout_sec,
            )

    try:
        log_event(
            f"ATTACK {new_id} construct_start",
            level="verbose",
            context=context,
        )

        set_stage("workspace_prepare")

        workspace = AttackWorkspace(
            output_dir,
            new_id,
            str(row.get("id", "")),
            operator.name,
        )
        workspace.prepare(source_files)

        set_stage("construct_profile")
        # The source was already profiled before selection. Reuse it instead
        # of reopening the same Excel/CSV from the workspace copy.
        construct_profile = source_profile

        set_stage("construct_prompt")

        construct_prompt = render_construct_prompt(
            row=row,
            profile=construct_profile,
            operator=operator,
            workspace=workspace,
            context=context,
            selection=selection,
        )

        set_stage("construct_model")

        construct_payload, construct_history = (
            context["agent"].complete_json(
                construct_prompt,
                workspace=workspace,
                stage="construct",
                allow_tools=True,
                deadline_monotonic=deadline,
                max_rounds=attack_limits[
                    "construct_max_rounds"
                ],
            )
        )

        set_stage("write_candidate")

        candidate_path.write_text(
            json.dumps(
                {
                    "id": new_id,
                    "source_id": row.get("id"),
                    "attack_type": operator.name,
                    "construct": construct_payload,
                    "construct_history": construct_history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if construct_payload.get("status") != "constructed":
            reason = construct_payload.get("reject_reason")

            rejected = build_rejected(
                new_id,
                row,
                operator,
                construct_payload,
                "construct_rejected",
            )

            rejected_path.write_text(
                json.dumps(
                    rejected,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            log_event(
                f"ATTACK {new_id} rejected "
                f"{short_reason(reason)}",
                context=context,
            )

            return {
                "id": new_id,
                "status": "rejected",
                "reason": reason,
            }

        set_stage("normalize_final_files")

        input_file = normalize_constructed_file_package(
            operator=operator,
            construct_payload=construct_payload,
            workspace=workspace,
        )

        set_stage("validate_file_references")

        missing = workspace.validate_file_refs(input_file)

        if missing:
            reason = f"missing final files: {missing}"

            rejected = build_rejected(
                new_id,
                row,
                operator,
                construct_payload,
                reason,
            )

            rejected_path.write_text(
                json.dumps(
                    rejected,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            log_event(
                f"ATTACK {new_id} rejected "
                f"{short_reason(reason)}",
                context=context,
            )

            return {
                "id": new_id,
                "status": "rejected",
                "reason": reason,
            }

        set_stage("write_manifest")

        manifest = workspace.write_manifest(
            input_file=input_file,
            modified=bool(
                construct_payload.get("file_edit_required")
            ),
            edit_summary=str(
                construct_payload.get("edit_summary")
                or construct_payload.get("edit_plan")
                or ""
            ),
            original_files=[
                path.name
                for path in source_files
            ],
        )

        set_stage("validate_profile")

        final_profile = profile_files(
            [
                workspace.final / name
                for name in workspace.final_file_names()
            ],
            virtual_root="/mnt/data",
        )

        set_stage("validate_prompt")

        validate_prompt = render_validate_prompt(
            construct_payload=construct_payload,
            row=row,
            operator=operator,
            selection=selection,
            source_profile=source_profile,
            final_profile=final_profile,
            workspace=workspace,
            context=context,
        )

        set_stage("validate_model")

        validator_agent = (
            context.get("validator_agent")
            or context["agent"]
        )

        validate_payload, validate_history = (
            validator_agent.complete_json(
                validate_prompt,
                workspace=workspace,
                stage="validate",
                allow_tools=True,
                deadline_monotonic=deadline,
                max_rounds=attack_limits[
                    "validate_max_rounds"
                ],
            )
        )

        set_stage("build_record")

        record = build_attack_record(
            new_id=new_id,
            row=row,
            operator=operator,
            construct_payload=construct_payload,
            validate_payload=validate_payload,
            manifest=manifest,
        )

        record["construction"][
            "construct_history"
        ] = construct_history
        record["construction"][
            "validate_history"
        ] = validate_history

        set_stage("static_validation")

        validation_passed = validate_unanswerable_payload(
            validate_payload,
            attack_type=operator.name,
        )

        if validation_passed:
            rebuild_lock = context["dataset_rebuild_lock"]
            phase_output_dirs = context["phase_output_dirs"]
            combined_output_path = context.get(
                "combined_output_path"
            )

            runtime["stage"] = "persist_accepted"

            with rebuild_lock:
                write_json_atomic(
                    accepted_path,
                    record,
                )
                append_attack_record(
                    output_dir / "attack_dataset.jsonl",
                    record,
                )
                if combined_output_path:
                    append_attack_record(
                        combined_output_path,
                        record,
                    )

            log_event(
                f"ATTACK {new_id} accepted",
                context=context,
            )

            return {
                "id": new_id,
                "status": "accepted",
            }

        runtime["stage"] = "persist_rejected"

        record["construction"][
            "validation_status"
        ] = "failed"
        record["construction"][
            "failure_reason"
        ] = validation_failure_reason(
            validate_payload,
            attack_type=operator.name,
        )

        rejected_path.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        reason = record["construction"]["failure_reason"]

        log_event(
            f"ATTACK {new_id} rejected "
            f"{short_reason(reason)}",
            context=context,
        )

        return {
            "id": new_id,
            "status": "rejected",
            "reason": reason,
        }

    except AgentOperationTimeout as exc:
        runtime["elapsed_sec"] = (
            time.monotonic() - started_at
        )
        raise AttackTimeoutError(
            attack_id=new_id,
            stage=str(runtime.get("stage", "unknown")),
            elapsed_sec=runtime["elapsed_sec"],
            timeout_sec=timeout_sec,
        ) from exc


def render_construct_prompt(
    row: Dict[str, Any],
    profile: Dict[str, Any],
    operator: Operator,
    workspace: AttackWorkspace,
    context: Dict[str, Any],
    selection: Dict[str, Any],
) -> str:
    prompt_path = (
        context["project_root"]
        / operator.construct_prompt
    )

    return render_template(
        prompt_path.read_text(
            encoding="utf-8"
        ),
        {
            "phase_name": str(
                context.get(
                    "phase_name",
                    "",
                )
            ),
            "attack_type": operator.name,
            "attack_definition": (
                operator.definition
            ),
            "selection_json": compact_json(
                selection
            ),
            "sample_id": str(
                row.get("id", "")
            ),
            "question": str(
                row.get("question", "")
            ),
            "reference": str(
                row.get("reference", "")
            ),
            "file_profile_json": compact_json(
                profile
            ),
            "virtual_file_list": "\n".join(
                workspace.virtual_files(
                    stage="construct"
                )
            ),
        },
    )

def render_validate_prompt(
    row: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    construct_payload: Dict[str, Any],
    source_profile: Dict[str, Any],
    final_profile: Dict[str, Any],
    workspace: AttackWorkspace,
    context: Dict[str, Any],
) -> str:
    prompt_path = (
        context["project_root"]
        / "prompts"
        / "reliability"
        / "validate_unanswerable.md"
    )

    input_names = [
        part.strip()
        for part in str(
            construct_payload.get(
                "input_file"
            )
            or ""
        ).splitlines()
        if part.strip()
    ]

    final_files = workspace.virtual_files(
        stage="validate",
        file_names=input_names or None,
    )

    original_files = (
        workspace.virtual_original_files()
    )

    return render_template(
        prompt_path.read_text(
            encoding="utf-8"
        ),
        {
            "attack_type": operator.name,
            "attack_definition": (
                operator.definition
            ),
            "selection_json": compact_json(
                selection
            ),
            "source_question": str(
                row.get(
                    "question",
                    "",
                )
            ),
            "new_question": str(
                construct_payload.get(
                    "new_question",
                    "",
                )
            ),
            "expected_answer_json": (
                compact_json(
                    construct_payload.get(
                        "expected_answer",
                        {},
                    )
                )
            ),
            "edit_summary": str(
                construct_payload.get(
                    "edit_summary",
                    "",
                )
            ),
            "original_virtual_file_list": (
                "\n".join(original_files)
            ),
            "virtual_file_list": (
                "\n".join(final_files)
            ),
            "source_file_profile_json": (
                compact_json(
                    source_profile
                )
            ),
            "file_profile_json": (
                compact_json(
                    final_profile
                )
            ),
        },
    )

def build_attack_record(
    new_id: str,
    row: Dict[str, Any],
    operator: Operator,
    construct_payload: Dict[str, Any],
    validate_payload: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": new_id,
        "source_id": row.get("id"),
        "scenario": "table_reason",
        "dimension": operator.dimension,
        "attack_type": operator.name,
        "question": construct_payload.get("new_question"),
        "answer": construct_payload.get("expected_answer"),
        "input_file": manifest.get("input_file"),
        "file_root": manifest.get("file_root"),
        "source_question": row.get("question"),
        "source_reference": row.get("reference"),
        "construction": {
            "edit_required": bool(
                construct_payload.get(
                    "file_edit_required"
                )
            ),
            "edit_summary": (
                construct_payload.get(
                    "edit_summary"
                )
                or construct_payload.get(
                    "edit_plan"
                )
            ),
            "base_attack_components": (
                construct_payload.get(
                    "base_attack_components",
                    [],
                )
            ),
            "reasoning_chain": (
                construct_payload.get(
                    "reasoning_chain",
                    [],
                )
            ),
            "attack_evidence": (
                construct_payload.get(
                    "attack_evidence",
                    {},
                )
            ),
            "hardness_check": (
                construct_payload.get(
                    "hardness_check",
                    {},
                )
            ),
            "validation_status": (
                validate_payload.get(
                    "verdict"
                )
            ),
            "validation": validate_payload,
            "manifest_path": str(
                Path(
                    str(
                        manifest.get(
                            "file_root"
                        )
                    )
                ).parent
                / "manifest.json"
            ),
        },
    }


def build_rejected(
    new_id: str,
    row: Dict[str, Any],
    operator: Operator,
    payload: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "id": new_id,
        "source_id": row.get("id"),
        "scenario": "table_reason",
        "dimension": operator.dimension,
        "attack_type": operator.name,
        "source_question": row.get("question"),
        "source_reference": row.get("reference"),
        "rejected_reason": reason,
        "payload": payload,
    }


def public_attack_record(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    public = dict(record)
    construction = dict(
        public.get("construction", {})
    )
    construction.pop("construct_history", None)
    construction.pop("validate_history", None)
    public["construction"] = construction
    return public


def append_attack_record(
    dataset_path: Path,
    record: Dict[str, Any],
) -> None:
    """Append one accepted record while the caller holds dataset_rebuild_lock."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        public_attack_record(record),
        ensure_ascii=False,
    ) + "\n"
    with dataset_path.open("a", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())

def rebuild_attack_dataset(
    output_dir: Path,
) -> None:
    dataset_path = (
        output_dir
        / "attack_dataset.jsonl"
    )

    temp_path = (
        output_dir
        / (
            "attack_dataset.jsonl."
            f"{threading.get_ident()}.tmp"
        )
    )

    accepted_dir = (
        output_dir
        / "accepted"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as out:
            for path in sorted(
                accepted_dir.glob(
                    "*.json"
                )
            ):
                record = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                public = public_attack_record(record)

                out.write(
                    json.dumps(
                        public,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        os.replace(
            temp_path,
            dataset_path,
        )

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

def rebuild_combined_attack_dataset(
    phase_output_dirs: List[Path],
    combined_output_path: Path,
) -> None:
    combined_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        combined_output_path.with_name(
            combined_output_path.name
            + f".{threading.get_ident()}.tmp"
        )
    )

    seen_ids = set()

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as out:
            for output_dir in (
                phase_output_dirs
            ):
                dataset_path = (
                    output_dir
                    / "attack_dataset.jsonl"
                )

                if not dataset_path.exists():
                    continue

                with dataset_path.open(
                    "r",
                    encoding="utf-8",
                ) as source:
                    for line in source:
                        if not line.strip():
                            continue

                        record = json.loads(
                            line
                        )

                        record_id = str(
                            record.get(
                                "id",
                                "",
                            )
                        )

                        if (
                            record_id
                            in seen_ids
                        ):
                            continue

                        seen_ids.add(
                            record_id
                        )

                        out.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        os.replace(
            temp_path,
            combined_output_path,
        )

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def normalize_file_names(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        raw = [part.strip() for part in values.replace(";", "\n").replace("\uFF1B", "\n").splitlines() if part.strip()]
    else:
        raw = [str(item).strip() for item in values if str(item).strip()]
    return [Path(item).name for item in raw]


def normalize_input_file(value: str) -> str:
    names = normalize_file_names(value)
    return "\n".join(names)


def render_template(text: str, values: Dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


if __name__ == "__main__":
    main()

