from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent import (
    AgentFatalError,
    AgentOutputError,
    CONSTRUCTOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    OpenAIWorkspaceAgent,
    expand_env,
)
from src.attack_integrity import IntegrityReport, validate_attack_integrity
from src.dataset import compact_json, load_dabench_rows, load_jsonl, profile_files, resolve_input_files, rows_to_limit
from src.operators import Operator, get_enabled_operators
from src.runner import Runner
from src.validators import validate_robustness_payload, validation_failure_reason
from src.workspace import AttackWorkspace


TERMINAL_STATUSES = {"accepted", "rejected", "ineligible"}
REQUIRED_QUALITY_FLAGS = (
    "question_unchanged",
    "attack_effective",
    "necessary_evidence_preserved",
    "unique_answer_preserved",
    "answer_equivalent",
    "no_new_ambiguity_or_conflict",
    "files_readable",
)
SEMANTIC_REPAIRABLE_CATEGORIES = {
    "attack_not_effective",
    "answer_changed",
    "unanswerable",
    "ambiguity_introduced",
    "evidence_lost",
    "invalid_file",
}
_ROW_LOCKS: Dict[str, threading.Lock] = {}
_ROW_LOCKS_GUARD = threading.Lock()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    # Configuration files may live in configs/ or another subdirectory. Keep
    # every framework-relative path anchored at table_reason_robust/, not at
    # the location of the selected YAML file.
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)
    if args.limit is not None:
        config.setdefault("runner", {})["limit"] = args.limit
    if args.num_workers is not None:
        config.setdefault("runner", {})["num_workers"] = args.num_workers
    if args.max_tool_calls_per_stage is not None:
        for model_key in ("model", "judge_model"):
            config.setdefault(model_key, {})["max_tool_calls_per_stage"] = args.max_tool_calls_per_stage
    if args.skip_preflight:
        config.setdefault("runner", {})["preflight"] = False
    if args.dataset_path is not None:
        config["dataset_path"] = args.dataset_path
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.attacks is not None:
        config.setdefault("attacks", {})["enabled"] = [
            name.strip() for name in args.attacks.split(",") if name.strip()
        ]

    selected_ids = parse_sample_ids(args.sample_id, args.sample_ids)
    output_dir = resolve_config_path(project_root, config["output_dir"])
    ensure_output_dirs(output_dir)
    rows, data_root = load_dataset_rows(config, project_root, selected_ids)
    rows = rows_to_limit(rows, config.get("runner", {}).get("limit"))
    if selected_ids:
        rows = [row for row in rows if str(row.get("id")) in selected_ids]
        if not rows:
            raise RuntimeError(f"Sample ids not found after applying limit: {sorted(selected_ids)}")
    operators = get_enabled_operators(config.get("attacks", {}).get("enabled", []))
    if not operators:
        raise RuntimeError("No robustness attacks are enabled in config.attacks.enabled.")

    runner_config = config.get("runner", {})
    resume = summarize_resume(rows, output_dir, operators)
    log_event(
        "START "
        f"constructor={config.get('model', {}).get('name', '')} "
        f"judge={config.get('judge_model', {}).get('name', '')} "
        f"limit={runner_config.get('limit')} "
        f"num_workers={runner_config.get('num_workers', 4)} "
        f"enabled={','.join(op.name for op in operators)} "
        f"output_dir={output_dir}",
        context={"config": config},
    )
    log_event(
        f"RESUME loaded={resume['loaded']} completed_for_enabled={resume['completed']} remaining={resume['remaining']}",
        context={"config": config},
    )

    agent = build_agent(config)
    judge_agent = build_judge_agent(config)
    judge_available = True
    judge_preflight_error: Optional[str] = None
    if runner_config.get("preflight", True):
        log_event("PREFLIGHT constructor_start", context={"config": config})
        agent.preflight()
        log_event("PREFLIGHT constructor_ok", context={"config": config})
        log_event("PREFLIGHT judge_start", context={"config": config})
        try:
            judge_agent.preflight()
            log_event("PREFLIGHT judge_ok", context={"config": config})
        except AgentFatalError as exc:
            judge_available = False
            judge_preflight_error = short_reason(exc)
            log_event(f"PREFLIGHT judge_failed {judge_preflight_error}", level="quiet", context={"config": config})

    context = {
        "project_root": project_root,
        "config": config,
        "output_dir": output_dir,
        "data_root": data_root,
        "operators": operators,
        "agent": agent,
        "judge_agent": judge_agent,
        "judge_available": judge_available,
        "judge_preflight_error": judge_preflight_error,
    }
    runner = Runner(
        num_workers=runner_config.get("num_workers", 4),
        stall_timeout_sec=runner_config.get("stall_timeout_sec", 600),
    )
    try:
        runner.run(rows, lambda row: process_row(row, context))
    except Exception as exc:
        log_event(f"FATAL runner {short_reason(exc)}", level="quiet", context=context)
        raise
    rebuild_attack_dataset(output_dir)
    log_event(f"Done. Attack dataset: {output_dir / 'attack_dataset.jsonl'}", level="quiet", context=context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct answer-preserving table robustness attack samples.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--max-tool-calls-per-stage",
        type=int,
        default=None,
        help="Override the tool-call cap for both constructor and judge; useful for bounded smoke runs.",
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--attacks", default=None, help="Comma-separated attack names overriding config.attacks.enabled")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-ids", default=None, help="Comma-separated sample IDs; useful for bounded smoke runs")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip API preflight for repeated small smoke runs")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_agent(config: Dict[str, Any]) -> OpenAIWorkspaceAgent:
    return _build_model_agent(config.get("model", {}), CONSTRUCTOR_SYSTEM_PROMPT)


def build_judge_agent(config: Dict[str, Any]) -> OpenAIWorkspaceAgent:
    constructor = config.get("model", {})
    judge = config.get("judge_model")
    validate_judge_config(constructor, judge, allow_same=allow_same_judge_model(config))
    return _build_model_agent(judge, JUDGE_SYSTEM_PROMPT)


def validate_judge_config(constructor: Dict[str, Any], judge: Any, allow_same: bool = False) -> None:
    if not isinstance(judge, dict):
        raise RuntimeError("Missing required judge_model configuration.")
    missing = [key for key in ("base_url", "api_key", "name") if not str(judge.get(key, "")).strip()]
    if missing:
        raise RuntimeError(f"judge_model is missing required fields: {', '.join(missing)}")
    if _model_identity(constructor) == _model_identity(judge) and not allow_same:
        raise RuntimeError("model and judge_model must use distinct normalized base_url + name identities.")


def allow_same_judge_model(config: Dict[str, Any]) -> bool:
    return bool(config.get("judge_model", {}).get("allow_same_judge_model", False))


def parse_sample_ids(single: Optional[str], multiple: Optional[str]) -> set[str]:
    values = []
    if single:
        values.append(single)
    if multiple:
        values.extend(part.strip() for part in multiple.split(",") if part.strip())
    return {value if value.startswith("DA_") else value for value in values}


def load_dataset_rows(
    config: Dict[str, Any], project_root: Path, selected_ids: set[str]
) -> Tuple[List[Dict[str, Any]], Path]:
    dataset_config = config.get("dataset")
    if isinstance(dataset_config, dict) and str(dataset_config.get("kind", "")).lower() == "dabench":
        required = ("questions_path", "labels_path", "table_root")
        missing = [key for key in required if not str(dataset_config.get(key, "")).strip()]
        if missing:
            raise RuntimeError(f"DABENCH dataset config missing required fields: {', '.join(missing)}")
        table_root = resolve_config_path(project_root, dataset_config["table_root"])
        rows = load_dabench_rows(
            resolve_config_path(project_root, dataset_config["questions_path"]),
            resolve_config_path(project_root, dataset_config["labels_path"]),
            table_root,
            selected_ids=selected_ids or None,
        )
        return rows, table_root
    dataset_path = resolve_config_path(project_root, config["dataset_path"])
    data_root = resolve_config_path(project_root, config["data_root"])
    return load_jsonl(dataset_path), data_root


def _model_identity(model: Any) -> Tuple[str, str]:
    if not isinstance(model, dict):
        return "", ""
    return (
        expand_env(model.get("base_url")).strip().rstrip("/").lower(),
        str(model.get("name", "")).strip().lower(),
    )


def _build_model_agent(model: Dict[str, Any], system_prompt: str) -> OpenAIWorkspaceAgent:
    return OpenAIWorkspaceAgent(
        api_key=expand_env(model.get("api_key")),
        base_url=expand_env(model.get("base_url")),
        model_name=model.get("name", "gpt-4.1"),
        max_rounds=model.get("max_rounds", 12),
        temperature=model.get("temperature", 0.0),
        max_tokens=model.get("max_tokens", 8192),
        request_timeout=model.get("request_timeout", 120),
        max_retries=model.get("max_retries", 2),
        tool_timeout_sec=model.get("tool_timeout_sec", 180),
        max_tool_calls_per_stage=model.get("max_tool_calls_per_stage", 4),
        system_prompt=system_prompt,
    )


def resolve_config_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root / path


def ensure_output_dirs(output_dir: Path) -> None:
    for name in ["accepted", "rejected", "candidates", "logs", "workspaces"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def summarize_resume(rows: List[Dict[str, Any]], output_dir: Path, operators: Sequence[Operator]) -> Dict[str, int]:
    completed = 0
    enabled = {op.name for op in operators}
    for row in rows:
        log = load_json(output_dir / "logs" / f"{row.get('id', 'unknown')}.json", default={})
        states = log.get("attacks", {}) if isinstance(log, dict) else {}
        if enabled and all(states.get(name, {}).get("status") in TERMINAL_STATUSES for name in enabled):
            completed += 1
    return {"loaded": len(rows), "completed": completed, "remaining": len(rows) - completed}


def process_row(row: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(row.get("id", "unknown"))
    with _named_lock(_ROW_LOCKS, _ROW_LOCKS_GUARD, sample_id):
        return _process_row_unlocked(row, context, sample_id)


def _process_row_unlocked(row: Dict[str, Any], context: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    output_dir: Path = context["output_dir"]
    log_path = output_dir / "logs" / f"{sample_id}.json"
    row_log = load_json(log_path, default={"id": sample_id, "attacks": {}, "runs": []})
    row_log.setdefault("id", sample_id)
    row_log.setdefault("attacks", {})
    row_log.setdefault("runs", [])
    operators: List[Operator] = context["operators"]
    enabled_names = [op.name for op in operators]
    recover_existing_results(sample_id, operators, row_log, output_dir)
    restore_retryable_states(operators, row_log)

    if all(row_log["attacks"].get(name, {}).get("status") in TERMINAL_STATUSES for name in enabled_names):
        log_event(f"SAMPLE {sample_id} skipped completed_for_enabled", context=context)
        return {"id": sample_id, "status": "skipped_completed_for_enabled"}

    run_entry: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "enabled_attacks": enabled_names,
        "results": [],
    }
    row_log["runs"].append(run_entry)
    write_json(log_path, row_log)

    try:
        source_files = resolve_input_files(row, context["data_root"])
        source_profile = profile_files(source_files, virtual_root="/mnt/data")
        to_select = [op for op in operators if op.name not in row_log["attacks"]]
        if to_select:
            selection_states = select_attacks(row, source_files, source_profile, to_select, context)
            row_log["attacks"].update(selection_states)
            write_json(log_path, row_log)
    except AgentFatalError as exc:
        run_entry["fatal_error"] = exception_payload(exc)
        run_entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(log_path, row_log)
        log_event(f"SAMPLE {sample_id} selection_fatal {short_reason(exc)}", context=context)
        return {"id": sample_id, "status": "fatal_error", "stage": "selection", "error": str(exc)}
    except Exception as exc:
        run_entry["selection_error"] = exception_payload(exc)
        write_json(log_path, row_log)
        log_event(f"SAMPLE {sample_id} select_failed {short_reason(exc)}", context=context)
        return {"id": sample_id, "status": "select_failed", "error": str(exc)}

    eligible = [op for op in operators if row_log["attacks"].get(op.name, {}).get("status") == "eligible"]
    max_attacks = context["config"].get("attacks", {}).get("max_attacks_per_sample")
    if max_attacks is not None and str(max_attacks).strip().lower() not in {"", "none", "null", "0"}:
        eligible = eligible[: max(0, int(max_attacks))]

    for operator in eligible:
        selection = constrain_l4_selection(
            row_log["attacks"][operator.name].get("selection", {}), operator.name
        )
        new_id = stable_attack_id(sample_id, operator.name)
        try:
            result = construct_and_validate(
                row=row,
                source_files=source_files,
                source_profile=source_profile,
                operator=operator,
                selection=selection,
                new_id=new_id,
                context=context,
            )
        except AgentFatalError as exc:
            result = {"id": new_id, "status": "fatal_error", "error": exception_payload(exc)}
            row_log["attacks"][operator.name]["status"] = "fatal_error"
            row_log["attacks"][operator.name]["result"] = result
            run_entry["results"].append(result)
            write_json(log_path, row_log)
            log_event(f"ATTACK {new_id} fatal {short_reason(exc)}", context=context)
            break
        except AgentOutputError as exc:
            result = write_exception_rejected(new_id, row, operator, selection, context, exc)
        except Exception as exc:
            result = {"id": new_id, "status": "fatal_error", "error": exception_payload(exc)}
            row_log["attacks"][operator.name]["status"] = "fatal_error"
            row_log["attacks"][operator.name]["result"] = result
            run_entry["results"].append(result)
            write_json(log_path, row_log)
            raise

        row_log["attacks"][operator.name]["status"] = result["status"]
        row_log["attacks"][operator.name]["result"] = result
        run_entry["results"].append(result)
        write_json(log_path, row_log)

    run_entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
    row_log["completed_for_enabled"] = all(
        row_log["attacks"].get(name, {}).get("status") in TERMINAL_STATUSES for name in enabled_names
    )
    write_json(log_path, row_log)
    counts: Dict[str, int] = {}
    for name in enabled_names:
        status = str(row_log["attacks"].get(name, {}).get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
    log_event(
        f"SAMPLE {sample_id} done " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        context=context,
    )
    return {"id": sample_id, "status": "processed", "counts": counts}


def constrain_l4_selection(selection: Dict[str, Any], attack_type: str) -> Dict[str, Any]:
    """Give L4 construction one physical target, even for multi-file questions.

    A question may compare several files, but an L4 perturbation is deliberately
    a local context-noise attack. Letting the constructor edit every mentioned
    file both widens the attack surface and violates the one-pack/one-table
    integrity contract.
    """
    result = copy.deepcopy(selection) if isinstance(selection, dict) else {}
    if attack_type not in {"decoy_feature_pack_injection", "non_observation_row_injection"}:
        return result
    target = result.get("target")
    if not isinstance(target, dict):
        return result
    files = target.get("files")
    sheets = target.get("sheets")
    changed = False
    if isinstance(files, list) and len(files) > 1:
        target["files"] = [files[0]]
        changed = True
    if isinstance(sheets, list) and len(sheets) > 1:
        target["sheets"] = [sheets[0]]
        changed = True
    if changed:
        result["host_l4_target_constrained"] = True
        result["host_l4_target_constraint"] = "one file and one Sheet per L4 attack"
    return result


def _named_lock(
    registry: Dict[str, threading.Lock],
    guard: threading.Lock,
    name: str,
) -> threading.Lock:
    with guard:
        lock = registry.get(name)
        if lock is None:
            lock = threading.Lock()
            registry[name] = lock
        return lock


def select_attacks(
    row: Dict[str, Any],
    source_files: List[Path],
    profile: Dict[str, Any],
    operators: Sequence[Operator],
    context: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    states: Dict[str, Dict[str, Any]] = {}
    suffixes = {path.suffix.lower() for path in source_files if path.exists()}
    model_operators: List[Operator] = []
    for operator in operators:
        if not suffixes.intersection(operator.supported_extensions):
            states[operator.name] = {
                "status": "ineligible",
                "reason": f"no supported attack target; requires one of {operator.supported_extensions}",
            }
        else:
            model_operators.append(operator)
    if not model_operators:
        return states

    config = context["config"]
    prompt_path = context["project_root"] / "prompts" / "robustness" / "select_attack.md"
    catalog = [
        {
            "attack_type": op.name,
            "definition": op.definition,
            "supported_extensions": list(op.supported_extensions),
        }
        for op in model_operators
    ]
    prompt = render_template(
        prompt_path.read_text(encoding="utf-8"),
        {
            "sample_id": str(row.get("id", "")),
            "question": str(row.get("question", "")),
            "reference": str(row.get("reference", "")),
            "file_profile_json": compact_json(profile),
            "attack_catalog": compact_json(catalog),
            "allowed_attack_names_json": compact_json([op.name for op in model_operators]),
            "min_confidence": str(config.get("attacks", {}).get("min_confidence", 0.7)),
        },
    )
    payload, _history = context["agent"].complete_json(prompt, workspace=None, allow_tools=False)
    eligible = payload.get("eligible_attacks") or []
    rejected = payload.get("rejected_attacks") or []
    expected = {op.name for op in model_operators}
    seen: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for status, items in (("eligible", eligible), ("ineligible", rejected)):
        if not isinstance(items, list):
            raise AgentOutputError(f"selection field for {status} must be a list")
        for item in items:
            name = str(item.get("attack_type", ""))
            if name not in expected:
                raise AgentOutputError(f"selection returned unknown attack: {name}")
            if name in seen:
                raise AgentOutputError(f"selection returned duplicate attack: {name}")
            seen[name] = (status, item)
    missing = sorted(expected - set(seen))
    if missing:
        raise AgentOutputError(f"selection did not account for attacks: {missing}")

    min_confidence = float(config.get("attacks", {}).get("min_confidence", 0.7))
    for name, (status, item) in seen.items():
        if status == "eligible" and float(item.get("confidence", 0.0)) < min_confidence:
            states[name] = {
                "status": "ineligible",
                "reason": f"confidence below threshold: {item.get('confidence')}",
                "selection": item,
            }
        else:
            states[name] = {
                "status": status,
                "reason": item.get("reason"),
                "selection": item,
            }
    return states


def construct_and_validate(
    row: Dict[str, Any],
    source_files: List[Path],
    source_profile: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    new_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir: Path = context["output_dir"]
    accepted_path = output_dir / "accepted" / f"{new_id}.json"
    rejected_path = output_dir / "rejected" / f"{new_id}.json"
    candidate_path = output_dir / "candidates" / f"{new_id}.json"
    if accepted_path.exists():
        return {"id": new_id, "status": "accepted", "recovered": True}
    if rejected_path.exists():
        return {"id": new_id, "status": "rejected", "recovered": True}

    if not context.get("judge_available", True):
        reason = f"judge_error: {context.get('judge_preflight_error') or 'judge preflight failed'}"
        rejected = build_rejected(new_id, row, operator, selection, {}, reason)
        rejected["judge"] = judge_metadata(context)
        rejected["validation"] = judge_error_payload(reason)
        write_json(rejected_path, rejected)
        log_event(f"ATTACK {new_id} rejected {short_reason(reason)}", context=context)
        return {"id": new_id, "status": "rejected", "reason": reason}

    workspace = AttackWorkspace(output_dir, new_id, str(row.get("id", "")), operator.name)
    checkpoint = load_construct_checkpoint(candidate_path, workspace, row)
    construct_attempts: List[Dict[str, Any]] = []
    validation_attempts: List[Dict[str, Any]] = []
    semantic_feedback: Optional[str] = None
    max_semantic_repairs = max(
        0, int(context["config"].get("runner", {}).get("semantic_repair_attempts", 1))
    )

    for semantic_index in range(max_semantic_repairs + 1):
        if semantic_index == 0 and checkpoint is not None:
            stage = checkpoint
            construct_attempts = list(stage.get("construct_attempts", []))
            log_event(f"ATTACK {new_id} construct_checkpoint_reused", context=context)
        else:
            stage = run_construct_stage(
                row,
                source_files,
                operator,
                selection,
                new_id,
                candidate_path,
                rejected_path,
                workspace,
                context,
                prior_attempts=construct_attempts,
                initial_prompt_feedback=semantic_feedback,
            )
            if stage.get("terminal_result") is not None:
                return stage["terminal_result"]
            construct_attempts = list(stage.get("construct_attempts", construct_attempts))

        construct_payload = stage["construct_payload"]
        construct_history = stage["construct_history"]
        integrity_report = stage["integrity_report"]
        manifest = stage["manifest"]
        original_paths = [workspace.original / name for name in workspace.original_file_names()]
        original_profile = profile_files(original_paths, virtual_root="/mnt/original")
        attacked_paths = [workspace.final / name for name in workspace.final_file_names()]
        attacked_profile = profile_files(attacked_paths, virtual_root="/mnt/data")
        workspace.prepare_judge_snapshot(semantic_index + 1)
        validate_prompt = render_judge_validate_prompt(
            row,
            original_profile,
            attacked_profile,
            operator,
            workspace,
            context,
            manifest,
        )
        log_event(
            f"ATTACK {new_id} judge_validate_start attempt={semantic_index + 1}/{max_semantic_repairs + 1} "
            f"judge={judge_metadata(context)['model_name']}",
            context=context,
        )
        try:
            validate_payload, validate_history = context["judge_agent"].complete_json(
                validate_prompt, workspace=workspace, stage="judge", allow_tools=True
            )
        except (AgentFatalError, AgentOutputError) as exc:
            validate_payload = judge_error_payload(short_reason(exc))
            validate_history = []
        validation_attempts.append(
            {
                "attempt": semantic_index + 1,
                "validation": validate_payload,
                "judge_history": validate_history,
                "judge": judge_metadata(context),
                "integrity_report": integrity_report,
            }
        )
        write_validation_checkpoint(candidate_path, validation_attempts)
        record = build_attack_record(
            new_id,
            row,
            operator,
            selection,
            construct_payload,
            validate_payload,
            manifest,
        )
        record["construction"]["construct_history"] = construct_history
        record["construction"]["judge_history"] = validate_history
        record["construction"]["judge"] = judge_metadata(context)
        record["construction"]["construct_attempts"] = construct_attempts
        record["construction"]["validation_attempts"] = validation_attempts
        record["construction"]["integrity_report"] = integrity_report
        if validate_robustness_payload(validate_payload, operator.name, construct_payload):
            write_json(accepted_path, record)
            log_event(f"ATTACK {new_id} accepted", context=context)
            return {"id": new_id, "status": "accepted"}

        reason = validation_failure_reason(validate_payload, operator.name, construct_payload)
        category = str(validate_payload.get("failure_category") or "")
        if semantic_index < max_semantic_repairs and category in SEMANTIC_REPAIRABLE_CATEGORIES:
            semantic_feedback = render_semantic_repair_feedback(validate_payload, semantic_index + 2)
            log_event(
                f"ATTACK {new_id} semantic_check_failed repair={semantic_index + 1}/{max_semantic_repairs} "
                f"category={category} {short_reason(reason)}",
                context=context,
            )
            continue

        record["construction"]["validation_status"] = "failed"
        record["construction"]["failure_reason"] = reason
        write_json(rejected_path, record)
        log_event(f"ATTACK {new_id} rejected {short_reason(reason)}", context=context)
        return {"id": new_id, "status": "rejected", "reason": reason}

    raise RuntimeError("unreachable semantic repair loop")


def load_construct_checkpoint(
    candidate_path: Path,
    workspace: AttackWorkspace,
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not candidate_path.exists() or not workspace.manifest_path.exists() or not workspace.final.exists():
        return None
    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        construct_payload = candidate.get("construct", {})
        if construct_payload.get("status") != "constructed":
            return None
        if validate_construct_payload(construct_payload, row):
            return None
        manifest = json.loads(workspace.manifest_path.read_text(encoding="utf-8"))
        input_names = set(normalize_file_names(manifest.get("input_file")))
        if not input_names or input_names != set(workspace.final_file_names()):
            return None
        if workspace.validate_file_refs(str(manifest.get("input_file", ""))):
            return None
        file_diff = workspace.file_diff()
        if not any(file_diff.get(key) for key in ("added", "removed", "changed")):
            return None
        integrity_report = validate_attack_integrity(
            workspace,
            workspace.attack_type,
            construct_payload,
            candidate.get("selection", {}),
        )
        if not integrity_report.passed:
            return None
        manifest["file_diff"] = file_diff
        manifest["integrity_report"] = integrity_report.to_dict()
        return {
            "construct_payload": construct_payload,
            "construct_history": candidate.get("construct_history", []),
            "construct_attempts": candidate.get("construct_attempts", []),
            "integrity_report": integrity_report.to_dict(),
            "manifest": manifest,
        }
    except Exception:
        return None


def run_construct_stage(
    row: Dict[str, Any],
    source_files: List[Path],
    operator: Operator,
    selection: Dict[str, Any],
    new_id: str,
    candidate_path: Path,
    rejected_path: Path,
    workspace: AttackWorkspace,
    context: Dict[str, Any],
    prior_attempts: Optional[List[Dict[str, Any]]] = None,
    initial_prompt_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    configured_repairs = context["config"].get("runner", {}).get("construct_repair_attempts", 2)
    max_repairs = max(0, int(configured_repairs))
    attempts: List[Dict[str, Any]] = list(prior_attempts or [])
    repair_feedback: Optional[str] = None

    for attempt_index in range(max_repairs + 1):
        attempt_number = len(attempts) + 1
        log_event(
            f"ATTACK {new_id} construct_start attempt={attempt_number}/{max_repairs + 1}",
            context=context,
        )
        workspace.prepare(source_files)
        construct_profile = profile_files(sorted(workspace.original.iterdir()), virtual_root="/mnt/data")
        construct_prompt = render_construct_prompt(row, construct_profile, operator, selection, workspace, context)
        if attempt_index == 0 and initial_prompt_feedback:
            construct_prompt += initial_prompt_feedback
        if repair_feedback:
            construct_prompt += render_construct_repair_feedback(
                attempt_number,
                len(attempts) + max_repairs + 1,
                repair_feedback,
                attempts[-1].get("construct", {}),
            )
        construct_payload, construct_history = context["agent"].complete_json(
            construct_prompt, workspace=workspace, stage="construct", allow_tools=True
        )
        canonicalize_construct_question(construct_payload, row)

        if construct_payload.get("status") != "constructed":
            reason = str(construct_payload.get("reject_reason") or "construct_rejected")
            attempts.append(
                {
                    "attempt": attempt_number,
                    "construct": construct_payload,
                    "construct_history": construct_history,
                    "host_check_error": None,
                    "model_rejected": True,
                }
            )
            write_construct_candidate(candidate_path, new_id, row, operator, selection, attempts)
            rejected = build_rejected(new_id, row, operator, selection, construct_payload, reason)
            rejected["construction_attempts"] = attempts
            write_json(rejected_path, rejected)
            log_event(f"ATTACK {new_id} rejected {short_reason(reason)}", context=context)
            return {"terminal_result": {"id": new_id, "status": "rejected", "reason": reason}}

        host_error, input_file, integrity_report = inspect_constructed_files(
            construct_payload, row, workspace, operator, selection
        )
        attempts.append(
            {
                "attempt": attempt_number,
                "construct": construct_payload,
                "construct_history": construct_history,
                "host_check_error": host_error,
                "model_rejected": False,
            }
        )
        write_construct_candidate(candidate_path, new_id, row, operator, selection, attempts)
        if host_error is None:
            construct_payload["host_repair_attempts_used"] = attempt_index
            construct_payload["host_repair_failures"] = [
                item["host_check_error"] for item in attempts if item.get("host_check_error")
            ]
            manifest = workspace.write_manifest(
                input_file=input_file,
                modified=True,
                edit_summary=str(construct_payload.get("edit_summary") or construct_payload.get("edit_plan") or ""),
                original_files=[path.name for path in source_files],
                transformation_record=construct_payload.get("transformation_record", {}),
                integrity_report=integrity_report.to_dict(),
            )
            write_construct_candidate(candidate_path, new_id, row, operator, selection, attempts)
            return {
                "terminal_result": None,
                "construct_payload": construct_payload,
                "construct_history": construct_history,
                "construct_attempts": attempts,
                "integrity_report": integrity_report.to_dict(),
                "manifest": manifest,
            }

        if attempt_index < max_repairs:
            repair_feedback = host_error
            log_event(
                f"ATTACK {new_id} host_check_failed repair={attempt_number}/{max_repairs} "
                f"{short_reason(host_error)}",
                context=context,
            )
            continue

        rejected = build_rejected(new_id, row, operator, selection, construct_payload, host_error)
        rejected["construction_attempts"] = attempts
        write_json(rejected_path, rejected)
        log_event(f"ATTACK {new_id} rejected {short_reason(host_error)}", context=context)
        return {"terminal_result": {"id": new_id, "status": "rejected", "reason": host_error}}

    raise RuntimeError("unreachable construct repair loop")


def inspect_constructed_files(
    construct_payload: Dict[str, Any],
    row: Dict[str, Any],
    workspace: AttackWorkspace,
    operator: Operator,
    selection: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], str, IntegrityReport]:
    empty_report = IntegrityReport(attack_type=operator.name)
    payload_error = validate_construct_payload(construct_payload, row)
    if payload_error:
        empty_report.add("invalid_construct_payload", payload_error)
        return payload_error, "", empty_report
    output_files = normalize_file_names(construct_payload.get("output_files"))
    workspace.normalize_final_files(output_files)
    input_file = normalize_input_file(construct_payload.get("input_file") or "\n".join(output_files))
    missing = workspace.validate_file_refs(input_file)
    if missing:
        empty_report.add("missing_final_files", "declared final files do not exist", actual=missing)
        return f"missing final files: {missing}", input_file, empty_report
    input_names = set(normalize_file_names(input_file))
    final_names = set(workspace.final_file_names())
    if input_names != final_names:
        error = (
            f"input_file must exactly match final package: input={sorted(input_names)} "
            f"final={sorted(final_names)}"
        )
        empty_report.add("file_manifest_mismatch", error)
        return error, input_file, empty_report
    integrity_report = validate_attack_integrity(
        workspace, operator.name, construct_payload, selection or {}
    )
    integrity_error = integrity_report.error_message()
    if integrity_error:
        return integrity_error, input_file, integrity_report
    file_diff = workspace.file_diff()
    if not any(file_diff.get(key) for key in ("added", "removed", "changed")):
        empty_report.add("attack_not_changed", "attack did not change the final file package")
        return "attack did not change the final file package", input_file, empty_report
    return None, input_file, integrity_report


def write_construct_candidate(
    candidate_path: Path,
    new_id: str,
    row: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    attempts: List[Dict[str, Any]],
) -> None:
    latest = attempts[-1]
    write_json(
        candidate_path,
        {
            "id": new_id,
            "source_id": row.get("id"),
            "attack_type": operator.name,
            "selection": selection,
            "construct": latest.get("construct", {}),
            "construct_history": latest.get("construct_history", []),
            "construct_attempts": attempts,
        },
    )


def write_validation_checkpoint(
    candidate_path: Path,
    validation_attempts: List[Dict[str, Any]],
) -> None:
    if not candidate_path.exists():
        return
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["validation_attempts"] = validation_attempts
    write_json(candidate_path, candidate)


def render_construct_repair_feedback(
    attempt_number: int,
    total_attempts: int,
    host_error: str,
    previous_payload: Dict[str, Any],
) -> str:
    return (
        "\n\n[Framework integrity validation failed; reconstruction is required]\n"
        f"This is construction attempt {attempt_number}/{total_attempts}. The previous files have been discarded, "
        "and the workspace has been reset from the read-only original files. Do not rely on variables or output files from the previous process.\n"
        f"Framework integrity validation error: {host_error}\n"
        "Previous JSON declaration:\n"
        f"{compact_json(previous_payload)[:8000]}\n"
        "First call Python to reread /mnt/data, then fix the specific error above with a different implementation. "
        "Do not merely edit the JSON declaration or claim that checks passed; you must actually rewrite /mnt/output and verify it again. "
        "The final response must still be exactly one valid JSON object.\n"
    )


def render_semantic_repair_feedback(payload: Dict[str, Any], next_attempt: int) -> str:
    feedback = minimal_judge_feedback(payload)
    return (
        "\n\n[Independent semantic validation failed; redesign the attack from the original files]\n"
        f"The next round is semantic-repair construction attempt {next_attempt}. The previous files will be discarded.\n"
        f"Minimal feedback: {compact_json(feedback)}\n"
        "Revise the attack design according to the failure reason, instead of changing the question or only changing the JSON declaration. "
        "Reread /mnt/data, reconstruct /mnt/output, and show that the answer is unique and the attack is effective.\n"
    )


def minimal_judge_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return repair evidence without teaching the constructor the judge's answer.

    Answers, counterfactuals, audit rows, and tool traces would let a constructor
    optimize against the judge rather than construct a naturally robust sample.
    """
    checked = payload.get("checked_evidence")
    allowed = ("original_files", "attacked_files", "fields", "filters", "joins", "units", "transformation_checks")
    safe_checked = {
        key: checked.get(key)
        for key in allowed
        if isinstance(checked, dict) and key in checked
    }
    reason = str(payload.get("failure_reason") or validation_failure_reason(payload))
    return {
        "failure_category": payload.get("failure_category"),
        "failure_reason": reason[:1200],
        "checked_evidence": safe_checked,
    }


def validate_construct_payload(payload: Dict[str, Any], row: Dict[str, Any]) -> Optional[str]:
    if str(payload.get("new_question", "")) != str(row.get("question", "")):
        return "constructor changed the question"
    if payload.get("file_edit_required") is not True:
        return "robustness attack must edit the file package"
    if not normalize_file_names(payload.get("output_files")):
        return "output_files is empty"
    transformation = payload.get("transformation_record")
    if not isinstance(transformation, dict) or not transformation:
        return "transformation_record is missing or empty"
    quality = payload.get("quality_check")
    if not isinstance(quality, dict):
        return "quality_check is missing"
    failed = [name for name in REQUIRED_QUALITY_FLAGS if quality.get(name) is not True]
    if failed:
        return "quality_check failed or omitted: " + ", ".join(failed)
    return None


def canonicalize_construct_question(payload: Dict[str, Any], row: Dict[str, Any]) -> None:
    source_question = str(row.get("question", ""))
    model_question = str(payload.get("new_question", ""))
    if model_question != source_question:
        payload["model_new_question"] = model_question
        payload["question_canonicalized_by_host"] = True
    payload["new_question"] = source_question


def render_construct_prompt(
    row: Dict[str, Any],
    profile: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    workspace: AttackWorkspace,
    context: Dict[str, Any],
) -> str:
    prompt_path = context["project_root"] / operator.construct_prompt
    instruction_path = context["project_root"] / operator.instruction_prompt
    return render_template(
        prompt_path.read_text(encoding="utf-8"),
        {
            "attack_type": operator.name,
            "attack_definition": operator.definition,
            "attack_instruction": instruction_path.read_text(encoding="utf-8"),
            "selection_json": compact_json(selection),
            "sample_id": str(row.get("id", "")),
            "question": str(row.get("question", "")),
            "reference": str(row.get("reference", "")),
            "file_profile_json": compact_json(profile),
            "virtual_file_list": "\n".join(workspace.virtual_files(stage="construct")),
        },
    )


def render_judge_validate_prompt(
    row: Dict[str, Any],
    original_profile: Dict[str, Any],
    attacked_profile: Dict[str, Any],
    operator: Operator,
    workspace: AttackWorkspace,
    context: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    prompt_path = context["project_root"] / "prompts" / "robustness" / "judge_validate_robustness.md"
    instruction_path = context["project_root"] / operator.instruction_prompt
    file_diff = manifest.get("file_diff") if isinstance(manifest, dict) else {}
    host_facts = {
        "integrity_gate_passed": True,
        "attack_package_changed": bool(
            isinstance(file_diff, dict)
            and any(file_diff.get(key) for key in ("added", "removed", "changed"))
        ),
        "original_file_count": len(workspace.original_file_names()),
        "attacked_file_count": len(workspace.final_file_names()),
        "context_protocol": "blind_judge_v1",
    }
    return render_template(
        prompt_path.read_text(encoding="utf-8"),
        {
            "attack_type": operator.name,
            "attack_instruction": instruction_path.read_text(encoding="utf-8"),
            "judge_requirements": judge_requirements(operator.name),
            "question": str(row.get("question", "")),
            "reference": str(row.get("reference", "")),
            "host_facts_json": compact_json(host_facts),
            "original_virtual_file_list": "\n".join(
                workspace.virtual_files(stage="judge", original=True)
            ),
            "attacked_virtual_file_list": "\n".join(workspace.virtual_files(stage="judge")),
            "original_profile_json": compact_json(original_profile),
            "attacked_profile_json": compact_json(attacked_profile),
        },
    )


def judge_metadata(context: Dict[str, Any]) -> Dict[str, str]:
    same_model = allow_same_judge_model(context.get("config", {}))
    return {
        "role": "self_judge_smoke" if same_model else "independent_semantic_judge",
        "model_name": str(context.get("config", {}).get("judge_model", {}).get("name", "")),
        "context_protocol": "self_judge_smoke_v1" if same_model else "blind_judge_v1",
        "workspace_protocol": "isolated_snapshot_v1",
    }


def judge_requirements(attack_type: str) -> str:
    """Supply only the audit obligations relevant to the enabled operator."""
    requirements = []
    if attack_type in {"header_synonym_substitution", "semantic_distractor_column"}:
        requirements.append(
            "Return a nonempty field_binding_audit. Bind the question concept to one selected field; "
            "every nearby alternative needs a concrete exclusion reason."
        )
    if attack_type == "header_synonym_substitution":
        requirements.append(
            "Return a nonempty synonym_audit. Every changed header must preserve concept, metric scope, "
            "granularity, time basis, and unit, and must not coexist as a distinct field."
        )
    if attack_type in {"header_synonym_substitution", "semantic_distractor_column"}:
        requirements.append(
            "Return interpretation_risk_audit. Independently identify a correct field interpretation and one "
            "business-plausible but insufficient interpretation; prove the latter has a different observable "
            "outcome while the question and table evidence still uniquely recover the correct field. Do not assume "
            "a particular model, parser, library, fixed column position, or single-value lookup."
        )
    if attack_type == "semantic_distractor_column":
        requirements.append(
            "Return a nonempty counterfactual_answer from plausibly misusing the actual added distractor field."
        )
    if attack_type == "equivalent_value_reencoding":
        requirements.append(
            "Return interpretation_risk_audit. Prove that completing the task requires an explicit, unambiguous "
            "normalization or semantic parsing step, and that a business-plausible interpretation that omits it "
            "produces a different observable outcome or cannot complete the required operation."
        )
    if attack_type == "unit_scale_conversion":
        requirements.append(
            "Return interpretation_risk_audit. Prove that treating attacked values as the prior unit or omitting "
            "the conversion changes the requested output; reject cases where all relevant operands scale together "
            "and the unit cancels from the task."
        )
    if attack_type == "decoy_feature_pack_injection":
        requirements.append(
            "Enumerate every actual added feature in decoy_feature_audit; each must be type-compatible, "
            "uniquely excludable, and yield a different result when misused. Return a nonempty counterfactual_answer."
        )
    if attack_type == "non_observation_row_injection":
        requirements.append(
            "Enumerate every actual injected row or row group in non_observation_row_audit; prove its marker, "
            "non-observation status, unique exclusion, and a changed result when it is incorrectly included."
        )
    return "\n".join(f"- {item}" for item in requirements) or "- No additional audit beyond the common requirements."


def judge_error_payload(reason: str) -> Dict[str, Any]:
    return {
        "verdict": "failed",
        "attack_effective": False,
        "task_still_answerable": False,
        "unique_answer_preserved": False,
        "normalized_equivalent": False,
        "original_answer": "",
        "attacked_answer": "",
        "equivalence_evidence": "",
        "checked_evidence": {},
        "counterfactual_answer": None,
        "field_binding_audit": [],
        "synonym_audit": [],
        "decoy_feature_audit": [],
        "non_observation_row_audit": [],
        "interpretation_risk_audit": {},
        "reference_comparison": {"matches": False, "method": "judge unavailable", "differences": []},
        "failure_category": "judge_error",
        "failure_reason": f"Independent judge unavailable: {str(reason)[:1200]}",
    }


def build_attack_record(
    new_id: str,
    row: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    construct_payload: Dict[str, Any],
    validate_payload: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": new_id,
        "source_id": row.get("id"),
        "scenario": "table_reason",
        "dimension": "robustness",
        "attack_type": operator.name,
        "question": row.get("question"),
        "answer": row.get("reference"),
        "answer_relation": "invariant",
        "input_file": manifest.get("input_file"),
        "file_root": manifest.get("file_root"),
        "source_question": row.get("question"),
        "source_reference": row.get("reference"),
        "source_input_file": row.get("input_file"),
        "source_metadata": copy.deepcopy(row.get("metadata", {})),
        "transformation_record": construct_payload.get("transformation_record"),
        "construction": {
            "selection": selection,
            "edit_summary": construct_payload.get("edit_summary") or construct_payload.get("edit_plan"),
            "file_diff": manifest.get("file_diff"),
            "validation_status": validate_payload.get("verdict"),
            "validation": validate_payload,
            "manifest_path": str(Path(str(manifest.get("file_root"))).parent / "manifest.json"),
        },
    }


def build_rejected(
    new_id: str,
    row: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    payload: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "id": new_id,
        "source_id": row.get("id"),
        "scenario": "table_reason",
        "dimension": "robustness",
        "attack_type": operator.name,
        "source_question": row.get("question"),
        "source_reference": row.get("reference"),
        "source_metadata": copy.deepcopy(row.get("metadata", {})),
        "selection": selection,
        "rejected_reason": reason,
        "payload": payload,
    }


def write_exception_rejected(
    new_id: str,
    row: Dict[str, Any],
    operator: Operator,
    selection: Dict[str, Any],
    context: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    reason = f"exception: {type(exc).__name__}: {exc}"
    rejected = build_rejected(new_id, row, operator, selection, {"exception": exception_payload(exc)}, reason)
    write_json(context["output_dir"] / "rejected" / f"{new_id}.json", rejected)
    return {"id": new_id, "status": "rejected", "reason": reason}


def recover_existing_results(
    sample_id: str,
    operators: Sequence[Operator],
    row_log: Dict[str, Any],
    output_dir: Path,
) -> None:
    states = row_log.setdefault("attacks", {})
    for operator in operators:
        new_id = stable_attack_id(sample_id, operator.name)
        if (output_dir / "accepted" / f"{new_id}.json").exists():
            states.setdefault(operator.name, {})["status"] = "accepted"
            states[operator.name]["result"] = {"id": new_id, "status": "accepted", "recovered": True}
        elif (output_dir / "rejected" / f"{new_id}.json").exists():
            states.setdefault(operator.name, {})["status"] = "rejected"
            states[operator.name]["result"] = {"id": new_id, "status": "rejected", "recovered": True}


def restore_retryable_states(operators: Sequence[Operator], row_log: Dict[str, Any]) -> None:
    states = row_log.setdefault("attacks", {})
    for operator in operators:
        state = states.get(operator.name)
        if not isinstance(state, dict):
            continue
        if state.get("status") == "fatal_error" and isinstance(state.get("selection"), dict):
            state["previous_fatal_error"] = state.get("result")
            state["status"] = "eligible"
            state.pop("result", None)


def stable_attack_id(sample_id: str, attack_type: str) -> str:
    return f"{sample_id}__{attack_type}__001"


def rebuild_attack_dataset(output_dir: Path) -> None:
    dataset_path = output_dir / "attack_dataset.jsonl"
    lines = []
    for path in sorted((output_dir / "accepted").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        public = dict(record)
        construction = dict(public.get("construction", {}))
        construction.pop("construct_history", None)
        construction.pop("validate_history", None)
        construction.pop("judge_history", None)
        construction.pop("construct_attempts", None)
        construction.pop("validation_attempts", None)
        public["construction"] = construction
        lines.append(json.dumps(public, ensure_ascii=False))
    write_text_atomically(dataset_path, "\n".join(lines) + ("\n" if lines else ""))


def normalize_file_names(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        raw = [part.strip() for part in values.replace(";", "\n").replace("；", "\n").splitlines() if part.strip()]
    else:
        raw = [str(item).strip() for item in values if str(item).strip()]
    return [Path(item).name for item in raw]


def normalize_input_file(value: Any) -> str:
    return "\n".join(normalize_file_names(value))


def render_template(text: str, values: Dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically persist a state record despite transient Windows/NFS locks."""
    write_text_atomically(path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_text_atomically(path: Path, content: str) -> None:
    """Persist text through a unique sibling temporary file with short retries."""
    last_error: Optional[OSError] = None
    for attempt in range(6):
        temporary = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    assert last_error is not None
    raise last_error


def exception_payload(exc: Exception) -> Dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def log_event(message: str, level: str = "info", context: Optional[Dict[str, Any]] = None) -> None:
    config = (context or {}).get("config", {}) if context else {}
    if not should_log(level, config):
        return
    # A detached stdout (for example, an external job monitor timing out) must
    # not turn a completed sample into a worker failure.
    try:
        tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    except OSError:
        pass


def should_log(level: str, config: Dict[str, Any]) -> bool:
    order = {"quiet": 0, "info": 1, "verbose": 2}
    configured = str(config.get("runner", {}).get("log_level", "info")).strip().lower()
    if configured not in order:
        configured = "info"
    return order.get(level, order["info"]) <= order[configured]


def short_reason(value: Any, limit: int = 240) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    main()
