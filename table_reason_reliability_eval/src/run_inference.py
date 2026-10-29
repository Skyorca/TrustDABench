from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from agent import AgentFatalError, OpenAIToolAgent
from dataset import load_dataset
from runner import Runner
from utils import append_jsonl, expand_env, load_completed_ids, load_yaml, log_event, read_text
from workspace import SampleWorkspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agentic inference on table reliability attack samples.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"), help="Path to config.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit overriding config.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional worker count overriding config.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and build first prompts without calling LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)

    dataset_path = Path(config["dataset_path"])
    output_path = Path(config["output_path"])
    runner_cfg = config.get("runner", {})
    agent_cfg = config.get("agent", {})
    model_cfg = config.get("model", {})
    log_level = runner_cfg.get("log_level", "info")

    limit = args.limit if args.limit is not None else _optional_int(runner_cfg.get("limit"))
    num_workers = args.num_workers if args.num_workers is not None else int(runner_cfg.get("num_workers", 1))

    system_prompt = read_text(PROJECT_ROOT / "prompts" / "system_prompt.txt")
    user_template = read_text(PROJECT_ROOT / "prompts" / "user_prompt.txt")

    rows = load_dataset(dataset_path, limit=limit)
    completed_ids = load_completed_ids(output_path)
    pending = [row for row in rows if str(row["id"]) not in completed_ids]

    log_event(
        f"Loaded {len(rows)} samples; completed_successfully {len(completed_ids)}; "
        f"remaining {len(pending)}; workers={num_workers}; output={output_path}",
        log_level=log_level,
    )

    if args.dry_run:
        dry_run(rows[: min(3, len(rows))], user_template, agent_cfg, output_path, log_level)
        return

    log_event("Running LLM preflight...", log_level=log_level)
    build_agent(config, system_prompt).preflight()
    log_event("LLM preflight passed.", log_level=log_level)

    if not pending:
        log_event(f"Done. Predictions: {output_path}", log_level=log_level)
        return

    runner = Runner(num_workers=num_workers, stall_timeout_sec=runner_cfg.get("stall_timeout_sec", 600))

    def process(row: Dict[str, Any]) -> Dict[str, Any]:
        return process_sample(row, config, system_prompt, user_template)

    for result in runner.run_completed(pending, process):
        append_jsonl(output_path, result)
        status = result.get("status")
        detail = result.get("prediction", {}).get("finish_reason") or result.get("error", "")
        log_event(f"SAMPLE {result.get('id')} {status} {detail}", log_level=log_level)

    log_event(f"Done. Predictions: {output_path}", log_level=log_level)


def process_sample(
    row: Dict[str, Any],
    config: Dict[str, Any],
    system_prompt: str,
    user_template: str,
) -> Dict[str, Any]:
    started = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    agent_cfg = config.get("agent", {})
    output_path = Path(config["output_path"])
    keep_workspaces = bool(agent_cfg.get("keep_workspaces", False))
    workspace_base = output_path.parent / "workspaces" if keep_workspaces else None

    base_result = {
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "scenario": row.get("scenario"),
        "dimension": row.get("dimension"),
        "attack_type": row.get("attack_type"),
        "question": row.get("question"),
        "gold_answer": row.get("answer"),
        "source_question": row.get("source_question"),
        "source_reference": row.get("source_reference"),
        "construction": row.get("construction"),
        "input_file": row.get("input_file"),
        "file_root": row.get("file_root"),
        "model": config.get("model", {}).get("name"),
        "started_at": started_at,
    }

    try:
        with SampleWorkspace(row, keep=keep_workspaces, base_dir=workspace_base) as workspace:
            prompt = build_user_prompt(user_template, workspace.file_names, str(row["question"]))
            agent = build_agent(config, system_prompt)
            prediction = agent.run(prompt, workspace.working_dir)
            status = "success" if not prediction.get("error") else "run_failed"
            return {
                **base_result,
                "status": status,
                "provided_files": workspace.file_names,
                "workspace_manifest": workspace.manifest,
                "prompt": {
                    "system": system_prompt,
                    "user": prompt,
                },
                "prediction": prediction,
                "error": prediction.get("error"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_sec": round(time.time() - started, 3),
            }
    except AgentFatalError as exc:
        return _failed_result(base_result, "api_failed", exc, started)
    except Exception as exc:
        return _failed_result(base_result, "workspace_failed", exc, started)


def build_agent(config: Dict[str, Any], system_prompt: str) -> OpenAIToolAgent:
    model_cfg = config.get("model", {})
    agent_cfg = config.get("agent", {})
    return OpenAIToolAgent(
        api_key=expand_env(model_cfg.get("api_key")),
        base_url=str(model_cfg.get("base_url") or ""),
        model_name=str(model_cfg.get("name") or ""),
        system_prompt=system_prompt,
        max_rounds=int(agent_cfg.get("max_rounds", 12)),
        max_tokens=model_cfg.get("max_tokens"),
        request_timeout=float(model_cfg.get("request_timeout", 120)),
        max_retries=int(model_cfg.get("max_retries", 2)),
        tool_timeout_sec=int(agent_cfg.get("tool_timeout_sec", 1200)),
        tool_output_max_chars=agent_cfg.get("tool_output_max_chars", 12000),
        temperature=model_cfg.get("temperature"),
        top_p=model_cfg.get("top_p"),
    )


def build_user_prompt(template: str, file_names: List[str], question: str) -> str:
    file_list = "\n".join(f"- {name}" for name in file_names)
    return template.format(file_list=file_list, question=question)


def dry_run(
    rows: List[Dict[str, Any]],
    user_template: str,
    agent_cfg: Dict[str, Any],
    output_path: Path,
    log_level: str,
) -> None:
    keep_workspaces = bool(agent_cfg.get("keep_workspaces", False))
    workspace_base = output_path.parent / "workspaces" if keep_workspaces else None
    for row in rows:
        with SampleWorkspace(row, keep=keep_workspaces, base_dir=workspace_base) as workspace:
            prompt = build_user_prompt(user_template, workspace.file_names, str(row["question"]))
            if "D:\\" in prompt or "C:\\" in prompt:
                raise RuntimeError(f"Prompt path leak detected for sample {row['id']}")
            log_event(f"DRY {row['id']} files={workspace.file_names} prompt_chars={len(prompt)}", log_level=log_level)
    log_event("Dry run passed.", log_level=log_level)


def _failed_result(base_result: Dict[str, Any], status: str, exc: Exception, started: float) -> Dict[str, Any]:
    return {
        **base_result,
        "status": status,
        "provided_files": [],
        "workspace_manifest": {},
        "prompt": {},
        "prediction": {
            "final_answer": "",
            "finish_reason": status,
            "messages": [],
            "tool_events": [],
            "tool_call_count": 0,
            "rounds": 0,
            "error": f"{type(exc).__name__}: {exc}",
        },
        "error": f"{type(exc).__name__}: {exc}",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started, 3),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
