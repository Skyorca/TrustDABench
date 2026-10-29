from __future__ import annotations

import argparse
import concurrent.futures
import json

import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from utils import append_jsonl, expand_env, iter_jsonl, load_yaml, log_event, read_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {"correct", "incorrect"}
VALID_SCORES = {0, 1}
JUDGE_MODES = {"llm_refusal", "dabench_closed_form"}


class JudgeFatalError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge whether model predictions are correct.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "judge_config.yaml"), help="Path to judge_config.yaml.")
    parser.add_argument("--input", action="append", default=None, help="Input predictions JSONL. Can be passed multiple times.")
    parser.add_argument("--output", default=None, help="Optional output judgments JSONL. Only valid with one input.")
    parser.add_argument("--report", default=None, help="Optional output summary report JSON. Only valid with one input.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total sample limit.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional worker count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    runner_cfg = config.get("runner", {})
    judge_cfg = config.get("judge", {})
    judge_mode = str(judge_cfg.get("mode") or "llm_refusal").strip()
    if judge_mode not in JUDGE_MODES:
        raise JudgeFatalError(f"Unknown judge.mode: {judge_mode}. Expected one of {sorted(JUDGE_MODES)}.")
    log_level = runner_cfg.get("log_level", "info")

    input_paths = [Path(p) for p in (args.input or config.get("input_paths") or [])]
    if not input_paths:
        raise JudgeFatalError("No input JSONL paths configured. Set input_paths in judge_config.yaml or pass --input.")

    if (args.output or args.report) and len(input_paths) != 1:
        raise JudgeFatalError("--output/--report overrides are only allowed when there is exactly one input path.")
    limit = args.limit if args.limit is not None else _optional_int(runner_cfg.get("limit"))
    num_workers = args.num_workers if args.num_workers is not None else int(runner_cfg.get("num_workers", 1))

    judge_prompt = ""
    dabench_scorer = None
    if judge_mode == "llm_refusal":
        prompt_path = Path(judge_cfg.get("prompt_path") or PROJECT_ROOT / "prompts" / "judge_refusal.md")
        if not prompt_path.is_absolute():
            prompt_path = PROJECT_ROOT / prompt_path
        judge_prompt = read_text(prompt_path)
    else:
        dabench_scorer = build_dabench_scorer(config)

    input_groups, input_stats = load_prediction_groups(input_paths, limit=limit)
    total_rows = sum(len(rows) for _, rows in input_groups)
    output_jobs = []
    for input_path, rows in input_groups:
        model_name = _extract_model_name(rows) or "model"
        output_path, report_path = derive_output_paths(input_path, args.output, args.report, model_name, judge_mode)
        completed = load_completed_keys(output_path) if bool(judge_cfg.get("skip_completed", True)) else set()
        pending = [row for row in rows if row_key(row) not in completed]
        output_jobs.append((input_path, output_path, report_path, rows, completed, pending))

    log_event(
        f"Loaded selected={total_rows} physical_rows={input_stats['physical_rows']} "
        f"unique_tasks={input_stats['unique_tasks']} collapsed_attempts={input_stats['collapsed_attempts']} "
        f"from {len(input_groups)} file(s); judge_mode={judge_mode}; workers={num_workers}",
        log_level=log_level,
    )
    for input_path, output_path, _, rows, completed, pending in output_jobs:
        log_event(
            f"INPUT {input_path.name}: loaded={len(rows)} completed={len(completed)} remaining={len(pending)} output={output_path.name}",
            log_level=log_level,
        )

    needs_llm = judge_mode == "llm_refusal" and any(is_llm_judgable(row) for _, _, _, _, _, pending in output_jobs for row in pending)
    if needs_llm:
        log_event("Running judge LLM preflight...", log_level=log_level)
        build_judge(config).preflight()
        log_event("Judge LLM preflight passed.", log_level=log_level)

    for input_path, output_path, report_path, _, _, pending in output_jobs:
        if pending:
            if judge_mode == "dabench_closed_form":
                assert dabench_scorer is not None
                run_parallel(
                    pending,
                    lambda row: judge_one_dabench(row, dabench_scorer),
                    output_path,
                    num_workers,
                    runner_cfg,
                    log_level,
                )
            else:
                run_parallel(pending, lambda row: judge_one(row, config, judge_prompt), output_path, num_workers, runner_cfg, log_level)

        report = build_report(output_path)
        if judge_mode == "dabench_closed_form":
            assert dabench_scorer is not None
            report["dabench_closed_form"] = dabench_scorer.build_dataset_report(iter_jsonl(output_path))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log_event(f"Done. {input_path.name} judgments: {output_path}", log_level=log_level)
        log_event(f"Done. {input_path.name} report: {report_path}", log_level=log_level)


def load_prediction_groups(
    input_paths: List[Path], limit: Optional[int]
) -> Tuple[List[Tuple[Path, List[Dict[str, Any]]]], Dict[str, int]]:
    canonical_groups: List[Tuple[Path, List[Dict[str, Any]]]] = []
    physical_rows = 0
    unique_tasks = 0
    for input_path in input_paths:
        rows, stats = canonical_prediction_rows(iter_jsonl(input_path), input_path)
        canonical_groups.append((input_path, rows))
        physical_rows += stats["physical_rows"]
        unique_tasks += stats["unique_tasks"]

    groups: List[Tuple[Path, List[Dict[str, Any]]]] = []
    remaining = limit
    for input_path, rows in canonical_groups:
        if remaining is not None and remaining <= 0:
            break
        selected = rows if remaining is None else rows[:remaining]
        groups.append((input_path, selected))
        if remaining is not None:
            remaining -= len(selected)
    return groups, {
        "physical_rows": physical_rows,
        "unique_tasks": unique_tasks,
        "collapsed_attempts": physical_rows - unique_tasks,
    }


def canonical_prediction_rows(
    rows: Iterable[Dict[str, Any]], input_path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    ordered_ids: List[str] = []
    latest: Dict[str, Dict[str, Any]] = {}
    latest_success: Dict[str, Dict[str, Any]] = {}
    attempt_counts: Dict[str, int] = defaultdict(int)
    physical_rows = 0

    for line_number, row in enumerate(rows, 1):
        physical_rows += 1
        item = dict(row)
        raw_id = item.get("id")
        if raw_id is None or not str(raw_id).strip():
            raise JudgeFatalError(f"Missing non-empty id at {input_path}:{line_number}")
        item_id = str(raw_id)
        item["_input_path"] = str(input_path)
        item["_input_line"] = line_number
        if item_id not in latest:
            ordered_ids.append(item_id)
        latest[item_id] = item
        attempt_counts[item_id] += 1
        if item.get("status") == "success" and extract_final_answer(item).strip():
            latest_success[item_id] = item

    selected: List[Dict[str, Any]] = []
    for item_id in ordered_ids:
        item = dict(latest_success.get(item_id, latest[item_id]))
        item["_attempt_count"] = attempt_counts[item_id]
        selected.append(item)
    return selected, {
        "physical_rows": physical_rows,
        "unique_tasks": len(selected),
        "collapsed_attempts": physical_rows - len(selected),
    }


def load_prediction_rows(input_paths: List[Path], limit: Optional[int]) -> List[Dict[str, Any]]:
    groups, _ = load_prediction_groups(input_paths, limit=limit)
    return [row for _, rows in groups for row in rows]


def derive_output_paths(
    input_path: Path,
    output_override: Optional[str],
    report_override: Optional[str],
    model_name: str = "model",
    judge_mode: str = "llm_refusal",
) -> Tuple[Path, Path]:
    if output_override:
        output_path = Path(output_override)
    else:
        suffix = "_dabench_closed_form_judgments.jsonl" if judge_mode == "dabench_closed_form" else "_judgments.jsonl"
        output_path = input_path.parent / f"{model_name}{suffix}"
    if report_override:
        report_path = Path(report_override)
    else:
        suffix = "_dabench_closed_form_report.json" if judge_mode == "dabench_closed_form" else "_report.json"
        report_path = input_path.parent / f"{model_name}{suffix}"
    return output_path, report_path


def _extract_model_name(rows: List[Dict[str, Any]]) -> Optional[str]:
    for row in rows:
        name = row.get("model")
        if isinstance(name, str) and name.strip():
            return re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    return None


def load_completed_keys(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for row in iter_jsonl(path):
        key = row.get("judgment_id")
        if key:
            completed.add(str(key))
    return completed


def row_key(row: Dict[str, Any]) -> str:
    return f"{row.get('_input_path')}::{row.get('id')}::{row.get('_input_line')}"


def is_llm_judgable(row: Dict[str, Any]) -> bool:
    return row.get("status") == "success" and bool(extract_final_answer(row).strip())


def build_dabench_scorer(config: Dict[str, Any]) -> Any:
    from scorer_dabench import DabenchClosedFormScorer

    dabench_cfg = config.get("dabench", {})
    labels_path = dabench_cfg.get("labels_path")
    if not labels_path:
        raise JudgeFatalError("judge.mode=dabench_closed_form requires dabench.labels_path.")
    questions_path = dabench_cfg.get("questions_path")
    return DabenchClosedFormScorer(
        labels_path=resolve_project_path(labels_path),
        questions_path=resolve_project_path(questions_path) if questions_path else None,
        id_prefix=str(dabench_cfg.get("id_prefix") or "DA_"),
    )


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def judge_one(row: Dict[str, Any], config: Dict[str, Any], judge_prompt: str) -> Dict[str, Any]:
    started = time.time()
    final_answer = extract_final_answer(row)
    base = {
        "judgment_id": row_key(row),
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "attack_type": row.get("attack_type"),
        "model": row.get("model"),
        "run_status": row.get("status"),
        "input_path": row.get("_input_path"),
        "input_line": row.get("_input_line"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "final_answer": final_answer,
        "judge_model": config.get("model", {}).get("name"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    if row.get("status") != "success" or not final_answer.strip():
        return {
            **base,
            "judge_status": "not_evaluable",
            "score": 0,
            "label": "incorrect",
            "is_correct": False,
            "reason": "模型没有产出可判别的最终答案，按 0 分计。",
            "raw_judge_response": "",
            "error": row.get("error") or "prediction_not_evaluable",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - started, 3),
        }

    try:
        payload, raw = build_judge(config).judge(
            prompt=judge_prompt,
            question=str(row.get("question") or ""),
            gold_answer=row.get("gold_answer"),
            final_answer=final_answer,
        )
        payload = normalize_judge_payload(payload)
        return {
            **base,
            "judge_status": "success",
            **payload,
            "raw_judge_response": raw,
            "error": None,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            **base,
            "judge_status": "judge_failed",
            "score": 0,
            "label": "incorrect",
            "is_correct": False,
            "reason": "判别器调用或解析失败，按 0 分计。",
            "raw_judge_response": "",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - started, 3),
        }


def judge_one_dabench(row: Dict[str, Any], scorer: Any) -> Dict[str, Any]:
    started = time.time()
    final_answer = extract_final_answer(row)
    base = {
        "judgment_id": row_key(row),
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "attack_type": row.get("attack_type"),
        "model": row.get("model"),
        "run_status": row.get("status"),
        "input_path": row.get("_input_path"),
        "input_line": row.get("_input_line"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "final_answer": final_answer,
        "judge_model": "dabench_closed_form",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        payload = scorer.score_prediction(
            prediction_id=row.get("id"),
            source_id=row.get("source_id"),
            response=final_answer,
            run_status=row.get("status"),
            error=row.get("error"),
        )
        return {
            **base,
            **payload,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            **base,
            "judge_status": "judge_failed",
            "score": 0,
            "label": "incorrect",
            "is_correct": False,
            "reason": "DABENCH closed-form scorer failed, counted as incorrect.",
            "label_id": None,
            "label_answers": None,
            "predicted_answers": {},
            "correctness": {},
            "judge_method": "dabench_closed_form_v1",
            "raw_judge_response": "",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - started, 3),
        }


def extract_final_answer(row: Dict[str, Any]) -> str:
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        final_answer = prediction.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            return final_answer
        raw = prediction.get("raw_final_answer")
        if isinstance(raw, str) and raw.strip():
            return raw
    for key in ("model_response", "response", "final_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalize_judge_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    score = payload.get("score", 0)
    try:
        score = int(float(score))
    except Exception:
        score = 0
    if score not in VALID_SCORES:
        score = 0  # default to incorrect if not valid

    label = str(payload.get("label") or "").strip()
    if label not in VALID_LABELS:
        label = score_to_label(score)

    is_correct = bool(payload.get("is_correct", score == 1))
    return {
        "score": score,
        "label": label,
        "is_correct": is_correct,
        "reason": str(payload.get("reason") or "")[:1000],
    }


def score_to_label(score: float) -> str:
    return "correct" if score == 1 else "incorrect"


def run_parallel(
    rows: List[Dict[str, Any]],
    process_func: Any,
    output_path: Path,
    num_workers: int,
    runner_cfg: Dict[str, Any],
    log_level: str,
) -> None:
    num_workers = max(1, int(num_workers))
    if num_workers == 1:
        iterator: Iterable[Dict[str, Any]] = (process_func(row) for row in rows)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(rows))
        for result in iterator:
            write_and_log(result, output_path, log_level)
        return

    stall_timeout = _optional_float(runner_cfg.get("stall_timeout_sec"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_row = {executor.submit(process_func, row): row for row in rows}
        pending = set(future_to_row)
        progress = tqdm(total=len(future_to_row)) if tqdm is not None else None
        try:
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=stall_timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    pending_ids = [str(future_to_row[future].get("id", "unknown")) for future in pending]
                    raise TimeoutError(f"No judgment finished within {stall_timeout} seconds. Pending ids: {pending_ids}")
                for future in done:
                    if progress is not None:
                        progress.update(1)
                    write_and_log(future.result(), output_path, log_level)
        finally:
            if progress is not None:
                progress.close()


def write_and_log(result: Dict[str, Any], output_path: Path, log_level: str) -> None:
    append_jsonl(output_path, result)
    log_event(
        f"JUDGE {result.get('id')} {result.get('judge_status')} score={result.get('score')} label={result.get('label')}",
        log_level=log_level,
    )


class Judge:
    def __init__(self, config: Dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise JudgeFatalError("The openai package is required. Install it with `pip install openai`.") from exc

        model_cfg = config.get("model", {})
        api_key = expand_env(model_cfg.get("api_key"))
        if not api_key:
            raise JudgeFatalError("Missing judge API key. Set judge_config.yaml model.api_key or OPENAI_API_KEY.")
        self.client = OpenAI(
            api_key=api_key,
            base_url=model_cfg.get("base_url") or None,
            timeout=float(model_cfg.get("request_timeout", 120)),
        )
        self.model_name = model_cfg.get("name")
        if not self.model_name:
            raise JudgeFatalError("Missing judge model name.")
        self.max_tokens = _optional_int(model_cfg.get("max_tokens"))
        self.max_retries = max(1, int(model_cfg.get("max_retries", 2)))

    def preflight(self) -> None:
        response = self._chat([{"role": "user", "content": "Reply with exactly: ok"}])
        if "ok" not in response.lower():
            raise JudgeFatalError(f"Judge preflight returned unexpected content: {response[:200]}")

    def judge(self, prompt: str, question: str, gold_answer: Any, final_answer: str) -> Tuple[Dict[str, Any], str]:
        user_content = (
            "question:\n"
            f"{question}\n\n"
            "gold_answer:\n"
            f"{json.dumps(gold_answer, ensure_ascii=False)}\n\n"
            "model_final_answer:\n"
            f"{final_answer}\n"
        )
        raw = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ]
        )
        return parse_json_object(raw), raw

    def _chat(self, messages: List[Dict[str, Any]]) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {"model": self.model_name, "messages": messages}
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise JudgeFatalError("Empty choices from judge response.")
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise JudgeFatalError(f"Judge API call failed after retries: {last_error}")


def build_judge(config: Dict[str, Any]) -> Judge:
    return Judge(config)


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge response is not a JSON object.")
    return value


def build_report(path: Path) -> Dict[str, Any]:
    rows = list(iter_jsonl(path)) if path.exists() else []
    by_attack: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_attack[str(row.get("attack_type") or "unknown")].append(row)
        by_model[str(row.get("model") or "unknown")].append(row)

    return {
        "total": len(rows),
        **summarize_rows(rows),
        "by_attack_type": {key: summarize_rows(value) for key, value in sorted(by_attack.items())},
        "by_model": {key: summarize_rows(value) for key, value in sorted(by_model.items())},
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    evaluable_rows = [row for row in rows if row.get("judge_status") == "success"]
    evaluable = len(evaluable_rows)
    not_evaluable = total - evaluable
    scores = [float(row.get("score") or 0) for row in rows]
    evaluable_scores = [float(row.get("score") or 0) for row in evaluable_rows]
    correct = sum(1 for score in scores if score >= 1)
    evaluable_correct = sum(1 for score in evaluable_scores if score >= 1)
    return {
        "total": total,
        "evaluable": evaluable,
        "not_evaluable": not_evaluable,
        "mean_score_all": round(sum(scores) / total, 4) if total else 0,
        "mean_score_evaluable": round(sum(evaluable_scores) / evaluable, 4) if evaluable else 0,
        "accuracy_all": round(correct / total, 4) if total else 0,
        "accuracy_evaluable": round(evaluable_correct / evaluable, 4) if evaluable else 0,
        "correct_count": correct,
        "incorrect_count": total - correct,
        "label_counts": dict(sorted(count_labels(rows).items())),
    }


def count_labels(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("label") or "unknown")] += 1
    return counts


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "0"}:
        return None
    return float(value)


if __name__ == "__main__":
    main()
