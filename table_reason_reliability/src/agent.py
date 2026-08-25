from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are a data-construction assistant for a reliability
benchmark on structured table analysis tasks.

Your goal is not to solve the original task. Your goal is to construct or
validate unanswerable attack samples from originally answerable samples, so that
the benchmark can test whether a model reliably refuses to answer when the
evidence-supported answer path is broken.

Important rules:
1. Use only virtual paths, such as /mnt/data/xxx.xlsx, /mnt/work/xxx.xlsx, or /mnt/output/xxx.xlsx.
2. Do not modify the original input files under /mnt/data.
3. If new files are needed, write them only under /mnt/output.
4. You may use the Python tool to inspect files, read headers, count rows, copy files, or modify files.
5. The final response must be exactly one valid JSON object. Do not output Markdown or explanatory prefixes/suffixes.
6. If a high-quality sample cannot be constructed, explicitly return rejected instead of forcing an attack.
7. Python tools run in isolated subprocesses. Every call must be self-contained and must not rely on variables from prior calls.
8. Do not use unbounded loops, threads, multiprocessing, huge matrices, or unconditional full-table scans. Prefer targeted reads of the necessary Sheets, columns, and rows.

During validation, use /mnt/original to read original files and /mnt/data to
read the final attacked files. Both /mnt/original and /mnt/data are read-only.
"""


class AgentError(RuntimeError):
    pass


class AgentFatalError(AgentError):
    pass


class AgentOutputError(AgentError):
    pass


class AgentTransientError(AgentError):
    """Temporary API/network failure. It must not terminate the full dataset run."""


class AgentOperationTimeout(AgentTransientError):
    """The current selection/attack exceeded its total time budget."""


class OpenAIWorkspaceAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_rounds: int = 12,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 8192,
        request_timeout: float = 120.0,
        max_retries: int = 2,
        tool_timeout_sec: float = 300.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentFatalError("The openai package is required. Install it with `pip install openai`.") from exc

        if not api_key:
            raise AgentFatalError("Missing API key. Set OPENAI_API_KEY or config.model.api_key.")
        self.request_timeout = float(request_timeout)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=self.request_timeout,
        )
        self.model_name = model_name
        self.max_rounds = int(max_rounds)
        self.temperature = float(temperature)
        self.max_retries = max(1, int(max_retries))
        self.tool_timeout_sec = max(1.0, float(tool_timeout_sec))
        if max_tokens is None or str(max_tokens).strip().lower() in {"", "none", "null"}:
            self.max_tokens = None
        else:
            self.max_tokens = int(max_tokens)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": (
                        "Execute Python code inside the current sample's isolated workspace. "
                        "Use the virtual paths /mnt/original, /mnt/data, /mnt/work, "
                        "/mnt/output, and /mnt/scratch."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute.",
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def complete_json(
        self,
        prompt: str,
        workspace: Any = None,
        stage: str = "construct",
        allow_tools: bool = True,
        deadline_monotonic: Optional[float] = None,
        max_rounds: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        namespace: Dict[str, Any] = {}
        tools = self.tools if allow_tools and workspace is not None else None
        round_limit = (
            self.max_rounds
            if max_rounds is None
            else max(1, int(max_rounds))
        )

        for _round in range(1, round_limit + 1):
            self._check_deadline(deadline_monotonic, stage)
            message = self._chat(
                messages,
                tools,
                deadline_monotonic=deadline_monotonic,
                stage=stage,
            )
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                assistant_entry = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                messages.append(assistant_entry)
                for tool_call in tool_calls:
                    result = self._handle_tool_call(
                        tool_call,
                        workspace,
                        stage,
                        namespace,
                        deadline_monotonic=deadline_monotonic,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result,
                        }
                    )
                continue

            content = message.content or ""
            messages.append({"role": "assistant", "content": content})
            try:
                return extract_json_object(content), messages
            except Exception as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer was not valid JSON. "
                            f"Parser error: {type(exc).__name__}: {exc}. "
                            "Please output exactly one valid JSON object only, without Markdown, comments, or extra text."
                        ),
                    }
                )
                continue

        raise AgentOutputError("Too many model/tool rounds without final JSON.")

    def preflight(self) -> None:
        try:
            payload, _history = self.complete_json(
                (
                    "This is an API connectivity and parameter compatibility preflight. "
                    "Output exactly one JSON object: {\"ok\": true}"
                ),
                workspace=None,
                allow_tools=False,
            )
        except AgentError as exc:
            raise AgentFatalError(f"LLM preflight failed: {exc}") from exc
        if payload.get("ok") is not True:
            raise AgentFatalError(f"LLM preflight returned unexpected payload: {payload}")

    def run_code(
        self,
        code: str,
        workspace: Any,
        stage: str = "construct",
        namespace: Optional[Dict[str, Any]] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> str:
        try:
            self._check_deadline(deadline_monotonic, stage)
            _check_code_safety(code)
            mapped = _map_virtual_paths(code, workspace.mapping(stage))

            remaining = self._remaining_seconds(deadline_monotonic)
            timeout_sec = self.tool_timeout_sec
            if remaining is not None:
                timeout_sec = min(timeout_sec, remaining)
            if timeout_sec <= 0:
                raise AgentOperationTimeout(
                    f"operation deadline exceeded before tool execution: stage={stage}"
                )

            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                popen_kwargs: Dict[str, Any] = {
                    "args": [sys.executable, "-u", "-c", mapped],
                    "stdout": stdout_file,
                    "stderr": stderr_file,
                }
                if os.name != "nt":
                    popen_kwargs["start_new_session"] = True

                process = subprocess.Popen(**popen_kwargs)
                timed_out = False
                try:
                    process.wait(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name != "nt":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait()

                out = _read_temp_output(stdout_file)
                err = _read_temp_output(stderr_file)

                if timed_out:
                    return _truncate(
                        "Execution timed out and the isolated tool process was killed. "
                        f"stage={stage}, timeout_sec={timeout_sec:.1f}. "
                        "Use a smaller, bounded, self-contained operation or return rejected."
                        + (f"\nSTDOUT:\n{out}" if out else "")
                        + (f"\nSTDERR:\n{err}" if err else "")
                    )

                if process.returncode == 0:
                    text = "Execution succeeded in an isolated process."
                else:
                    text = (
                        "Execution failed in an isolated process: "
                        f"returncode={process.returncode}. "
                        "Revise the bounded operation or return rejected."
                    )
                if out:
                    text += f"\nSTDOUT:\n{out}"
                if err:
                    text += f"\nSTDERR:\n{err}"
                return _truncate(text)
        except AgentOperationTimeout:
            raise
        except Exception as exc:
            text = (
                f"Execution blocked or failed: {type(exc).__name__}: {exc}\n"
                "This is a recoverable sandbox/tool result. Revise your approach and continue. "
                "Do not delete files under /mnt/data. For file_missing attacks, do not remove source files; "
                "instead set output_files/input_file to the subset of files that should remain in the final package."
            )
            return _truncate(text)

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        deadline_monotonic: Optional[float] = None,
        stage: str = "model",
    ) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self._check_deadline(deadline_monotonic, stage)
                remaining = self._remaining_seconds(deadline_monotonic)
                request_timeout = self.request_timeout
                if remaining is not None:
                    request_timeout = min(request_timeout, remaining)
                if request_timeout <= 0:
                    raise AgentOperationTimeout(
                        f"operation deadline exceeded before API request: stage={stage}"
                    )

                kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                response = self.client.with_options(
                    timeout=request_timeout
                ).chat.completions.create(**kwargs)
                if not response.choices:
                    raise AgentTransientError("Empty choices from model response.")
                return response.choices[0].message
            except AgentOperationTimeout:
                raise
            except Exception as exc:
                if self._is_content_inspection_error(exc):
                    raise AgentOutputError(
                        "API content inspection rejected the current sample: "
                        f"{type(exc).__name__}: {_truncate(str(exc), 1000)}"
                    ) from exc
                if self._is_fatal_api_error(exc):
                    raise AgentFatalError(
                        f"Non-retryable API error: {type(exc).__name__}: {exc}"
                    ) from exc
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = min(10.0, 2.0 ** attempt)
                    remaining = self._remaining_seconds(deadline_monotonic)
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining))
                    if delay > 0:
                        time.sleep(delay)
        raise AgentTransientError(
            f"API call failed after retries: {type(last_error).__name__}: {last_error}"
        )

    def _handle_tool_call(
        self,
        tool_call: Any,
        workspace: Any,
        stage: str,
        namespace: Dict[str, Any],
        deadline_monotonic: Optional[float] = None,
    ) -> str:
        if tool_call.function.name != "execute_code":
            return "Error: unknown tool."
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return "Error: tool arguments are not valid JSON."
        code = args.get("code", "")
        if not code:
            return "Error: empty code."
        return self.run_code(
            code,
            workspace=workspace,
            stage=stage,
            namespace=namespace,
            deadline_monotonic=deadline_monotonic,
        )

    @staticmethod
    def _remaining_seconds(
        deadline_monotonic: Optional[float],
    ) -> Optional[float]:
        if deadline_monotonic is None:
            return None
        return deadline_monotonic - time.monotonic()

    def _check_deadline(
        self,
        deadline_monotonic: Optional[float],
        stage: str,
    ) -> None:
        remaining = self._remaining_seconds(deadline_monotonic)
        if remaining is not None and remaining <= 0:
            raise AgentOperationTimeout(
                f"operation deadline exceeded: stage={stage}"
            )

    @staticmethod
    def _is_content_inspection_error(exc: Exception) -> bool:
        """Identify content-inspection errors that should reject only the current sample."""
        message = str(exc).lower()
        markers = (
            "data_inspection_failed",
            "input text data may contain inappropriate content",
            "content inspection failed",
            "content policy violation",
            "content_policy_violation",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _is_fatal_api_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        message = str(exc).lower()

        fatal_markers = (
            "invalid api key",
            "incorrect api key",
            "authentication",
            "model not exist",
            "model_not_found",
            "permission denied",
        )
        if any(marker in message for marker in fatal_markers):
            return True

        # Authentication, endpoint, method, and validation errors are usually not recoverable.
        if status_code in {401, 403, 404, 405, 422}:
            return True

        # Some OpenAI-compatible gateways wrap upstream 502/503/504 errors as HTTP 400.
        # These are transient gateway failures and should not terminate the dataset run.
        transient_markers = (
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
            "status code: 502",
            "status code: 503",
            "status code: 504",
            "status: 502",
            "status: 503",
            "status: 504",
            "cloudfront",
            "upstream timeout",
            "upstream timed out",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "connection timed out",
        )
        if any(marker in message for marker in transient_markers):
            return False

        # Real request-format, authentication, model, and parameter errors remain non-retryable.
        # 408, 429, and 5xx are intentionally excluded and continue through retry logic.
        return status_code == 400


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)
    raise AgentError(f"Could not parse JSON object from model output: {text[:500]}")


def _map_virtual_paths(code: str, mapping: Dict[str, Path]) -> str:
    mapped = code
    for virtual, real in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        mapped = mapped.replace(virtual, str(real).replace("\\", "/"))
    return mapped


def _check_code_safety(code: str) -> None:
    banned_fragments = [
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http://",
        "https://",
        "os.system",
        "popen",
        "eval(",
        "exec(",
        "rmtree",
        "unlink(",
        "remove(",
        "rmdir(",
        "D:/",
        "D:" + "\\",
        "C:/",
        "C:" + "\\",
    ]
    lowered = code.lower()
    for fragment in banned_fragments:
        if fragment.lower() in lowered:
            raise AgentError(f"Unsafe code rejected because it contains: {fragment}")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise AgentError(f"Python syntax error before execution: {exc}") from exc
    banned_imports = {"subprocess", "socket", "requests", "urllib", "httpx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in banned_imports:
                    raise AgentError(f"Unsafe import rejected: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in banned_imports:
                raise AgentError(f"Unsafe import rejected: {node.module}")


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:2000] + "\n...[truncated]...\n" + text[-2000:]


def _read_temp_output(file: Any, limit: int = 4000) -> str:
    file.flush()
    file.seek(0, os.SEEK_END)
    size = file.tell()
    if size <= limit:
        file.seek(0)
        data = file.read()
    else:
        half = limit // 2
        file.seek(0)
        first = file.read(half)
        file.seek(max(0, size - half))
        last = file.read(half)
        data = first + b"\n...[truncated]...\n" + last
    return data.decode("utf-8", errors="replace")


def expand_env(value: Optional[str]) -> str:
    if value is None:
        return ""
    match = re.fullmatch(r"\$\{([^}]+)\}", str(value).strip())
    if match:
        return os.environ.get(match.group(1), "")
    return str(value)

