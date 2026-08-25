from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CONSTRUCTOR_SYSTEM_PROMPT = """You are a data-construction and validation
assistant for a robustness benchmark on structured table analysis tasks.

Your goal is to construct or validate answer-preserving attacks from originally
answerable samples. After the attack, the task must remain answerable, and the
unique normalized correct answer must stay equivalent to the original answer.

Important rules:
1. Use only virtual paths, such as /mnt/data/xxx.xlsx, /mnt/work/xxx.xlsx, or /mnt/output/xxx.xlsx.
2. Do not modify the original input files under /mnt/data.
3. If new files are needed, write them only under /mnt/output.
4. You may use the Python tool to inspect files, read headers, count rows, copy files, or modify files.
5. The final response must be exactly one valid JSON object. Do not output Markdown or explanatory prefixes/suffixes.
6. The question must not be changed, and the attack must not remove necessary evidence to make the task unanswerable.
7. If a high-quality answer-preserving sample cannot be constructed, explicitly return rejected instead of forcing an attack.
"""

# The judge never receives constructor claims or the constructor's conversation.
# Its prompt supplies a blind evidence packet and an isolated file snapshot.
JUDGE_SYSTEM_PROMPT = """You are an independent, blind judge for answer-preserving
table robustness attacks. You must form your conclusion from the original and
attacked file snapshots, the question, the reference answer, and the attack rule
only. The constructor's plan, claimed answer, edits, and self-check are not
available and must not be inferred.

Use Python to inspect both snapshots and independently recompute the relevant
analysis. Treat any claim you cannot verify from those files as unproven. Do not
modify source or attacked data: `/mnt/scratch` is the only permitted write area.
Prefer one comprehensive, bounded script and use no more than five tool calls;
additional calls are justified only when a prior result exposes a concrete error.
Return exactly one JSON object following the requested schema, without Markdown
or chain-of-thought.
"""

# Backward-compatible name for code importing the previous constant.
SYSTEM_PROMPT = CONSTRUCTOR_SYSTEM_PROMPT


class AgentError(RuntimeError):
    pass


class AgentFatalError(AgentError):
    pass


class AgentOutputError(AgentError):
    pass


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
        tool_timeout_sec: float = 180.0,
        max_tool_calls_per_stage: Optional[int] = None,
        system_prompt: str = CONSTRUCTOR_SYSTEM_PROMPT,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentFatalError("The openai package is required. Install it with `pip install openai`.") from exc

        if not api_key:
            raise AgentFatalError("Missing API key. Set OPENAI_API_KEY or config.model.api_key.")
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=float(request_timeout))
        self.model_name = model_name
        self.max_rounds = int(max_rounds)
        self.temperature = float(temperature)
        self.max_retries = max(1, int(max_retries))
        self.tool_timeout_sec = max(1.0, float(tool_timeout_sec))
        self.max_tool_calls_per_stage = _optional_positive_int(max_tool_calls_per_stage)
        self.system_prompt = system_prompt
        if max_tokens is None or str(max_tokens).strip().lower() in {"", "none", "null"}:
            self.max_tokens = None
        else:
            self.max_tokens = int(max_tokens)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                "description": "Execute Python code in an isolated subprocess for the current sample workspace. Each call starts a fresh process, so include all required imports and reread files. During construction, use /mnt/data, /mnt/work, /mnt/output, and /mnt/scratch; during validation, /mnt/original is also available.",
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
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        namespace: Dict[str, Any] = {}
        tools = self.tools if allow_tools and workspace is not None else None
        tool_call_count = 0

        for _round in range(1, self.max_rounds + 1):
            active_tools = tools
            if (
                tools is not None
                and self.max_tool_calls_per_stage is not None
                and tool_call_count >= self.max_tool_calls_per_stage
            ):
                active_tools = None
            message = self._chat(messages, active_tools)
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
                    result = self._handle_tool_call(tool_call, workspace, stage, namespace)
                    tool_call_count += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result,
                        }
                    )
                if (
                    self.max_tool_calls_per_stage is not None
                    and tool_call_count >= self.max_tool_calls_per_stage
                ):
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The Python tool budget for this stage is exhausted. "
                                "Use the evidence already collected and output the required final JSON now."
                            ),
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

    def run_code(self, code: str, workspace: Any, stage: str = "construct", namespace: Optional[Dict[str, Any]] = None) -> str:
        del namespace  # Tool calls are deliberately isolated so a hung call can be terminated.
        out = ""
        err = ""
        label = str(getattr(workspace, "new_id", "unknown"))
        script_path: Optional[Path] = None
        try:
            _check_code_safety(code)
            mapped = _map_virtual_paths(code, workspace.mapping(stage))
            scratch = workspace.tool_scratch(stage) if hasattr(workspace, "tool_scratch") else workspace.scratch
            scratch.mkdir(parents=True, exist_ok=True)
            script_path = scratch / f"tool_{stage}_{time.time_ns()}.py"
            script_path.write_text(mapped, encoding="utf-8")
            print(
                f"[{time.strftime('%H:%M:%S')}] TOOL {label} stage={stage} start "
                f"timeout={self.tool_timeout_sec:g}s script={script_path.name}",
                flush=True,
            )
            started = time.monotonic()
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                completed = subprocess.run(
                    [sys.executable, "-B", str(script_path)],
                    cwd=str(scratch),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.tool_timeout_sec,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                out = _timeout_text(exc.stdout)
                err = _timeout_text(exc.stderr)
                elapsed = time.monotonic() - started
                print(
                    f"[{time.strftime('%H:%M:%S')}] TOOL {label} stage={stage} timeout "
                    f"elapsed={elapsed:.1f}s script={script_path.name}",
                    flush=True,
                )
                text = (
                    f"Execution timed out after {self.tool_timeout_sec:g} seconds. "
                    "The child process was terminated. Use a simpler, bounded, vectorized check; "
                    "avoid exhaustive loops and reload all required state in the next tool call."
                )
                if out:
                    text += f"\nSTDOUT before timeout:\n{out}"
                if err:
                    text += f"\nSTDERR before timeout:\n{err}"
                return _truncate(text)

            out = completed.stdout or ""
            err = completed.stderr or ""
            elapsed = time.monotonic() - started
            print(
                f"[{time.strftime('%H:%M:%S')}] TOOL {label} stage={stage} done "
                f"exit={completed.returncode} elapsed={elapsed:.1f}s script={script_path.name}",
                flush=True,
            )
            if completed.returncode == 0:
                text = "Execution succeeded."
            else:
                text = (
                    f"Execution failed in isolated child process with exit code {completed.returncode}.\n"
                    "This is recoverable. Revise the code and include all imports and file loading in the next call."
                )
            if out:
                text += f"\nSTDOUT:\n{out}"
            if err:
                text += f"\nSTDERR:\n{err}"
            return _truncate(text)
        except Exception as exc:
            text = (
                f"Execution blocked or failed: {type(exc).__name__}: {exc}\n"
                "This is a recoverable sandbox/tool result. Revise your approach and continue. "
                "Do not delete or modify files under /mnt/data or /mnt/original. "
                "Write constructed files only under /mnt/output and declare the final package with output_files/input_file."
            )
            if out:
                text += f"\nSTDOUT:\n{out}"
            if err:
                text += f"\nSTDERR:\n{err}"
            return _truncate(text)

    def _chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise AgentFatalError("Empty choices from model response.")
                return response.choices[0].message
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise AgentFatalError(f"API call failed after retries: {last_error}")

    def _handle_tool_call(self, tool_call: Any, workspace: Any, stage: str, namespace: Dict[str, Any]) -> str:
        if tool_call.function.name != "execute_code":
            return "Error: unknown tool."
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return "Error: tool arguments are not valid JSON."
        code = args.get("code", "")
        if not code:
            return "Error: empty code."
        return self.run_code(code, workspace=workspace, stage=stage, namespace=namespace)


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
        mapped = mapped.replace(virtual, str(real.resolve()).replace("\\", "/"))
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


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _optional_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "0"}:
        return None
    return max(1, int(value))


def expand_env(value: Optional[str]) -> str:
    if value is None:
        return ""
    match = re.fullmatch(r"\$\{([^}]+)\}", str(value).strip())
    if match:
        return os.environ.get(match.group(1), "")
    return str(value)

