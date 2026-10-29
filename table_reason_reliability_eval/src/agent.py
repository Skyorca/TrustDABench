from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from sandboxv2 import MyBox
from gemini_native import GeminiNativeClient, is_gemini_model
from utils import truncate_text


class AgentError(RuntimeError):
    pass


class AgentFatalError(AgentError):
    pass


class OpenAIToolAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        max_rounds: int = 12,
        max_tokens: Optional[int] = None,
        request_timeout: float = 120.0,
        max_retries: int = 2,
        tool_timeout_sec: int = 1200,
        tool_output_max_chars: Optional[int] = 12000,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        if not api_key:
            raise AgentFatalError("Missing API key. Set OPENAI_API_KEY or config.model.api_key.")
        if not model_name:
            raise AgentFatalError("Missing model name. Set config.model.name.")

        self.is_gemini = is_gemini_model(model_name)
        if self.is_gemini:
            self.client = None
            self.gemini_client = GeminiNativeClient(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                timeout=float(request_timeout),
                max_retries=max_retries,
                max_output_tokens=_normalize_optional_int(max_tokens) or 8192,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise AgentFatalError("The openai package is required. Install it with `pip install openai`.") from exc
            self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=float(request_timeout))
            self.gemini_client = None
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_rounds = int(max_rounds)
        self.max_tokens = _normalize_optional_int(max_tokens)
        self.max_retries = max(1, int(max_retries))
        self.tool_timeout_sec = int(tool_timeout_sec)
        self.tool_output_max_chars = _normalize_optional_int(tool_output_max_chars)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute Python code in a persistent sandbox whose current working directory contains the provided data files. Use print() to inspect outputs.",
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

    def preflight(self) -> None:
        messages = [
            {"role": "system", "content": self.system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": "Reply with exactly: ok"},
        ]
        message = self._chat(messages, tools=None)
        content = (getattr(message, "content", None) or "").strip().lower()
        if "ok" not in content:
            raise AgentFatalError(f"LLM preflight returned unexpected content: {content[:200]}")

    def run(self, prompt: str, working_dir: str) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        sandbox = MyBox(working_dir=working_dir, timeout=self.tool_timeout_sec)
        tool_events: List[Dict[str, Any]] = []
        raw_final_answer = ""
        final_answer = ""
        final_answer_source = ""
        finish_reason = ""
        error: Optional[str] = None

        try:
            for round_idx in range(1, self.max_rounds + 1):
                message = self._chat(messages, tools=self.tools)
                assistant_msg = _message_to_dict(message)
                messages.append(assistant_msg)

                tool_calls = getattr(message, "tool_calls", None) or []
                if tool_calls:
                    for tool_call in tool_calls:
                        output, event = self._execute_tool_call(tool_call, sandbox, round_idx)
                        tool_events.append(event)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": output,
                            }
                        )
                    continue

                raw_final_answer = getattr(message, "content", None) or ""
                final_answer, final_answer_source = _extract_final_answer(raw_final_answer)
                finish_reason = "assistant_final"
                break
            else:
                error = f"Too many model/tool rounds without final answer: {self.max_rounds}"
                finish_reason = "max_rounds"
        finally:
            try:
                sandbox.interpreter.cleanup_session(-1)
            except Exception:
                pass

        return {
            "raw_final_answer": raw_final_answer,
            "final_answer": final_answer,
            "final_answer_source": final_answer_source,
            "finish_reason": finish_reason,
            "messages": messages,
            "tool_events": tool_events,
            "tool_call_count": len(tool_events),
            "rounds": _count_assistant_rounds(messages),
            "error": error,
            "provider": "gemini_native" if self.is_gemini else "openai_compatible",
            "provider_metadata": [
                msg["provider_metadata"]
                for msg in messages
                if msg.get("role") == "assistant" and msg.get("provider_metadata")
            ],
        }

    def _execute_tool_call(self, tool_call: Any, sandbox: MyBox, round_idx: int) -> tuple[str, Dict[str, Any]]:
        event: Dict[str, Any] = {
            "round": round_idx,
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }
        if tool_call.function.name != "execute_code":
            output = "Error: unknown tool."
            event["output"] = output
            return output, event

        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            output = f"Error: tool arguments are not valid JSON: {exc}"
            event["output"] = output
            return output, event

        code = args.get("code", "")
        event["code"] = code
        if not code:
            output = "Error: empty code."
            event["output"] = output
            return output, event

        try:
            output = sandbox.run(code)
        except Exception as exc:
            output = f"Sandbox execution failed: {type(exc).__name__}: {exc}"
        output = truncate_text(str(output), self.tool_output_max_chars)
        event["output"] = output
        return output, event

    def _chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Any:
        if self.gemini_client is not None:
            try:
                return self.gemini_client.generate(messages, tools=tools)
            except Exception as exc:
                raise AgentFatalError(str(exc)) from exc

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise AgentFatalError("Empty choices from model response.")
                return response.choices[0].message
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise AgentFatalError(f"API call failed after retries: {last_error}")


def _message_to_dict(message: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {"role": "assistant", "content": getattr(message, "content", None) or ""}
    native_parts = getattr(message, "native_parts", None)
    if native_parts:
        item["_gemini_parts"] = native_parts
    metadata = getattr(message, "response_metadata", None)
    if metadata:
        item["provider_metadata"] = metadata
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        item["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return item


def _count_assistant_rounds(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for msg in messages if msg.get("role") == "assistant")


def _extract_final_answer(content: str) -> tuple[str, str]:
    match = re.search(r"<final_answer>(.*?)</final_answer>", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), "final_answer_tag"
    return content.strip(), "fallback_raw_content"


def _normalize_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return int(value)
