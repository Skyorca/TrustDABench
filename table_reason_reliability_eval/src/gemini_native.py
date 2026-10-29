"""Minimal Gemini generateContent adapter with OpenAI-like tool call objects."""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


class GeminiNativeError(RuntimeError):
    pass


class GeminiEmptyResponseError(GeminiNativeError):
    pass


@dataclass
class GeminiFunction:
    name: str
    arguments: str


@dataclass
class GeminiToolCall:
    id: str
    function: GeminiFunction
    type: str = "function"


@dataclass
class GeminiTurn:
    content: str
    tool_calls: List[GeminiToolCall]
    native_parts: List[Dict[str, Any]]
    finish_reason: str
    usage_metadata: Dict[str, Any]
    response_metadata: Dict[str, Any]

    def model_dump(self) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "_gemini_parts": copy.deepcopy(self.native_parts),
            "provider_metadata": copy.deepcopy(self.response_metadata),
        }
        if self.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return item


Transport = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


def is_gemini_model(model_name: str) -> bool:
    return "gemini" in str(model_name).lower()


class GeminiNativeClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float = 300.0,
        max_retries: int = 2,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        transport: Optional[Transport] = None,
    ):
        normalized_url = str(base_url or "").rstrip("/")
        if not normalized_url.endswith("/v1beta"):
            raise ValueError(
                "Gemini native base_url must end with '/v1beta', for example "
                "'https://api.tokenops.ai/v1beta'."
            )
        if not api_key:
            raise ValueError("Gemini native api_key is required.")
        if not model_name:
            raise ValueError("Gemini native model_name is required.")

        model_path = urllib.parse.quote(str(model_name), safe="-._")
        self.url = f"{normalized_url}/models/{model_path}:generateContent"
        self.api_key = api_key
        self.model_name = str(model_name)
        self.timeout = float(timeout)
        self.max_attempts = max(1, int(max_retries) + 1)
        self.max_output_tokens = int(max_output_tokens or 8192)
        self.temperature = temperature
        self.top_p = top_p
        self.transport = transport or _post_json

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> GeminiTurn:
        payload = self._build_payload(messages, tools)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                response = self.transport(self.url, headers, payload, self.timeout)
                return _parse_turn(response)
            except Exception as exc:
                last_error = exc
                if not _is_retryable(exc) or attempt == self.max_attempts - 1:
                    break
                time.sleep(min(2 ** attempt, 4))

        raise GeminiNativeError(
            f"Gemini native request failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        system_text, contents = _messages_to_contents(messages)
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            generation_config["temperature"] = float(self.temperature)
        if self.top_p is not None:
            generation_config["topP"] = float(self.top_p)

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        native_tools = _tools_to_native(tools or [])
        if native_tools:
            payload["tools"] = [{"functionDeclarations": native_tools}]
            payload["toolConfig"] = {
                "functionCallingConfig": {"mode": "AUTO"}
            }
        return payload


def _messages_to_contents(
    messages: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
    pending_function_responses: List[Dict[str, Any]] = []

    def flush_function_responses() -> None:
        if pending_function_responses:
            contents.append(
                {
                    "role": "user",
                    "parts": copy.deepcopy(pending_function_responses),
                }
            )
            pending_function_responses.clear()

    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            text = str(message.get("content") or "")
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            function_response = {
                "name": str(message.get("name") or "execute_code"),
                "response": {"output": str(message.get("content") or "")},
            }
            if call_id and not call_id.startswith("gemini_call_"):
                function_response["id"] = call_id
            pending_function_responses.append(
                {"functionResponse": function_response}
            )
            continue

        flush_function_responses()
        if role == "assistant":
            native_parts = message.get("_gemini_parts")
            if isinstance(native_parts, list) and native_parts:
                parts = copy.deepcopy(native_parts)
            else:
                parts = []
                if message.get("content"):
                    parts.append({"text": str(message["content"])})
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    try:
                        args = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    parts.append(
                        {
                            "functionCall": {
                                "name": str(function.get("name") or ""),
                                "args": args,
                            }
                        }
                    )
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": str(message.get("content") or "")}],
                }
            )

    flush_function_responses()
    return "\n\n".join(system_parts), contents


def _tools_to_native(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    declarations: List[Dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or {}
        if not function.get("name"):
            continue
        declarations.append(
            {
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": copy.deepcopy(
                    function.get("parameters") or {"type": "object"}
                ),
            }
        )
    return declarations


def _parse_turn(response: Dict[str, Any]) -> GeminiTurn:
    candidates = response.get("candidates") or []
    if not candidates:
        block_reason = (response.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise GeminiNativeError(f"Gemini prompt blocked: {block_reason}")
        raise GeminiEmptyResponseError("Gemini response has no candidates.")

    candidate = candidates[0] or {}
    parts = ((candidate.get("content") or {}).get("parts") or [])
    texts: List[str] = []
    tool_calls: List[GeminiToolCall] = []
    thought_parts_count = 0
    for index, part in enumerate(parts):
        text = part.get("text")
        if text and part.get("thought"):
            thought_parts_count += 1
        elif text:
            texts.append(str(text))
        function_call = part.get("functionCall")
        if isinstance(function_call, dict) and function_call.get("name"):
            call_id = str(function_call.get("id") or f"gemini_call_{index + 1}")
            tool_calls.append(
                GeminiToolCall(
                    id=call_id,
                    function=GeminiFunction(
                        name=str(function_call["name"]),
                        arguments=json.dumps(
                            function_call.get("args") or {},
                            ensure_ascii=False,
                        ),
                    ),
                )
            )

    if not texts and not tool_calls:
        raise GeminiEmptyResponseError(
            "Gemini candidate contains neither visible text nor functionCall "
            f"(finishReason={candidate.get('finishReason')!r}, "
            f"usageMetadata={response.get('usageMetadata') or {}})."
        )

    usage = copy.deepcopy(response.get("usageMetadata") or {})
    finish_reason = str(candidate.get("finishReason") or "")
    metadata = {
        "finishReason": finish_reason,
        "usageMetadata": usage,
        "safetyRatings": copy.deepcopy(candidate.get("safetyRatings") or []),
        "promptFeedback": copy.deepcopy(response.get("promptFeedback") or {}),
        "thoughtPartsCount": thought_parts_count,
    }
    return GeminiTurn(
        content="\n".join(texts).strip(),
        tool_calls=tool_calls,
        native_parts=copy.deepcopy(parts),
        finish_reason=finish_reason,
        usage_metadata=usage,
        response_metadata=metadata,
    )


def generated_token_count(turn: GeminiTurn) -> int:
    usage = turn.usage_metadata
    return int(usage.get("candidatesTokenCount") or 0) + int(
        usage.get("thoughtsTokenCount") or 0
    )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, GeminiEmptyResponseError):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 429} or exc.code >= 500
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    return False


def _post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        exc.args = (*exc.args, body[:2000])
        raise
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise GeminiNativeError("Gemini response is not a JSON object.")
    return value
