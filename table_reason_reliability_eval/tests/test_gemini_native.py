from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import OpenAIToolAgent
from gemini_native import GeminiNativeClient, GeminiNativeError, is_gemini_model


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run Python.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]


class GeminiNativeClientTests(unittest.TestCase):
    def test_payload_text_and_metadata(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload)
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "hidden reasoning", "thought": True},
                                {"text": "first"},
                                {"text": "second"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"candidatesTokenCount": 2},
            }

        client = GeminiNativeClient(
            api_key="secret",
            base_url="https://api.example/v1beta",
            model_name="Gemini-2.5-Pro",
            max_output_tokens=2048,
            temperature=0.1,
            top_p=0.8,
            transport=transport,
        )
        turn = client.generate(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            tools=TOOLS,
        )
        self.assertTrue(is_gemini_model("Gemini-2.5-Pro"))
        self.assertEqual("first\nsecond", turn.content)
        self.assertEqual(1, turn.response_metadata["thoughtPartsCount"])
        self.assertEqual("Bearer secret", captured["headers"]["Authorization"])
        self.assertEqual(
            2048,
            captured["payload"]["generationConfig"]["maxOutputTokens"],
        )
        self.assertEqual(
            "execute_code",
            captured["payload"]["tools"][0]["functionDeclarations"][0]["name"],
        )

    def test_function_response_preserves_thought_signature(self) -> None:
        payloads = []
        responses = iter(
            [
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "native-call-1",
                                            "name": "execute_code",
                                            "args": {"code": "print(1 + 1)"},
                                        },
                                        "thoughtSignature": "signature",
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "2"}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            ]
        )

        def transport(url, headers, payload, timeout):
            payloads.append(payload)
            return next(responses)

        client = GeminiNativeClient(
            api_key="secret",
            base_url="https://api.example/v1beta",
            model_name="gemini-2.5-pro",
            transport=transport,
        )
        messages = [{"role": "user", "content": "calculate"}]
        first = client.generate(messages, tools=TOOLS)
        messages.extend(
            [
                first.model_dump(),
                {
                    "role": "tool",
                    "name": "execute_code",
                    "tool_call_id": first.tool_calls[0].id,
                    "content": "2",
                },
            ]
        )
        second = client.generate(messages, tools=TOOLS)
        self.assertEqual("2", second.content)
        self.assertEqual(
            "signature",
            payloads[1]["contents"][1]["parts"][0]["thoughtSignature"],
        )
        self.assertEqual(
            "2",
            payloads[1]["contents"][2]["parts"][0]["functionResponse"]["response"]["output"],
        )
        self.assertEqual(
            "native-call-1",
            payloads[1]["contents"][2]["parts"][0]["functionResponse"]["id"],
        )

    def test_empty_response_is_not_success(self) -> None:
        attempts = 0

        def transport(url, headers, payload, timeout):
            nonlocal attempts
            attempts += 1
            return {"candidates": [{"content": {"parts": []}}]}

        client = GeminiNativeClient(
            api_key="secret",
            base_url="https://api.example/v1beta",
            model_name="gemini-2.5-pro",
            max_retries=1,
            transport=transport,
        )
        with patch("gemini_native.time.sleep"):
            with self.assertRaises(GeminiNativeError):
                client.generate([{"role": "user", "content": "question"}])
        self.assertEqual(2, attempts)

    def test_v1_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "v1beta"):
            GeminiNativeClient(
                api_key="secret",
                base_url="https://api.example/v1",
                model_name="gemini-2.5-pro",
            )


class GeminiToolAgentIntegrationTests(unittest.TestCase):
    def test_native_text_response_uses_existing_prediction_schema(self) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "<final_answer>done</final_answer>"}]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"candidatesTokenCount": 3},
        }
        with tempfile.TemporaryDirectory() as tmp:
            agent = OpenAIToolAgent(
                api_key="secret",
                base_url="https://api.example/v1beta",
                model_name="gemini-2.5-pro",
                system_prompt="system",
                max_rounds=2,
            )
            agent.gemini_client.transport = (
                lambda url, headers, payload, timeout: response
            )
            result = agent.run("question", tmp)

        self.assertEqual("done", result["final_answer"])
        self.assertEqual("gemini_native", result["provider"])
        self.assertEqual("STOP", result["provider_metadata"][0]["finishReason"])

    def test_native_function_call_uses_existing_tool_loop(self) -> None:
        responses = iter(
            [
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "execute_code",
                                            "args": {"code": "print(40 + 2)"},
                                        },
                                        "thoughtSignature": "tool-signature",
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "<final_answer>42</final_answer>"}
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
            ]
        )

        class FakeInterpreter:
            def cleanup_session(self, session_id):
                return None

        class FakeBox:
            def __init__(self, **kwargs):
                self.interpreter = FakeInterpreter()

            def run(self, code):
                return "42"

        with tempfile.TemporaryDirectory() as tmp:
            agent = OpenAIToolAgent(
                api_key="secret",
                base_url="https://api.example/v1beta",
                model_name="gemini-2.5-pro",
                system_prompt="system",
                max_rounds=3,
            )
            agent.gemini_client.transport = (
                lambda url, headers, payload, timeout: next(responses)
            )
            with patch("agent.MyBox", FakeBox):
                result = agent.run("calculate", tmp)

        self.assertEqual("42", result["final_answer"])
        self.assertEqual(1, result["tool_call_count"])
        self.assertEqual("42", result["tool_events"][0]["output"])
        self.assertEqual(2, result["rounds"])


if __name__ == "__main__":
    unittest.main()
