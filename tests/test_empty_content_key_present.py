# SPDX-License-Identifier: Apache-2.0
"""Regression test for ``message.content`` key being dropped when empty.

Bug: ``routes/chat.py`` initialized ``final_content = None`` and only
populated it if the engine returned non-empty text. Combined with
``model_dump_json(exclude_none=True)`` on the response, the ``content``
field was silently DROPPED from the JSON response when the engine
emitted no content tokens (e.g. prefix-cache hit that sampled EOS first,
or a reasoning-only response).

OpenAI's spec requires ``message.content`` to always be present in
non-tool-call responses (empty string ``""`` when no text). Clients
keyed off ``choices[0].message.content`` (LangChain, Vercel AI SDK)
crashed with ``KeyError`` / ``AttributeError`` on the missing field.

Fix: when ``final_content is None and not tool_calls``, default to ``""``
so the key is always serialized.

Tool-call responses are exempt — OpenAI's spec permits ``content: null``
when ``tool_calls`` is populated.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _EmptyTextEngine:
    """Engine that returns finish_reason=stop with empty text (only EOS)."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="",
            new_text="",
            tokens=[2],  # EOS only
            prompt_tokens=8,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _client(engine_cls=_EmptyTextEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine_cls()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


class TestEmptyContentKeyPresent:
    """Bug C from iter15-hybrid onboarding: empty-text response missing
    ``content`` field entirely. Verify the key survives serialization."""

    def test_empty_text_response_includes_content_key(self):
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            assert resp.status_code == 200, resp.text
            msg = resp.json()["choices"][0]["message"]
            assert "content" in msg, (
                "OpenAI spec requires 'content' to always be present in "
                "non-tool-call responses (empty string when no text). "
                "Pre-fix, this key was dropped by exclude_none=True."
            )
            assert msg["content"] == ""
        finally:
            reset_config()

    def test_empty_text_response_role_still_present(self):
        """Sanity: role must also survive serialization."""
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            assert resp.status_code == 200
            msg = resp.json()["choices"][0]["message"]
            assert msg["role"] == "assistant"
        finally:
            reset_config()

    def test_tool_call_response_content_may_be_null(self, monkeypatch):
        """OpenAI exemption: when ``tool_calls`` is populated and the model
        emitted no content tokens, ``content`` is allowed to be omitted
        (matches OpenAI's own response shape). The fix MUST NOT clobber a
        legitimate tool-call response by forcing ``content: ""`` — that
        would imply the model produced an empty text response IN ADDITION
        to the tool call, which is a different (and rare) semantic.

        Pre-fix behavior already satisfied this (content stayed None →
        dropped by ``exclude_none=True``); the test pins it as an invariant
        so a future refactor moving the default doesn't silently break
        tool clients."""
        import vllm_mlx.routes.chat as chat_module

        # Sentinel: verify the monkeypatched parser actually fires. The
        # cfg.tool_parser=None default in _client() is meant for the empty-
        # text case where no parser config is needed; routes/chat.py:850
        # calls ``_parse_tool_calls_with_parser`` unconditionally today, so
        # the patch is exercised — but if a future refactor adds a
        # ``if cfg.tool_parser is not None`` guard, this counter goes to
        # zero and the test fails loudly instead of silently passing on the
        # wrong code path (DeepSeek round-2 finding #1).
        parse_call_count = {"n": 0}

        def _fake_parse(text, request):
            from vllm_mlx.api.models import FunctionCall, ToolCall

            parse_call_count["n"] += 1
            return (
                "",
                [
                    ToolCall(
                        id="call_x",
                        function=FunctionCall(name="get_weather", arguments="{}"),
                    )
                ],
            )

        monkeypatch.setattr(chat_module, "_parse_tool_calls_with_parser", _fake_parse)

        class _ToolCallEngine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="",
                    new_text="",
                    tokens=[10, 11, 12],
                    prompt_tokens=8,
                    completion_tokens=3,
                    finished=True,
                    finish_reason="stop",
                    channel=None,
                )

        client = _client(engine_cls=_ToolCallEngine)
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "max_tokens": 16,
                },
            )
            assert resp.status_code == 200, resp.text
            assert parse_call_count["n"] >= 1, (
                "monkeypatched parser must be invoked; if this fires, the "
                "production route added a guard that bypasses parsing and "
                "this whole test is testing the wrong code path."
            )
            msg = resp.json()["choices"][0]["message"]
            assert msg.get("tool_calls"), "tool_calls must be present"
            # Pinned invariant: ``content`` is OMITTED entirely (matches OpenAI
            # response shape — ``exclude_none=True`` drops the None field).
            # Use ``not in`` rather than ``.get(...) is None`` so a future
            # change from omitted to explicit ``null`` (or worse, ``""``)
            # fails the test loudly (DeepSeek round-2 finding #2).
            assert "content" not in msg, (
                f"tool_call response must omit content key entirely; "
                f"explicit null OR empty string would be a regression. "
                f"got: {msg.get('content')!r}"
            )
        finally:
            reset_config()

    def test_normal_text_response_unchanged(self):
        """The fix only affects the empty-text path; normal text round-trips."""

        class _NormalEngine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="hello world",
                    new_text="hello world",
                    tokens=[1, 2, 3],
                    prompt_tokens=8,
                    completion_tokens=3,
                    finished=True,
                    finish_reason="stop",
                    channel=None,
                )

        client = _client(engine_cls=_NormalEngine)
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            assert resp.status_code == 200
            msg = resp.json()["choices"][0]["message"]
            assert msg["content"] == "hello world"
        finally:
            reset_config()
