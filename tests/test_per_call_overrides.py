"""
Regression tests for invoke()/ainvoke()/stream()/astream() accepting
per-call **overrides — mirrors the fix in autourgos-openaichat 2.3.0
(same closed-signature bug, same BaseLLM contract, shared by this package).

Without this fix, any caller passing extra keywords (e.g.
autourgos-agent's AgentLoopMixin, which injects per-iteration
overrides via an on_before_iteration middleware hook) hits
TypeError: invoke() got an unexpected keyword argument.
"""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from autourgos_responses import OpenAIResponse


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="ok"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


class _Event:
    def __init__(self, delta):
        self.type = "response.output_text.delta"
        self.delta = delta


def test_invoke_accepts_per_call_overrides():
    llm = _make_response(temperature=0.7)
    captured = {}

    def fake_attempt(client, params, label):
        captured.update(params)
        return _mock_response_obj("ok")

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hi", temperature=0.1, top_p=0.5)

    assert captured["temperature"] == 0.1  # per-call override wins
    assert captured["top_p"] == 0.5


def test_invoke_overrides_cannot_hijack_input_or_model():
    llm = _make_response()
    captured = {}

    def fake_attempt(client, params, label):
        captured.update(params)
        return _mock_response_obj("ok")

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hi", model="not-a-real-model", input="hijacked")

    assert captured["model"] == "gpt-4o"
    assert captured["input"] == "hi"


@pytest.mark.asyncio
async def test_ainvoke_accepts_per_call_overrides():
    llm = _make_response()
    captured = {}

    async def fake_attempt(client, params, label):
        captured.update(params)
        return _mock_response_obj("ok")

    llm._attempt_async_create = fake_attempt
    # Overrides are raw Responses API param names, not constructor kwarg
    # names — the Responses API field is "max_output_tokens", not "max_tokens"
    # (unlike Chat Completions, where autourgos-openaichat's equivalent test
    # uses "max_tokens" directly).
    await llm.ainvoke("hi", temperature=0.2, max_output_tokens=64)

    assert captured["temperature"] == 0.2
    assert captured["max_output_tokens"] == 64


def test_stream_accepts_per_call_overrides():
    llm = _make_response(temperature=0.7)
    client = MagicMock()
    captured = {}

    def fake_create(**params):
        captured.update(params)
        return iter([_Event("ok")])

    client.responses.create.side_effect = fake_create
    llm._client = client

    list(llm.stream("hi", temperature=0.1))
    assert captured["temperature"] == 0.1
