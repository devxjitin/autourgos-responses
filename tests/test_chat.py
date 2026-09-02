"""
Tests for chat()/achat() routing through the same pipeline as invoke()/ainvoke()
— provider fallback, PII redaction, and the session budget cap all apply.
"""

from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import BudgetExceededException, OpenAIResponse
from autourgos_responses.response import (
    OpenAIResponseAllProvidersFailedError,
    OpenAIResponseAPIError,
    OpenAIResponseRedactionBlockedError,
)


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="ok", input_tokens=10, output_tokens=5):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
    return resp


def test_chat_returns_text():
    llm = _make_response()
    llm._create_across_providers = MagicMock(
        return_value=(_mock_response_obj("hi there"), "primary", llm._model_name, (None, None))
    )

    result = llm.chat([{"role": "user", "content": "hello"}])

    assert result == "hi there"
    assert llm.last_metadata["provider_used"] == "primary"


@pytest.mark.asyncio
async def test_achat_returns_text():
    llm = _make_response()

    async def fake_acreate_across_providers(params):
        return _mock_response_obj("hi async"), "primary", llm._model_name, (None, None)

    llm._acreate_across_providers = fake_acreate_across_providers

    result = await llm.achat([{"role": "user", "content": "hello"}])

    assert result == "hi async"
    assert llm.last_metadata["provider_used"] == "primary"


def test_chat_falls_back_to_secondary_provider_on_primary_failure():
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fallback"}])

    def fake_attempt(client, params, label, deadline=None):
        if label == "primary":
            raise OpenAIResponseAPIError("primary is down")
        return _mock_response_obj("from fallback")

    llm._attempt_sync_create = fake_attempt

    result = llm.chat([{"role": "user", "content": "hello"}])

    assert result == "from fallback"
    assert llm.last_metadata["provider_used"].startswith("fallback")


def test_chat_raises_when_all_providers_fail():
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fallback"}])
    llm._attempt_sync_create = MagicMock(side_effect=OpenAIResponseAPIError("down"))

    with pytest.raises(OpenAIResponseAllProvidersFailedError) as exc_info:
        llm.chat([{"role": "user", "content": "hello"}])
    assert len(exc_info.value.attempts) == 2


def test_chat_redacts_pii_before_sending():
    llm = _make_response(redact_pii=True)
    captured = {}

    def fake_create_across_providers(params):
        captured["input"] = params["input"]
        return _mock_response_obj(), "primary", llm._model_name, (None, None)

    llm._create_across_providers = fake_create_across_providers
    llm.chat([{"role": "user", "content": "my email is jane@example.com"}])

    sent_content = captured["input"][0]["content"]
    assert "jane@example.com" not in sent_content
    assert "[REDACTED:email]" in sent_content
    assert llm.last_redacted_categories == ["email"]


def test_chat_redact_mode_block_raises():
    llm = _make_response(redact_pii=True, redact_mode="block")
    llm._create_across_providers = MagicMock(
        return_value=(_mock_response_obj(), "primary", llm._model_name, (None, None))
    )

    with pytest.raises(OpenAIResponseRedactionBlockedError):
        llm.chat([{"role": "user", "content": "my email is jane@example.com"}])
    llm._create_across_providers.assert_not_called()


def test_chat_blocked_once_session_budget_exceeded():
    llm = _make_response(input_pricing=1_000_000.0, output_pricing=1_000_000.0, max_session_cost=100.0)
    llm._create_across_providers = MagicMock(
        return_value=(
            _mock_response_obj(input_tokens=100, output_tokens=50),
            "primary",
            llm._model_name,
            (llm.input_pricing, llm.output_pricing),
        )
    )

    # First call costs $100 (input) + $50 (output) = $150, exceeding the $100 cap.
    llm.chat([{"role": "user", "content": "hi"}])
    assert llm.session_cost_used > 0

    with pytest.raises(BudgetExceededException):
        llm.chat([{"role": "user", "content": "hi again"}])
