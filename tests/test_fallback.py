"""
Tests for the provider fallback chain ported from autourgos-openaichat.
"""

from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import OpenAIResponse
from autourgos_responses.response import OpenAIResponseAllProvidersFailedError, OpenAIResponseAPIError


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="ok"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def test_falls_back_to_secondary_provider_on_primary_failure():
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fallback"}])

    call_count = {"n": 0}

    def fake_attempt(client, params, label):
        call_count["n"] += 1
        if label == "primary":
            raise OpenAIResponseAPIError("primary is down")
        return _mock_response_obj("from fallback")

    llm._attempt_sync_create = fake_attempt

    text = llm.invoke("hello")
    assert text == "from fallback"
    assert llm.last_metadata["provider_used"].startswith("fallback")


def test_all_providers_failed_raises():
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fallback"}])
    llm._attempt_sync_create = MagicMock(side_effect=OpenAIResponseAPIError("down"))

    with pytest.raises(OpenAIResponseAllProvidersFailedError) as exc_info:
        llm.invoke("hello")
    assert len(exc_info.value.attempts) == 2


def test_missing_model_key_in_fallback_config_raises():
    from autourgos_responses.response import OpenAIResponseConfigError

    with pytest.raises(OpenAIResponseConfigError):
        _make_response(fallback_providers=[{"api_key": "sk-no-model"}])


def test_fallback_metadata_reports_its_own_model_not_the_primarys():
    """Regression: llm.last_metadata['model'] must reflect whichever provider
    actually answered, not always the primary's model name."""
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fallback"}])

    def fake_attempt(client, params, label):
        if label == "primary":
            raise OpenAIResponseAPIError("primary is down")
        return _mock_response_obj("from fallback")

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hello")
    assert llm.last_metadata["model"] == "gpt-4o-mini"


def test_fallback_cost_uses_its_own_pricing_not_the_primarys():
    """Regression: cost for a fallback response must come from that fallback
    entry's own pricing, not the primary's (different model, different price)."""
    llm = _make_response(
        input_pricing=1000, output_pricing=1000,
        fallback_providers=[
            {"model": "gpt-4o-mini", "api_key": "sk-fallback", "input_pricing": 1.0, "output_pricing": 2.0}
        ],
    )

    def fake_attempt(client, params, label):
        if label == "primary":
            raise OpenAIResponseAPIError("primary is down")
        resp = MagicMock()
        resp.output = [{"type": "message", "content": [{"text": "from fallback"}]}]
        resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return resp

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hello")
    expected = (10 / 1_000_000) * 1.0 + (5 / 1_000_000) * 2.0
    assert llm.last_metadata["total_cost"] == pytest.approx(expected)


def test_fallback_without_its_own_pricing_omits_cost():
    """A fallback entry with no pricing of its own must not silently borrow
    the primary's price for a different model — cost fields stay unset."""
    llm = _make_response(
        input_pricing=1000, output_pricing=1000,
        fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fallback"}],
    )

    def fake_attempt(client, params, label):
        if label == "primary":
            raise OpenAIResponseAPIError("primary is down")
        return _mock_response_obj("from fallback")

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hello")
    assert "total_cost" not in llm.last_metadata
