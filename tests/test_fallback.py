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
