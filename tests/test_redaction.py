"""
Tests for PII/secret redaction reused from autourgos-openaichat.
"""

from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import OpenAIResponse
from autourgos_responses.response import OpenAIResponseConfigError, OpenAIResponseRedactionBlockedError


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="ok"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def test_redact_pii_masks_outgoing_prompt():
    llm = _make_response(redact_pii=True)
    captured = {}

    def fake_create_across_providers(params):
        captured["input"] = params["input"]
        return _mock_response_obj(), "primary", llm._model_name, (None, None)

    llm._create_across_providers = fake_create_across_providers
    llm.invoke("my email is jane@example.com")

    assert "jane@example.com" not in captured["input"]
    assert "[REDACTED:email]" in captured["input"]
    assert llm.last_redacted_categories == ["email"]


def test_redact_mode_block_raises():
    llm = _make_response(redact_pii=True, redact_mode="block")
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj(), "primary", llm._model_name, (None, None)))

    with pytest.raises(OpenAIResponseRedactionBlockedError):
        llm.invoke("my email is jane@example.com")
    llm._create_across_providers.assert_not_called()


def test_invalid_custom_regex_raises_config_error_not_raw_re_error():
    """
    Regression: re.error is not a ValueError subclass, so a malformed
    redact_custom_patterns regex used to bypass the constructor's
    `except ValueError` guard entirely and leak out as a raw re.error
    instead of the library's own OpenAIResponseConfigError. Same root cause
    as autourgos-openaichat -- both packages share compile_patterns().
    """
    with pytest.raises(OpenAIResponseConfigError):
        _make_response(redact_pii=True, redact_custom_patterns={"bad": "(unbalanced"})


def test_no_redaction_when_disabled():
    llm = _make_response(redact_pii=False)
    captured = {}

    def fake_create_across_providers(params):
        captured["input"] = params["input"]
        return _mock_response_obj(), "primary", llm._model_name, (None, None)

    llm._create_across_providers = fake_create_across_providers
    llm.invoke("my email is jane@example.com")

    assert "jane@example.com" in captured["input"]
