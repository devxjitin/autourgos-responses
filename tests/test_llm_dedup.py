"""
Tests confirming the LLM base-layer de-duplication between
autourgos-responses and autourgos-openaichat.

These tests do not make real network calls: the openai client construction
is mocked out.
"""

from unittest.mock import MagicMock, patch

import pytest

import autourgos_openaichat
import autourgos_responses
from autourgos_responses import OpenAIResponse
from autourgos_responses.llm import BaseLLM, CircuitBreakerOpenException
from autourgos_responses.response import OpenAIResponseConfigError, OpenAIResponseRedactionBlockedError


def _make_response(model="gpt-4o", **kwargs):
    """Construct an OpenAIResponse with the openai client construction mocked."""
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def test_baseLLM_is_same_class_object():
    """Proves real reuse, not just a copy: identity check."""
    assert autourgos_responses.BaseLLM is autourgos_openaichat.BaseLLM
    assert BaseLLM is autourgos_openaichat.BaseLLM


def test_openai_response_is_subclass_of_baseLLM_and_constructs():
    llm = _make_response()
    assert isinstance(llm, BaseLLM)
    assert isinstance(llm, autourgos_openaichat.BaseLLM)
    assert llm.model == "gpt-4o"


def test_circuit_breaker_trips_after_consecutive_failures():
    """
    Force several consecutive failures via a mocked client that raises,
    then confirm CircuitBreakerOpenException is raised on the next call.
    """
    llm = _make_response(circuit_failure_threshold=2, circuit_cooldown_time=30.0)

    # Make the underlying raw API call always raise a generic (non-whitelisted) error.
    llm._attempt_sync_create = MagicMock(side_effect=RuntimeError("boom"))

    # First failure.
    with pytest.raises(RuntimeError):
        llm.invoke("hello")
    # Second failure trips the breaker (threshold=2).
    with pytest.raises(RuntimeError):
        llm.invoke("hello")

    # Circuit should now be open — next call raises CircuitBreakerOpenException
    # instead of attempting the underlying call at all.
    with pytest.raises(CircuitBreakerOpenException):
        llm.invoke("hello")


def test_circuit_breaker_ignores_config_errors_as_non_transient():
    """
    Regression: a caller/config mistake (e.g. invoke_structured() with a
    non-Pydantic output_schema) must not trip the circuit breaker -- it's not
    a sign the provider is unhealthy, and previously counted as a failure,
    letting repeated config mistakes block unrelated, healthy invoke() calls
    on the same instance.
    """
    llm = _make_response(circuit_failure_threshold=1)
    with pytest.raises(OpenAIResponseConfigError):
        llm.invoke_structured("give me a number")  # output_schema=None -> ConfigError
    assert llm._consecutive_failures == 0

    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": "Paris"}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    llm._create_across_providers = MagicMock(
        return_value=(resp, "primary", llm._model_name, (None, None))
    )
    assert llm.invoke("hi") == "Paris"  # not blocked by a tripped circuit


def test_circuit_breaker_ignores_redaction_blocked_as_non_transient():
    """
    Regression: redact_mode="block" correctly refusing a PII-matching prompt
    is the redaction policy working as designed, not a provider failure --
    previously counted toward the circuit breaker, so a burst of legitimately
    blocked prompts could trip it and block unrelated, clean calls too.
    """
    llm = _make_response(redact_pii=True, redact_mode="block", circuit_failure_threshold=1)
    with pytest.raises(OpenAIResponseRedactionBlockedError):
        llm.invoke("my email is bob@example.com")
    assert llm._consecutive_failures == 0

    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": "ok"}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    llm._create_across_providers = MagicMock(
        return_value=(resp, "primary", llm._model_name, (None, None))
    )
    assert llm.invoke("hi") == "ok"  # not blocked by a tripped circuit
