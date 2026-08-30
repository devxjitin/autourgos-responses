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
