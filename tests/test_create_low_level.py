"""
Tests for the low-level create()/acreate() passthrough (autourgos_responses).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autourgos_responses import OpenAIResponse


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def test_create_input_override_cannot_hijack_validated_input_data():
    """
    Regression: an `input=` kwarg passed alongside `input_data=` used to
    silently replace the already-validated input_data via the unfiltered
    **overrides merge -- the ValueError check above it became meaningless.
    """
    llm = _make_response()
    llm._client = MagicMock()
    llm._client.responses.create.return_value = MagicMock()
    llm.create(input_data="validated prompt", input="SNUCK IN")
    kwargs = llm._client.responses.create.call_args.kwargs
    assert kwargs["input"] == "validated prompt"


def test_create_stream_override_is_ignored():
    """
    Regression: `stream=True` used to silently flip create() into streaming
    mode instead of the documented non-streaming response.
    """
    llm = _make_response()
    llm._client = MagicMock()
    llm._client.responses.create.return_value = MagicMock()
    llm.create(input_data="hi", stream=True)
    kwargs = llm._client.responses.create.call_args.kwargs
    assert kwargs["stream"] is False


def test_create_model_override_still_works():
    """model= must stay overridable -- only input/stream are protected."""
    llm = _make_response()
    llm._client = MagicMock()
    llm._client.responses.create.return_value = MagicMock()
    llm.create(input_data="hi", model="gpt-4o-mini")
    kwargs = llm._client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_acreate_input_and_stream_overrides_are_ignored():
    llm = _make_response()
    llm._async_client = MagicMock()
    llm._async_client.responses.create = AsyncMock(return_value=MagicMock())
    await llm.acreate(input_data="validated prompt", input="SNUCK IN", stream=True)
    kwargs = llm._async_client.responses.create.call_args.kwargs
    assert kwargs["input"] == "validated prompt"
    assert kwargs["stream"] is False
