"""
Streaming + fallback interaction: fallback should only kick in if a target
fails before emitting any chunk.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autourgos_responses import CircuitBreakerOpenException, OpenAIResponse
from autourgos_responses.response import OpenAIResponseAPIError, OpenAIResponseDeadlineExceededError


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


class _Event:
    def __init__(self, delta):
        self.type = "response.output_text.delta"
        self.delta = delta


def test_stream_falls_back_before_any_chunk_emitted():
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fb"}])

    primary_client = MagicMock()
    primary_client.responses.create.side_effect = RuntimeError("primary stream failed to start")

    fallback_client = MagicMock()
    fallback_client.responses.create.return_value = iter([_Event("hello "), _Event("world")])

    llm._client = primary_client
    llm._get_fallback_sync_client = MagicMock(return_value=fallback_client)

    chunks = list(llm.stream("hi"))
    assert "".join(chunks) == "hello world"


def test_stream_does_not_fall_back_mid_response():
    """Once a chunk has been emitted, a mid-stream failure must propagate, not silently switch providers."""
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fb"}])

    def broken_stream(**kwargs):
        yield _Event("partial ")
        raise RuntimeError("connection dropped mid-stream")

    primary_client = MagicMock()
    primary_client.responses.create.side_effect = lambda **kw: broken_stream(**kw)
    llm._client = primary_client

    with pytest.raises(OpenAIResponseAPIError):
        list(llm.stream("hi"))


def test_stream_failures_trip_the_circuit_breaker():
    """
    Regression: stream()/astream() used to bypass the circuit breaker
    entirely -- a mid-stream failure never incremented consecutive_failures,
    so no number of failing stream() calls could ever trip it.
    """
    llm = _make_response(circuit_failure_threshold=2, max_retries=1)

    def failing_stream(**kwargs):
        raise ConnectionError("boom")
        yield  # pragma: no cover -- makes this a generator function

    client = MagicMock()
    client.responses.create.side_effect = lambda **kw: failing_stream(**kw)
    llm._client = client

    for _ in range(2):
        with pytest.raises(OpenAIResponseAPIError):
            list(llm.stream("hi"))
    assert llm._consecutive_failures == 2

    # Circuit is now open -- must raise immediately, without even attempting
    # to iterate (and thus without ever reaching the client).
    client.responses.create.reset_mock(side_effect=True)
    with pytest.raises(CircuitBreakerOpenException):
        llm.stream("hi")
    client.responses.create.assert_not_called()


def test_circuit_breaker_already_open_blocks_stream_before_iteration():
    """A circuit tripped by invoke() failures must also block stream() calls,
    instead of letting them keep hitting a known-down provider."""
    llm = _make_response(circuit_failure_threshold=1, max_retries=1)
    client = MagicMock()
    client.responses.create.side_effect = ConnectionError("boom")
    llm._client = client
    with pytest.raises(OpenAIResponseAPIError):
        llm.invoke("hi")

    client.responses.create.reset_mock(side_effect=True)
    with pytest.raises(CircuitBreakerOpenException):
        llm.stream("hi")  # raised by calling stream(), before any iteration
    client.responses.create.assert_not_called()


def test_astream_failures_trip_the_circuit_breaker():
    llm = _make_response(circuit_failure_threshold=2, max_retries=1)

    async def failing_astream():
        raise ConnectionError("boom")
        yield  # pragma: no cover -- makes this an async generator function

    async def fake_create(**kwargs):
        # `await client.responses.create(...)` resolves to the stream object,
        # which is then async-iterated -- so the async mock must itself
        # return the (not-yet-started) async generator, not be one.
        return failing_astream()

    async_client = MagicMock()
    async_client.responses.create = AsyncMock(side_effect=fake_create)
    llm._async_client = async_client

    async def run():
        return [c async for c in llm.astream("hi")]

    for _ in range(2):
        with pytest.raises(OpenAIResponseAPIError):
            asyncio.run(run())
    assert llm._consecutive_failures == 2

    async_client.responses.create.reset_mock(side_effect=True)

    async def run_after_trip():
        return [c async for c in llm.astream("hi")]

    with pytest.raises(CircuitBreakerOpenException):
        asyncio.run(run_after_trip())
    async_client.responses.create.assert_not_called()


def test_max_call_duration_applies_to_stream():
    llm = _make_response(max_call_duration=0.0)
    client = MagicMock()
    llm._client = client
    with pytest.raises(OpenAIResponseDeadlineExceededError):
        list(llm.stream("hi"))
    client.responses.create.assert_not_called()
