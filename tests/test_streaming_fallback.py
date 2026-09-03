"""
Streaming + fallback interaction: fallback should only kick in if a target
fails before emitting any chunk.
"""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autourgos_responses import BudgetExceededException, CircuitBreakerOpenException, OpenAIResponse
from autourgos_responses.response import OpenAIResponseAPIError, OpenAIResponseDeadlineExceededError


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


class _Event:
    def __init__(self, delta):
        self.type = "response.output_text.delta"
        self.delta = delta


class _Usage:
    def __init__(self, input_tokens=9, output_tokens=10, total_tokens=19):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class _CompletedEvent:
    """The terminal event carrying the full Response (with .usage) -- see
    extract_final_response_from_stream_event()."""
    def __init__(self, usage=None):
        self.type = "response.completed"
        self.response = MagicMock(usage=usage or _Usage())


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


def _read_ledger_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY id").fetchall()]
    conn.close()
    return rows


def test_invoke_streaming_tracks_budget_cost_and_ledger(tmp_path):
    """
    Regression test: invoke(streaming=True) used to bypass budget admission,
    cost tracking, and the ledger entirely -- max_session_cost was silently
    inert for streaming calls, and streaming calls never appeared in the
    audit ledger. The terminal response.completed event (which already
    carries the full Response with .usage on the Responses API, no
    stream_options opt-in needed) now lets the streaming path compute cost
    and log to the ledger exactly like the non-streaming path.
    """
    db_path = tmp_path / "calls.db"
    llm = _make_response(
        streaming=True, input_pricing=1000, output_pricing=1000,
        max_session_cost=0.01, ledger_path=str(db_path),
    )
    llm._client = MagicMock()
    llm._client.responses.create.return_value = iter([_Event("Hello"), _CompletedEvent()])

    assert llm.invoke("hi") == "Hello"
    assert llm.session_cost_used >= 0.01  # (9/1e6)*1000 + (10/1e6)*1000 = 0.019
    assert llm.last_metadata["total_tokens"] == 19

    row = _read_ledger_rows(db_path)[0]
    assert row["call_type"] == "invoke"
    assert row["total_tokens"] == 19

    with pytest.raises(BudgetExceededException):
        llm.invoke("hi again")


def test_invoke_streaming_restores_redacted_text():
    """redact_restore_in_response must apply on the streaming invoke() path too --
    it used to be skipped entirely, since streaming short-circuited before that code ran."""
    llm = _make_response(streaming=True, redact_pii=True, redact_categories=["email"],
                          redact_restore_in_response=True)
    llm._client = MagicMock()
    llm._client.responses.create.return_value = iter([
        _Event("Sure, noted: "), _Event("[REDACTED:email:1]"), _CompletedEvent(),
    ])
    result = llm.invoke("contact bob@example.com please")
    assert result == "Sure, noted: bob@example.com"


def test_invoke_streaming_without_usage_event_degrades_gracefully():
    """A provider that never sends response.completed still returns text;
    cost is simply left untracked for that call rather than crashing."""
    llm = _make_response(streaming=True, input_pricing=1000, output_pricing=1000)
    llm._client = MagicMock()
    llm._client.responses.create.return_value = iter([_Event("ok")])  # no terminal event

    assert llm.invoke("hi") == "ok"
    assert llm.last_metadata.get("total_cost") is None
    assert llm.session_cost_used == 0.0


@pytest.mark.asyncio
async def test_ainvoke_streaming_tracks_budget_cost_and_ledger(tmp_path):
    """Async counterpart of test_invoke_streaming_tracks_budget_cost_and_ledger."""
    db_path = tmp_path / "calls.db"
    llm = _make_response(
        streaming=True, input_pricing=1000, output_pricing=1000,
        max_session_cost=0.01, ledger_path=str(db_path),
    )

    async def fake_astream():
        yield _Event("Hello")
        yield _CompletedEvent()

    async def fake_create(**kwargs):
        return fake_astream()

    async_client = MagicMock()
    async_client.responses.create = AsyncMock(side_effect=fake_create)
    llm._async_client = async_client

    assert await llm.ainvoke("hi") == "Hello"
    assert llm.session_cost_used >= 0.01
    assert llm.last_metadata["total_tokens"] == 19

    row = _read_ledger_rows(db_path)[0]
    assert row["call_type"] == "ainvoke"
    assert row["total_tokens"] == 19

    with pytest.raises(BudgetExceededException):
        await llm.ainvoke("hi again")
