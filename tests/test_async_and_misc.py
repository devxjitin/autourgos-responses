"""
Deeper coverage: async paths, redact-restore, ledger shadow table, extra_body,
context-manager cleanup — for the features ported in this pass.
"""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from autourgos_responses import OpenAIResponse
from autourgos_responses.llm import FunctionCall
from autourgos_responses.response import (
    OpenAIResponseAllProvidersFailedError,
    OpenAIResponseAPIError,
    OpenAIResponseDeadlineExceededError,
)


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="ok"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def _mock_tool_call_response():
    resp = MagicMock()
    resp.output = [
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "Paris"}', "call_id": "call_1"}
    ]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


class Answer(BaseModel):
    value: int


TOOLS = [{"name": "get_weather", "description": "d", "parameters": {"type": "object", "properties": {}}}]


@pytest.mark.asyncio
async def test_ainvoke_basic():
    llm = _make_response()
    llm._acreate_across_providers = AsyncMock(
        return_value=(_mock_response_obj("async hi"), "primary", llm._model_name, (None, None))
    )
    text = await llm.ainvoke("hello")
    assert text == "async hi"


@pytest.mark.asyncio
async def test_ainvoke_with_tools_parses_function_call():
    llm = _make_response()
    llm._acreate_across_providers = AsyncMock(
        return_value=(_mock_tool_call_response(), "primary", llm._model_name, (None, None))
    )
    result = await llm.ainvoke_with_tools("weather?", TOOLS)
    assert result.has_tool_calls
    assert result.tool_calls == [FunctionCall(name="get_weather", arguments={"city": "Paris"}, call_id="call_1")]


@pytest.mark.asyncio
async def test_ainvoke_with_tools_accepts_per_call_overrides():
    llm = _make_response()
    llm._acreate_across_providers = AsyncMock(
        return_value=(_mock_response_obj("ok"), "primary", llm._model_name, (None, None))
    )
    await llm.ainvoke_with_tools("hi", TOOLS, temperature=0.2, max_output_tokens=64)
    sent_params = llm._acreate_across_providers.call_args[0][0]
    assert sent_params["temperature"] == 0.2
    assert sent_params["max_output_tokens"] == 64


@pytest.mark.asyncio
async def test_ainvoke_structured_returns_validated_instance():
    llm = _make_response(output_schema=Answer)
    llm._acreate_across_providers = AsyncMock(
        return_value=(_mock_response_obj('{"value": 9}'), "primary", llm._model_name, (None, None))
    )
    result = await llm.ainvoke_structured("give a number")
    assert isinstance(result, Answer)
    assert result.value == 9


@pytest.mark.asyncio
async def test_async_fallback_to_secondary_provider():
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fb"}])

    async def fake_attempt(client, params, label, deadline=None):
        if label == "primary":
            raise OpenAIResponseAPIError("primary down")
        return _mock_response_obj("from fallback")

    llm._attempt_async_create = fake_attempt
    text = await llm.ainvoke("hi")
    assert text == "from fallback"


@pytest.mark.asyncio
async def test_async_all_providers_failed():
    llm = _make_response(fallback_providers=[{"model": "gpt-4o-mini", "api_key": "sk-fb"}])
    llm._attempt_async_create = AsyncMock(side_effect=OpenAIResponseAPIError("down"))
    with pytest.raises(OpenAIResponseAllProvidersFailedError):
        await llm.ainvoke("hi")


@pytest.mark.asyncio
async def test_max_call_duration_applies_to_ainvoke():
    llm = _make_response(max_retries=3, backoff_factor=1.0, max_call_duration=0.05)
    async_client = MagicMock()
    async_client.responses.create = AsyncMock(side_effect=RuntimeError("boom"))
    llm._async_client = async_client
    with pytest.raises(OpenAIResponseDeadlineExceededError):
        await llm.ainvoke("hi")


@pytest.mark.asyncio
async def test_async_shadow_dispatch():
    llm = _make_response(shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}])
    llm._acreate_across_providers = AsyncMock(
        return_value=(_mock_response_obj("primary answer"), "primary", llm._model_name, (None, None))
    )

    shadow_client = MagicMock()
    shadow_client.responses.create = AsyncMock(return_value=_mock_response_obj("primary answer"))
    llm._get_shadow_async_client = MagicMock(return_value=shadow_client)

    text = await llm.ainvoke("hello")
    assert text == "primary answer"
    assert len(llm.last_shadow_results) == 1
    assert llm.last_shadow_results[0]["similarity"] == 1.0


@pytest.mark.asyncio
async def test_ainvoke_shadow_still_fires_and_is_logged_when_primary_fails():
    """
    Regression test: since shadow dispatch now starts before the primary's
    own request (to achieve real concurrency), an in-flight shadow request
    can no longer be silently skipped if the primary ends up failing. Its
    result must still be collected and logged, and the primary's own
    exception must still propagate correctly to the caller.
    """
    llm = _make_response(shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}], max_retries=1)
    llm._acreate_across_providers = AsyncMock(side_effect=RuntimeError("primary down"))

    shadow_client = MagicMock()
    shadow_client.responses.create = AsyncMock(return_value=_mock_response_obj("shadow-answer"))
    llm._get_shadow_async_client = MagicMock(return_value=shadow_client)

    with pytest.raises(RuntimeError):
        await llm.ainvoke("hello")

    assert len(llm.last_shadow_results) == 1
    shadow = llm.last_shadow_results[0]
    assert shadow["response"] == "shadow-answer"
    assert shadow["similarity"] is None


def test_redact_restore_in_response_restores_output_text():
    llm = _make_response(redact_pii=True, redact_restore_in_response=True)
    # Model echoes the placeholder back in its response.
    llm._create_across_providers = MagicMock(
        side_effect=lambda params: (
            _mock_response_obj(f"Sure, I noted {params['input']}"),
            "primary",
            llm._model_name,
            (None, None),
        )
    )
    text = llm.invoke("my email is jane@example.com")
    assert "jane@example.com" in text
    assert "[REDACTED:email]" not in text


@pytest.mark.asyncio
async def test_redact_restore_concurrent_ainvoke_never_cross_contaminates():
    """
    Regression test: _apply_redaction()/_resolve_prompt() used to write the
    redaction map to shared instance attributes (self._last_redaction_map),
    so two concurrent ainvoke() calls sharing one instance could restore
    using each other's mapping -- e.g. user A's response getting user B's
    secret spliced in if B's call happened to finish first. The fix makes
    the redaction map call-local (threaded through the return value), so
    each call must always restore its own secret regardless of completion
    order.
    """
    import asyncio

    llm = _make_response(redact_pii=True, redact_categories=["email"], redact_restore_in_response=True)

    async def fake_acreate(params):
        content = params["input"]
        # Whichever call is for "userA" resolves *last*, well after "userB" --
        # if redaction state were shared, userA's restore would race against
        # userB's already-overwritten mapping.
        await asyncio.sleep(0.05 if "userA" in content else 0.01)
        placeholder = "[REDACTED:email:1]"
        return _mock_response_obj(f"Confirming contact: {placeholder}"), "primary", llm._model_name, (None, None)

    llm._acreate_across_providers = fake_acreate

    results = await asyncio.gather(
        llm.ainvoke("userA email is alice-private@corp.internal"),
        llm.ainvoke("userB email is bob-private@corp.internal"),
    )
    assert results[0] == "Confirming contact: alice-private@corp.internal"
    assert results[1] == "Confirming contact: bob-private@corp.internal"
    assert "bob-private@corp.internal" not in results[0]
    assert "alice-private@corp.internal" not in results[1]


def test_ledger_records_shadow_calls_table(tmp_path):
    ledger_path = str(tmp_path / "ledger.db")
    llm = _make_response(
        ledger_path=ledger_path,
        shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}],
    )
    llm._create_across_providers = MagicMock(
        return_value=(_mock_response_obj("primary"), "primary", llm._model_name, (None, None))
    )
    shadow_client = MagicMock()
    shadow_client.responses.create.return_value = _mock_response_obj("primary")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    llm.invoke("hello")

    conn = sqlite3.connect(ledger_path)
    rows = conn.execute("SELECT provider_used, similarity FROM shadow_calls").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0].startswith("shadow")
    assert rows[0][1] == 1.0


def test_extra_body_merged_into_params():
    llm = _make_response(extra_body={"guided_json": {"type": "object"}})
    captured = {}

    def fake_create_across_providers(params):
        captured["params"] = params
        return _mock_response_obj(), "primary", llm._model_name, (None, None)

    llm._create_across_providers = fake_create_across_providers
    llm.invoke("hi")
    assert captured["params"]["extra_body"] == {"guided_json": {"type": "object"}}


def test_context_manager_closes_ledger(tmp_path):
    ledger_path = str(tmp_path / "ledger.db")
    llm = _make_response(ledger_path=ledger_path)
    assert llm._ledger_conn is not None
    with llm:
        pass
    assert llm._ledger_conn is None


def test_sync_context_manager_closes_client():
    llm = _make_response()
    llm._client = MagicMock()
    close_mock = llm._client.close
    with llm as ctx:
        assert ctx is llm
    close_mock.assert_called_once()
    assert llm._client is None


def test_async_context_manager_closes_both_clients():
    llm = _make_response()
    llm._client = MagicMock()
    llm._async_client = MagicMock()
    llm._async_client.aclose = AsyncMock()
    aclose_mock = llm._async_client.aclose

    async def run():
        async with llm as ctx:
            return ctx

    result = asyncio.run(run())
    assert result is llm
    aclose_mock.assert_called_once()
    assert llm._client is None
    assert llm._async_client is None


def test_close_method_releases_sync_client_without_with_statement():
    llm = _make_response()
    llm._client = MagicMock()
    close_mock = llm._client.close
    llm.close()
    close_mock.assert_called_once()
    assert llm._client is None


def test_aclose_method_releases_both_clients_without_async_with():
    llm = _make_response()
    llm._client = MagicMock()
    llm._async_client = MagicMock()
    llm._async_client.aclose = AsyncMock()
    aclose_mock = llm._async_client.aclose
    asyncio.run(llm.aclose())
    aclose_mock.assert_called_once()
    assert llm._client is None
    assert llm._async_client is None


def test_batch_invoke_runs_sequentially():
    llm = _make_response()
    llm._create_across_providers = MagicMock(
        side_effect=[
            (_mock_response_obj("a"), "primary", llm._model_name, (None, None)),
            (_mock_response_obj("b"), "primary", llm._model_name, (None, None)),
        ]
    )
    results = llm.batch_invoke(["p1", "p2"])
    assert results == ["a", "b"]
