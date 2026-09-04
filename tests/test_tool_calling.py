"""
Tests for native tool/function calling ported from autourgos-openaichat,
adapted to the Responses API's flat tool schema and function_call output items.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import BudgetExceededException, OpenAIResponse
from autourgos_responses.core import normalize_native_tool_calling_input
from autourgos_responses.llm import FunctionCall
from autourgos_responses.response import OpenAIResponseConfigError


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_tool_call_response():
    resp = MagicMock()
    resp.output = [
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
            "call_id": "call_123",
        }
    ]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def _mock_final_answer_response(text="Paris is sunny."):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


TOOLS = [{"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]


def test_invoke_with_tools_parses_function_call():
    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(_mock_tool_call_response(), "primary", llm._model_name, (None, None)))

    result = llm.invoke_with_tools("What's the weather in Paris?", TOOLS)

    assert result.has_tool_calls
    assert result.tool_calls == [FunctionCall(name="get_weather", arguments={"city": "Paris"}, call_id="call_123")]

    sent_params = llm._create_across_providers.call_args[0][0]
    assert sent_params["tools"][0] == {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }


def test_invoke_with_tools_malformed_arguments_json_falls_back_to_empty_dict():
    """Malformed `arguments` JSON used to silently become {} with no way to
    tell it apart from a genuinely empty-args call; arguments_parse_error
    now surfaces that distinction."""
    resp = MagicMock()
    resp.output = [
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": "{not valid json",
            "call_id": "call_999",
        }
    ]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(resp, "primary", llm._model_name, (None, None)))

    result = llm.invoke_with_tools("What's the weather?", TOOLS)

    assert result.has_tool_calls
    assert result.tool_calls[0].arguments == {}
    assert result.tool_calls[0].arguments_parse_error is not None


def test_invoke_with_tools_accepts_per_call_overrides():
    llm = _make_response(temperature=0.7)
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary", llm._model_name, (None, None)))

    llm.invoke_with_tools("hi", TOOLS, temperature=0.1, max_output_tokens=64)

    sent_params = llm._create_across_providers.call_args[0][0]
    assert sent_params["temperature"] == 0.1  # per-call override wins over constructor default
    assert sent_params["max_output_tokens"] == 64


def test_invoke_with_tools_returns_text_when_no_tool_call():
    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary", llm._model_name, (None, None)))

    result = llm.invoke_with_tools("What's the weather in Paris?", TOOLS)

    assert not result.has_tool_calls
    assert result.is_final_answer
    assert result.text == "Paris is sunny."


def test_invoke_with_tools_rejects_files_with_list_prompt():
    """
    Regression test: invoke_with_tools() must route list prompts through the
    same _resolve_prompt() validation as invoke() — not bypass it — so an
    invalid files+list combination is still rejected instead of silently
    reaching the API.
    """
    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary", llm._model_name, (None, None)))
    with pytest.raises(OpenAIResponseConfigError):
        llm.invoke_with_tools([{"role": "user", "content": "hi"}], TOOLS, files=["img.png"])
    llm._create_across_providers.assert_not_called()


def test_invoke_with_tools_missing_name_raises_config_error_not_key_error():
    """
    Regression: a tool dict missing "name" used to raise a raw, unhelpful
    KeyError instead of this library's own OpenAIResponseConfigError.
    """
    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary", llm._model_name, (None, None)))
    with pytest.raises(OpenAIResponseConfigError, match="index 0"):
        llm.invoke_with_tools("hi", [{"description": "missing its name key"}])
    llm._create_across_providers.assert_not_called()


def test_invoke_with_tools_rejects_empty_list_prompt():
    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary", llm._model_name, (None, None)))
    with pytest.raises(ValueError):
        llm.invoke_with_tools([], TOOLS)


# -- normalize_native_tool_calling_input -----------------------------------------
#
# Regression: autourgos-agent's native tool-calling loop builds a canonical,
# Chat-Completions-shaped message list ({"role": "assistant", "tool_calls":
# [...]}, {"role": "tool", "tool_call_id", "content"}) for ANY LLM wrapper
# implementing invoke_with_tools()/ainvoke_with_tools() -- it has no
# knowledge of the Responses API's very different item shapes. Passed
# straight through, the Responses API rejects these outright. This function
# is what invoke_with_tools()/ainvoke_with_tools() now run every list-shaped
# prompt through before sending it.

def test_normalize_converts_assistant_tool_calls_message():
    messages = [
        {"role": "user", "content": "what's 2+3?"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}},
            ],
        },
    ]
    result = normalize_native_tool_calling_input(messages)

    assert result[0] == {"role": "user", "content": "what's 2+3?"}
    assert result[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "add",
        "arguments": '{"a": 2, "b": 3}',
    }


def test_normalize_converts_tool_role_message():
    messages = [{"role": "tool", "tool_call_id": "call_1", "content": "5"}]
    result = normalize_native_tool_calling_input(messages)

    assert result == [{"type": "function_call_output", "call_id": "call_1", "output": "5"}]


def test_normalize_converts_multiple_tool_calls_in_one_assistant_message():
    """A single 'assistant' message can carry several concurrent tool
    calls (autourgos-agent's native loop supports concurrent tool
    execution in one turn) -- each must become its own function_call item."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "add", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "sub", "arguments": "{}"}},
            ],
        }
    ]
    result = normalize_native_tool_calling_input(messages)

    assert len(result) == 2
    assert result[0]["call_id"] == "c1" and result[0]["name"] == "add"
    assert result[1]["call_id"] == "c2" and result[1]["name"] == "sub"


def test_normalize_leaves_plain_messages_and_string_prompt_untouched():
    assert normalize_native_tool_calling_input("plain string prompt") == "plain string prompt"
    assert normalize_native_tool_calling_input(None) is None

    plain = [{"role": "user", "content": "hi"}, {"role": "system", "content": "be nice"}]
    assert normalize_native_tool_calling_input(plain) == plain


def test_normalize_leaves_already_responses_shaped_items_untouched():
    """A caller building its own Responses-API-native conversation directly
    (bypassing autourgos-agent's native loop) must not have its already-
    correct function_call/function_call_output items altered."""
    already_native = [
        {"type": "function_call", "call_id": "c1", "name": "add", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "5"},
    ]
    assert normalize_native_tool_calling_input(already_native) == already_native


def test_invoke_with_tools_sends_responses_shaped_input_not_chat_completions_shaped():
    """End-to-end: invoke_with_tools() itself must apply the conversion
    before sending, not just the standalone function in isolation."""
    llm = _make_response()
    llm._create_across_providers = MagicMock(
        return_value=(_mock_final_answer_response(), "primary", llm._model_name, (None, None))
    )

    messages = [
        {"role": "user", "content": "what's 2+3?"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "5"},
    ]
    llm.invoke_with_tools(messages, TOOLS)

    sent_input = llm._create_across_providers.call_args[0][0]["input"]
    assert not any(
        isinstance(item, dict) and (item.get("role") == "tool" or "tool_calls" in item)
        for item in sent_input
    )
    assert any(isinstance(item, dict) and item.get("type") == "function_call" for item in sent_input)
    assert any(isinstance(item, dict) and item.get("type") == "function_call_output" for item in sent_input)


# ── Sprint 0 regression: invoke_with_tools()/ainvoke_with_tools() used to
# bypass budget/ledger/redaction-restore entirely (native tool-calling mode
# called _create_across_providers() directly, skipping the same machinery
# invoke() goes through). ────────────────────────────────────────────────────

def test_budget_governor_blocks_invoke_with_tools_once_cap_reached():
    llm = _make_response(input_pricing=1_000_000.0, output_pricing=1_000_000.0, max_session_cost=10.0)
    llm._create_across_providers = MagicMock(
        return_value=(_mock_final_answer_response(), "primary", llm._model_name, (llm.input_pricing, llm.output_pricing))
    )
    llm.invoke_with_tools("hi", TOOLS)  # 1st call costs $10 (input) + $5 (output) = $15, over the $10 cap
    assert llm.session_cost_used > 0

    with pytest.raises(BudgetExceededException):
        llm.invoke_with_tools("hi again", TOOLS)
    assert llm._create_across_providers.call_count == 1  # 2nd call never reached the API


def test_budget_governor_blocks_ainvoke_with_tools_once_cap_reached():
    llm = _make_response(input_pricing=1_000_000.0, output_pricing=1_000_000.0, max_session_cost=10.0)

    async def fake_acreate(params):
        return _mock_final_answer_response(), "primary", llm._model_name, (llm.input_pricing, llm.output_pricing)

    llm._acreate_across_providers = fake_acreate

    async def run():
        await llm.ainvoke_with_tools("hi", TOOLS)
        with pytest.raises(BudgetExceededException):
            await llm.ainvoke_with_tools("hi again", TOOLS)

    import asyncio
    asyncio.run(run())


def test_ledger_records_invoke_with_tools(tmp_path):
    ledger_path = str(tmp_path / "ledger.db")
    llm = _make_response(ledger_path=ledger_path)
    llm._create_across_providers = MagicMock(
        return_value=(_mock_final_answer_response("Paris is sunny."), "primary", llm._model_name, (None, None))
    )
    llm.invoke_with_tools("Weather in Paris?", TOOLS)

    conn = sqlite3.connect(ledger_path)
    rows = conn.execute("SELECT call_type, response FROM calls").fetchall()
    conn.close()
    assert rows == [("invoke_with_tools", "Paris is sunny.")]


def test_ledger_records_ainvoke_with_tools(tmp_path):
    ledger_path = str(tmp_path / "ledger.db")
    llm = _make_response(ledger_path=ledger_path)

    async def fake_acreate(params):
        return _mock_final_answer_response("Berlin is cold."), "primary", llm._model_name, (None, None)

    llm._acreate_across_providers = fake_acreate

    async def run():
        return await llm.ainvoke_with_tools("Weather in Berlin?", TOOLS)

    import asyncio
    asyncio.run(run())

    conn = sqlite3.connect(ledger_path)
    rows = conn.execute("SELECT call_type, response FROM calls").fetchall()
    conn.close()
    assert rows == [("ainvoke_with_tools", "Berlin is cold.")]


def test_ledger_records_none_response_when_tool_called(tmp_path):
    """When the model calls a tool (no final-answer text), the ledger's
    response column should reflect that (None), not crash on it."""
    ledger_path = str(tmp_path / "ledger.db")
    llm = _make_response(ledger_path=ledger_path)
    llm._create_across_providers = MagicMock(
        return_value=(_mock_tool_call_response(), "primary", llm._model_name, (None, None))
    )
    llm.invoke_with_tools("Weather in Paris?", TOOLS)

    conn = sqlite3.connect(ledger_path)
    rows = conn.execute("SELECT call_type, response FROM calls").fetchall()
    conn.close()
    assert rows == [("invoke_with_tools", None)]


def test_redact_restore_in_response_applies_to_invoke_with_tools():
    """
    Regression: invoke_with_tools() discarded its redaction_map entirely, so
    redact_restore_in_response=True (working correctly in invoke()) silently
    had no effect in native tool-calling mode -- the caller got back the
    masked placeholder instead of the original text.
    """
    llm = _make_response(redact_pii=True, redact_restore_in_response=True)
    llm._create_across_providers = MagicMock(
        return_value=(_mock_final_answer_response("Sure, I'll email jane@example.com"), "primary", llm._model_name, (None, None))
    )
    result = llm.invoke_with_tools("contact jane@example.com please", TOOLS)
    assert result.text == "Sure, I'll email jane@example.com"


def test_redact_restore_in_response_applies_to_ainvoke_with_tools():
    llm = _make_response(redact_pii=True, redact_restore_in_response=True)

    async def fake_acreate(params):
        return _mock_final_answer_response("Sure, I'll email jane@example.com"), "primary", llm._model_name, (None, None)

    llm._acreate_across_providers = fake_acreate

    async def run():
        return await llm.ainvoke_with_tools("contact jane@example.com please", TOOLS)

    import asyncio
    result = asyncio.run(run())
    assert result.text == "Sure, I'll email jane@example.com"
