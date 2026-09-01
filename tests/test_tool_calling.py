"""
Tests for native tool/function calling ported from autourgos-openaichat,
adapted to the Responses API's flat tool schema and function_call output items.
"""

from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import OpenAIResponse
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
    llm._create_across_providers = MagicMock(return_value=(_mock_tool_call_response(), "primary"))

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
    llm._create_across_providers = MagicMock(return_value=(resp, "primary"))

    result = llm.invoke_with_tools("What's the weather?", TOOLS)

    assert result.has_tool_calls
    assert result.tool_calls[0].arguments == {}
    assert result.tool_calls[0].arguments_parse_error is not None


def test_invoke_with_tools_accepts_per_call_overrides():
    llm = _make_response(temperature=0.7)
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary"))

    llm.invoke_with_tools("hi", TOOLS, temperature=0.1, max_output_tokens=64)

    sent_params = llm._create_across_providers.call_args[0][0]
    assert sent_params["temperature"] == 0.1  # per-call override wins over constructor default
    assert sent_params["max_output_tokens"] == 64


def test_invoke_with_tools_returns_text_when_no_tool_call():
    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary"))

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
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary"))
    with pytest.raises(OpenAIResponseConfigError):
        llm.invoke_with_tools([{"role": "user", "content": "hi"}], TOOLS, files=["img.png"])
    llm._create_across_providers.assert_not_called()


def test_invoke_with_tools_rejects_empty_list_prompt():
    llm = _make_response()
    llm._create_across_providers = MagicMock(return_value=(_mock_final_answer_response(), "primary"))
    with pytest.raises(ValueError):
        llm.invoke_with_tools([], TOOLS)
