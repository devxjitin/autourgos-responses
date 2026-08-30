"""
Tests for the budget governor ported from autourgos-openaichat's BaseLLM.
"""

from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import BudgetExceededException, OpenAIResponse
from autourgos_responses.response import OpenAIResponseConfigError


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(input_tokens=100, output_tokens=50, text="hello"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
    return resp


def test_max_session_cost_requires_pricing():
    with pytest.raises(OpenAIResponseConfigError):
        _make_response(max_session_cost=1.0)


def test_budget_exceeded_after_cap_reached():
    llm = _make_response(input_pricing=1_000_000.0, output_pricing=1_000_000.0, max_session_cost=100.0)
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj(), "primary"))

    # First call costs $100 (input) + $50 (output) = $150, exceeding the $100 cap.
    llm.invoke("hi")
    assert llm.session_cost_used > 0

    with pytest.raises(BudgetExceededException):
        llm.invoke("hi again")


def test_reset_session_budget_unblocks():
    llm = _make_response(input_pricing=1_000_000.0, output_pricing=1_000_000.0, max_session_cost=100.0)
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj(), "primary"))

    llm.invoke("hi")
    with pytest.raises(BudgetExceededException):
        llm.invoke("hi again")

    llm.reset_session_budget()
    assert llm.session_cost_used == 0.0
    # Should not raise now.
    llm.invoke("hi once more")
