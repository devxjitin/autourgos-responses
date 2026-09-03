"""
Tests for strip_unsupported_sampling_params -- o-series reasoning models
reject temperature/top_p outright (400), so those params must be dropped
per-target rather than sent.
"""

from unittest.mock import MagicMock, patch

from autourgos_responses import OpenAIResponse
from autourgos_responses.response import OpenAIResponseAPIError


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="ok"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def test_o_series_primary_drops_temperature_and_top_p():
    llm = _make_response(model="o3-mini", temperature=0.7, top_p=0.9)
    captured = {}

    def fake_attempt(client, params, label, deadline=None):
        captured["params"] = params
        return _mock_response_obj()

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hi")
    assert "temperature" not in captured["params"]
    assert "top_p" not in captured["params"]


def test_regular_model_still_gets_temperature_and_top_p():
    llm = _make_response(temperature=0.7, top_p=0.9)
    captured = {}

    def fake_attempt(client, params, label, deadline=None):
        captured["params"] = params
        return _mock_response_obj()

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hi")
    assert captured["params"]["temperature"] == 0.7
    assert captured["params"]["top_p"] == 0.9


def test_fallback_to_o_series_drops_temperature_top_p_per_target():
    """
    The guard must apply per-target: a normal primary keeps temperature/
    top_p, but an o-series fallback must have them dropped from its own
    request even though the primary's params started out identical.
    """
    llm = _make_response(
        temperature=0.7, top_p=0.9,
        fallback_providers=[{"model": "o1-mini", "api_key": "sk-fallback"}],
    )
    captured = {}

    def fake_attempt(client, params, label, deadline=None):
        captured[label] = dict(params)
        if label == "primary":
            raise OpenAIResponseAPIError("primary down")
        return _mock_response_obj()

    llm._attempt_sync_create = fake_attempt
    llm.invoke("hi")

    assert captured["primary"]["temperature"] == 0.7
    assert captured["primary"]["top_p"] == 0.9
    assert "temperature" not in captured["fallback[0]:o1-mini"]
    assert "top_p" not in captured["fallback[0]:o1-mini"]
