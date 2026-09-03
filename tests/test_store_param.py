"""
Tests for the store= constructor setting (autourgos_responses.OpenAIResponse).
"""

from unittest.mock import MagicMock, patch

from autourgos_responses import OpenAIResponse


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="ok"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def _invoke_and_capture_params(llm, **invoke_kwargs):
    captured = {}

    def fake_create_across_providers(params):
        captured["params"] = params
        return _mock_response_obj(), "primary", llm._model_name, (None, None)

    llm._create_across_providers = fake_create_across_providers
    llm.invoke("hi", **invoke_kwargs)
    return captured["params"]


def test_store_true_is_sent_when_set_at_construction():
    llm = _make_response(store=True)
    params = _invoke_and_capture_params(llm)
    assert params["store"] is True


def test_store_false_is_sent_when_set_at_construction():
    llm = _make_response(store=False)
    params = _invoke_and_capture_params(llm)
    assert params["store"] is False


def test_store_omitted_by_default():
    """
    Regression: store used to be unreachable as a constructor setting at
    all. None (default) must omit the key entirely, not send a literal null.
    """
    llm = _make_response()
    params = _invoke_and_capture_params(llm)
    assert "store" not in params


def test_store_per_call_override_wins_over_constructor_default():
    llm = _make_response(store=True)
    params = _invoke_and_capture_params(llm, store=False)
    assert params["store"] is False
