"""
Tests for shadow-mode dual dispatch ported from autourgos-openaichat.
"""

from unittest.mock import MagicMock, patch

from autourgos_responses import OpenAIResponse


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def test_shadow_dispatch_populates_last_shadow_results():
    llm = _make_response(shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}])
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj("primary answer"), "primary"))

    shadow_client = MagicMock()
    shadow_client.responses.create.return_value = _mock_response_obj("primary answer")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    text = llm.invoke("hello")

    assert text == "primary answer"
    assert len(llm.last_shadow_results) == 1
    result = llm.last_shadow_results[0]
    assert result["provider_used"].startswith("shadow")
    assert result["similarity"] == 1.0
    assert result["error"] is None


def test_shadow_dispatch_records_error_without_failing_primary():
    llm = _make_response(shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}])
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj("primary answer"), "primary"))

    shadow_client = MagicMock()
    shadow_client.responses.create.side_effect = RuntimeError("shadow down")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    text = llm.invoke("hello")

    assert text == "primary answer"
    assert llm.last_shadow_results[0]["error"] is not None


def test_on_shadow_result_callback_invoked():
    callback = MagicMock()
    llm = _make_response(
        shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}],
        on_shadow_result=callback,
    )
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj("primary answer"), "primary"))

    shadow_client = MagicMock()
    shadow_client.responses.create.return_value = _mock_response_obj("primary answer")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    llm.invoke("hello")

    callback.assert_called_once()
