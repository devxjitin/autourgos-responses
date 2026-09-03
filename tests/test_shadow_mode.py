"""
Tests for shadow-mode dual dispatch ported from autourgos-openaichat.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import OpenAIResponse
from autourgos_responses.response import OpenAIResponseAPIError


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
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj("primary answer"), "primary", llm._model_name, (llm.input_pricing, llm.output_pricing)))

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
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj("primary answer"), "primary", llm._model_name, (llm.input_pricing, llm.output_pricing)))

    shadow_client = MagicMock()
    shadow_client.responses.create.side_effect = RuntimeError("shadow down")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    text = llm.invoke("hello")

    assert text == "primary answer"
    assert llm.last_shadow_results[0]["error"] is not None


def test_shadow_dispatch_runs_concurrently_with_primary():
    """
    Regression test: shadow dispatch used to only start after the primary
    call's entire budget-admission block (including its own network
    round-trip) had already finished -- sequential, not concurrent, despite
    the docstring/README claiming total latency is roughly
    max(primary, slowest shadow) rather than their sum. Both a slow primary
    and a slow shadow call here should now overlap.
    """
    llm = _make_response(shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}])
    delay = 0.2

    def slow_primary(**kw):
        time.sleep(delay)
        return _mock_response_obj("primary answer")

    def slow_shadow(**kw):
        time.sleep(delay)
        return _mock_response_obj("primary answer")

    llm._client = MagicMock()
    llm._client.responses.create.side_effect = slow_primary
    shadow_client = MagicMock()
    shadow_client.responses.create.side_effect = slow_shadow
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    start = time.perf_counter()
    text = llm.invoke("hello")
    elapsed = time.perf_counter() - start

    assert text == "primary answer"
    assert len(llm.last_shadow_results) == 1
    # Sequential would take ~2*delay; concurrent should stay well under that.
    assert elapsed < delay * 1.8, f"expected concurrent dispatch (~{delay}s), took {elapsed}s"


def test_shadow_still_fires_and_is_logged_when_primary_fails():
    """
    Regression test: since shadow dispatch now starts before the primary's
    own request (to achieve real concurrency), an in-flight shadow request
    can no longer be silently skipped if the primary ends up failing -- it's
    already been sent and can't be un-billed. Its result must still be
    collected and logged (with similarity=None), and the primary's own
    exception must still propagate correctly to the caller.
    """
    llm = _make_response(shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}], max_retries=1)
    llm._client = MagicMock()
    llm._client.responses.create.side_effect = RuntimeError("primary down")
    shadow_client = MagicMock()
    shadow_client.responses.create.return_value = _mock_response_obj("shadow-answer")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    with pytest.raises(OpenAIResponseAPIError):
        llm.invoke("hello")

    assert len(llm.last_shadow_results) == 1
    shadow = llm.last_shadow_results[0]
    assert shadow["response"] == "shadow-answer"
    assert shadow["similarity"] is None
    assert shadow["error"] is None


def test_shadow_cost_uses_its_own_pricing_not_the_primarys():
    """A shadow provider's cost must come from its own pricing, not the primary's."""
    llm = _make_response(
        input_pricing=1000, output_pricing=1000,
        shadow_providers=[{
            "model": "gpt-4o-mini", "api_key": "sk-shadow",
            "input_pricing": 1.0, "output_pricing": 2.0,
        }],
    )
    llm._create_across_providers = MagicMock(
        return_value=(_mock_response_obj("primary answer"), "primary", llm._model_name, (llm.input_pricing, llm.output_pricing))
    )

    shadow_resp = MagicMock()
    shadow_resp.output = [{"type": "message", "content": [{"text": "primary answer"}]}]
    shadow_resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    shadow_client = MagicMock()
    shadow_client.responses.create.return_value = shadow_resp
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    llm.invoke("hello")
    expected = (10 / 1_000_000) * 1.0 + (5 / 1_000_000) * 2.0
    assert llm.last_shadow_results[0]["total_cost"] == pytest.approx(expected)


def test_shadow_without_its_own_pricing_omits_cost():
    """A shadow entry with no pricing of its own must not silently borrow the
    primary's price for a different model — cost stays unset."""
    llm = _make_response(
        input_pricing=1000, output_pricing=1000,
        shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}],
    )
    llm._create_across_providers = MagicMock(
        return_value=(_mock_response_obj("primary answer"), "primary", llm._model_name, (llm.input_pricing, llm.output_pricing))
    )

    shadow_client = MagicMock()
    shadow_client.responses.create.return_value = _mock_response_obj("primary answer")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    llm.invoke("hello")
    assert llm.last_shadow_results[0]["total_cost"] is None


def test_on_shadow_result_callback_invoked():
    callback = MagicMock()
    llm = _make_response(
        shadow_providers=[{"model": "gpt-4o-mini", "api_key": "sk-shadow"}],
        on_shadow_result=callback,
    )
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj("primary answer"), "primary", llm._model_name, (llm.input_pricing, llm.output_pricing)))

    shadow_client = MagicMock()
    shadow_client.responses.create.return_value = _mock_response_obj("primary answer")
    llm._get_shadow_sync_client = MagicMock(return_value=shadow_client)

    llm.invoke("hello")

    callback.assert_called_once()
