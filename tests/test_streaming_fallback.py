"""
Streaming + fallback interaction: fallback should only kick in if a target
fails before emitting any chunk.
"""

from unittest.mock import MagicMock, patch

import pytest

from autourgos_responses import OpenAIResponse
from autourgos_responses.response import OpenAIResponseAPIError


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
