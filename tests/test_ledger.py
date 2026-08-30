"""
Tests for the SQLite call ledger reused from autourgos-openaichat.
"""

import sqlite3
from unittest.mock import MagicMock, patch

from autourgos_responses import OpenAIResponse


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text="hello"):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def test_ledger_records_a_call(tmp_path):
    ledger_path = str(tmp_path / "ledger.db")
    llm = _make_response(ledger_path=ledger_path)
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj(), "primary"))

    llm.invoke("hello there")

    conn = sqlite3.connect(ledger_path)
    rows = conn.execute("SELECT model, provider_used, call_type, prompt, response FROM calls").fetchall()
    conn.close()

    assert len(rows) == 1
    model, provider_used, call_type, prompt, response = rows[0]
    assert provider_used == "primary"
    assert call_type == "invoke"
    assert prompt == "hello there"
    assert response == "hello"


def test_ledger_store_content_false_omits_text(tmp_path):
    ledger_path = str(tmp_path / "ledger.db")
    llm = _make_response(ledger_path=ledger_path, ledger_store_content=False)
    llm._create_across_providers = MagicMock(return_value=(_mock_response_obj(), "primary"))

    llm.invoke("secret prompt")

    conn = sqlite3.connect(ledger_path)
    rows = conn.execute("SELECT prompt, response FROM calls").fetchall()
    conn.close()

    assert rows[0] == (None, None)
