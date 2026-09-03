"""
Tests for extract_text_from_response (autourgos_responses.model_runtime).
"""

from autourgos_responses.model_runtime import extract_text_from_response


def test_single_message_single_text_part():
    resp = {"output": [{"type": "message", "content": [{"text": "hello"}]}]}
    assert extract_text_from_response(resp) == "hello"


def test_single_message_multiple_text_parts_are_joined():
    """
    Regression: a message's content[] can carry more than one text part.
    Used to return only the first part and silently drop the rest.
    """
    resp = {
        "output": [
            {
                "type": "message",
                "content": [{"text": "hello "}, {"text": "world"}],
            }
        ]
    }
    assert extract_text_from_response(resp) == "hello world"


def test_multiple_message_items_are_joined():
    """
    Regression: output[] can contain more than one "message" item (e.g. a
    "reasoning" item followed by one or more "message" items). Used to
    return only the first message's text and drop any subsequent messages.
    """
    resp = {
        "output": [
            {"type": "reasoning", "content": [{"text": "internal, not returned"}]},
            {"type": "message", "content": [{"text": "part one. "}]},
            {"type": "message", "content": [{"text": "part two."}]},
        ]
    }
    assert extract_text_from_response(resp) == "part one. part two."


def test_empty_and_whitespace_text_parts_are_skipped():
    resp = {
        "output": [
            {
                "type": "message",
                "content": [{"text": ""}, {"text": "   "}, {"text": "real text"}],
            }
        ]
    }
    assert extract_text_from_response(resp) == "real text"


def test_falls_back_to_output_text_when_output_has_no_text():
    resp = {"output": [{"type": "reasoning", "content": []}], "output_text": "fallback"}
    assert extract_text_from_response(resp) == "fallback"


def test_none_response_returns_none():
    assert extract_text_from_response(None) is None


def test_plain_string_response_returned_as_is():
    assert extract_text_from_response("already text") == "already text"
