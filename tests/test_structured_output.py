"""
Tests for invoke_structured()/ainvoke_structured() validated structured output.
"""

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from autourgos_responses import OpenAIResponse
from autourgos_responses.core import build_text_config
from autourgos_responses.response import OpenAIResponseConfigError, OpenAIResponseValidationError


class Answer(BaseModel):
    value: int


def _make_response(model="gpt-4o", **kwargs):
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model=model, api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def test_strict_schema_requires_every_property_including_optional_and_defaulted():
    """
    Regression test (shared fix in autourgos_openaichat.core.
    enforce_additional_properties_false, reused here via build_text_config):
    OpenAI/Azure strict json_schema mode requires every key in `properties`
    to appear in `required`, but Pydantic's model_json_schema() only lists
    fields without a default -- Optional/defaulted fields used to be missing
    from `required`, which still 400s even with additionalProperties fixed.
    Covers both the top-level schema and a nested $defs model.
    """
    class Address(BaseModel):
        city: str
        zip_code: Optional[str] = None

    class Person(BaseModel):
        name: str
        nickname: str = "N/A"
        address: Optional[Address] = None

    text_config = build_text_config(output_schema=Person)
    schema = text_config["format"]["schema"]

    assert set(schema["required"]) == set(schema["properties"].keys())
    assert "nickname" in schema["required"]
    assert "address" in schema["required"]

    nested = schema["$defs"]["Address"]
    assert set(nested["required"]) == set(nested["properties"].keys())
    assert "zip_code" in nested["required"]


def test_requires_pydantic_output_schema():
    llm = _make_response()
    with pytest.raises(OpenAIResponseConfigError):
        llm.invoke_structured("give me a number")


def test_incompatible_with_streaming():
    llm = _make_response(output_schema=Answer, streaming=True)
    with pytest.raises(OpenAIResponseConfigError):
        llm.invoke_structured("give me a number")


def test_invoke_structured_returns_validated_instance():
    llm = _make_response(output_schema=Answer)
    llm._create_across_providers = MagicMock(
        return_value=(_mock_response_obj('{"value": 42}'), "primary", llm._model_name, (None, None))
    )

    result = llm.invoke_structured("give me a number")
    assert isinstance(result, Answer)
    assert result.value == 42
    assert llm.last_metadata["validation_retries"] == 0


def test_invoke_structured_retries_then_succeeds():
    llm = _make_response(output_schema=Answer, max_retries=1)
    bad = _mock_response_obj("not json")
    good = _mock_response_obj('{"value": 7}')
    llm._create_across_providers = MagicMock(
        side_effect=[
            (bad, "primary", llm._model_name, (None, None)),
            (good, "primary", llm._model_name, (None, None)),
        ]
    )

    result = llm.invoke_structured("give me a number", max_validation_retries=1)
    assert isinstance(result, Answer)
    assert result.value == 7
    assert llm.last_metadata["validation_retries"] == 1


def test_invoke_structured_raises_after_exhausting_retries():
    llm = _make_response(output_schema=Answer, max_retries=1)
    bad = _mock_response_obj("not json")
    llm._create_across_providers = MagicMock(return_value=(bad, "primary", llm._model_name, (None, None)))

    with pytest.raises(OpenAIResponseValidationError):
        llm.invoke_structured("give me a number", max_validation_retries=1)
