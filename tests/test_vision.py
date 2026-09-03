"""
Tests for vision file encoding (build_multimodal_prompt / _encode_file_part).
"""

from unittest.mock import MagicMock, patch

import pytest

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


@pytest.mark.parametrize("ext,expected_mime", [
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("png", "image/png"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
])
def test_encode_file_mime_types(tmp_path, ext, expected_mime):
    """
    Regression test: every OpenAI-vision-supported extension must map to its
    correct IANA media type -- .jpg in particular used to become the invalid
    image/jpg instead of image/jpeg.
    """
    img = tmp_path / f"photo.{ext}"
    img.write_bytes(b"fakedata")
    llm = _make_response()
    captured = {}

    def fake_create_across_providers(params):
        captured["input"] = params["input"]
        return _mock_response_obj(), "primary", llm._model_name, (None, None)

    llm._create_across_providers = fake_create_across_providers
    llm.invoke("describe", files=[str(img)])

    image_url = captured["input"][0]["content"][1]["image_url"]
    assert image_url.startswith(f"data:{expected_mime};base64,")


def test_encode_file_unsupported_extension_warns_but_still_sends(tmp_path, caplog):
    """
    An unrecognized extension (e.g. a PDF -- this module has no dedicated
    support for it, vision-only) used to silently become a fabricated
    image/<ext> MIME type with zero warning. It must still be sent (not
    silently dropped), but now with a clear local warning.
    """
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    llm = _make_response()
    captured = {}

    def fake_create_across_providers(params):
        captured["input"] = params["input"]
        return _mock_response_obj(), "primary", llm._model_name, (None, None)

    llm._create_across_providers = fake_create_across_providers
    with caplog.at_level("WARNING"):
        llm.invoke("describe", files=[str(doc)])

    image_url = captured["input"][0]["content"][1]["image_url"]
    assert image_url.startswith("data:image/pdf;base64,")  # still sent, not silently dropped
    assert any("isn't a recognized image" in r.message for r in caplog.records)
