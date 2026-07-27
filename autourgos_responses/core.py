"""
OpenAI Responses API helpers for autourgos-responses.

All utilities needed to configure clients, build Responses API request params,
handle multi-modal input, and parse streaming deltas.

The generic, byte-for-byte-identical helpers (module loading, API key/base
URL resolution, client construction/release, model name normalization) are
no longer duplicated here — they are re-exported from
``autourgos_openaichat.core`` so existing local imports
(``from .core import resolve_api_key`` etc.) keep working unchanged.
Responses-API-specific logic (reasoning/text config, multi-modal prompt
building, params building, streaming delta extraction) stays local because
it differs from the Chat Completions equivalents.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

from autourgos_openaichat.core import (
    configure_async_openai_client,
    configure_openai_client,
    load_openai_module,
    normalize_model_name,
    release_async_openai_client,
    release_openai_client,
    resolve_api_key,
    resolve_base_url,
)

# Kept local (not re-exported): the logger name is package-specific, so
# reusing openaichat's logger object would silently change the effective
# logger name used for autourgos-responses log records.
logger = logging.getLogger(__name__)

# ── Reasoning / text config ───────────────────────────────────────────────────

_VALID_REASONING_EFFORTS = {"low", "medium", "high"}
_VALID_TEXT_VERBOSITIES = {"concise", "detailed", "auto"}


def normalize_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    """Validate and normalize reasoning_effort. Returns None if not provided."""
    if effort is None:
        return None
    normalized = effort.strip().lower()
    if normalized not in _VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid reasoning_effort {effort!r}. "
            f"Must be one of: {sorted(_VALID_REASONING_EFFORTS)}"
        )
    return normalized


def normalize_text_verbosity(verbosity: Optional[str]) -> Optional[str]:
    """Validate and normalize text verbosity. Returns None if not provided."""
    if verbosity is None:
        return None
    normalized = verbosity.strip().lower()
    if normalized not in _VALID_TEXT_VERBOSITIES:
        raise ValueError(
            f"Invalid text_verbosity {verbosity!r}. "
            f"Must be one of: {sorted(_VALID_TEXT_VERBOSITIES)}"
        )
    return normalized


def build_reasoning_config(
    effort: Optional[str] = None,
    summary: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build the reasoning parameter dict for the Responses API."""
    if effort is None and summary is None:
        return None
    config: Dict[str, Any] = {}
    if effort is not None:
        config["effort"] = effort
    if summary is not None:
        config["summary"] = summary
    return config


def build_text_config(
    response_schema: Any = None,
    response_mime_type: Optional[str] = None,
    text_verbosity: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build the text parameter dict for the Responses API.

    Supports:
        - JSON schema output via response_schema
        - JSON mode via response_mime_type="application/json"
        - text verbosity hint
    """
    config: Dict[str, Any] = {}

    if response_schema is not None:
        schema_fn = getattr(response_schema, "model_json_schema", None)
        if callable(schema_fn):
            config["format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(response_schema, "__name__", "response"),
                    "schema": schema_fn(),
                    "strict": True,
                },
            }
        elif isinstance(response_schema, dict):
            config["format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                    "strict": True,
                },
            }
    elif response_mime_type and "json" in response_mime_type.lower():
        config["format"] = {"type": "json_object"}

    if text_verbosity is not None:
        config["verbosity"] = text_verbosity

    return config if config else None


# ── Multi-modal prompt building ───────────────────────────────────────────────

def _encode_file_part(file: Any) -> Optional[Dict[str, Any]]:
    """Encode a file into a Responses API image input part."""
    if isinstance(file, bytes):
        data = base64.b64encode(file).decode()
        return {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{data}",
        }
    if isinstance(file, str):
        try:
            with open(file, "rb") as fh:
                raw = fh.read()
            ext = file.rsplit(".", 1)[-1].lower() if "." in file else "png"
            mime = f"image/{ext}"
            data = base64.b64encode(raw).decode()
            return {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{data}",
            }
        except (OSError, IOError):
            # Treat as direct URL
            return {"type": "input_image", "image_url": file}
    if isinstance(file, dict):
        if "data" in file:
            raw = file["data"]
            if isinstance(raw, bytes):
                raw = base64.b64encode(raw).decode()
            mime = file.get("mime_type", "image/png")
            return {"type": "input_image", "image_url": f"data:{mime};base64,{raw}"}
        if "path" in file:
            return _encode_file_part(file["path"])
        if "url" in file:
            return {"type": "input_image", "image_url": file["url"]}
    return None


def build_multimodal_prompt(
    text: str,
    files: Optional[List[Any]] = None,
) -> Any:
    """
    Build the input for a Responses API request.

    - No files → returns plain string.
    - With files → returns a content-array list.
    """
    if not files:
        return text

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": text}]
    for f in files:
        part = _encode_file_part(f)
        if part is not None:
            content.append(part)

    return content


# ── Responses API params builder ──────────────────────────────────────────────

def build_response_create_params(
    model: str,
    input_data: Any,
    *,
    instructions: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    text: Optional[Dict[str, Any]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """Build the kwargs dict for client.responses.create()."""
    params: Dict[str, Any] = {
        "model": model,
        "input": input_data,
        "stream": stream,
    }
    if instructions:
        params["instructions"] = instructions
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    if reasoning is not None:
        params["reasoning"] = reasoning
    if text is not None:
        params["text"] = text
    return params


# ── Streaming delta extraction ────────────────────────────────────────────────

def extract_text_delta_from_event(event: Any) -> Optional[str]:
    """
    Extract incremental text from a Responses API streaming event.

    Responses API emits events with type like "response.output_text.delta".
    """
    # Responses API SSE events
    event_type = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)

    if event_type in ("response.output_text.delta", "response.text.delta"):
        delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else None)
        if isinstance(delta, str):
            return delta

    # Also handle content_block_delta style (defensive)
    if event_type == "content_block_delta":
        delta_obj = getattr(event, "delta", None)
        if delta_obj:
            text = getattr(delta_obj, "text", None)
            if isinstance(text, str):
                return text

    # Chat-style delta (if model ever returns it)
    choices = getattr(event, "choices", None)
    if choices:
        delta = getattr(choices[0], "delta", None)
        if delta:
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content

    if isinstance(event, dict):
        delta = event.get("delta")
        if isinstance(delta, str):
            return delta

    return None


__all__ = [
    "logger",
    "load_openai_module",
    "resolve_api_key",
    "resolve_base_url",
    "configure_openai_client",
    "configure_async_openai_client",
    "release_openai_client",
    "release_async_openai_client",
    "normalize_model_name",
    "normalize_reasoning_effort",
    "normalize_text_verbosity",
    "build_reasoning_config",
    "build_text_config",
    "build_multimodal_prompt",
    "build_response_create_params",
    "extract_text_delta_from_event",
]
