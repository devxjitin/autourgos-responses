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
import json
import logging
from typing import Any, Dict, List, Optional

from autourgos_openaichat import enforce_additional_properties_false
from autourgos_openaichat.core import (
    configure_async_openai_client,
    configure_openai_client,
    load_openai_module,
    model_requires_max_completion_tokens,
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

_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_VALID_TEXT_VERBOSITIES = {"low", "medium", "high"}


def strip_unsupported_sampling_params(params: Dict[str, Any], model_name: str) -> None:
    """
    Drop ``temperature``/``top_p`` from ``params`` in place if ``model_name``
    is an o-series reasoning model -- those models reject both params
    outright (400). Reuses ``model_requires_max_completion_tokens()`` from
    autourgos-openaichat.core for the model-family detection (same o-series
    regex, no need to duplicate it) but logs through this package's own
    logger, kept local for the same reason as ``logger`` above.

    Called at the same per-target sites as the model name is swapped in --
    params are built once before it's known which target (primary/
    fallback[i]/shadow[i]) will actually receive the request, and a
    fallback/shadow can be a different model family than the primary.

    Dropped rather than raised, so a caller with temperature/top_p set for a
    non-reasoning primary (with an o-series fallback configured, say) doesn't
    get a hard failure -- just a warning and the call proceeds without them.
    """
    if not model_requires_max_completion_tokens(model_name):
        return
    for key in ("temperature", "top_p"):
        if key in params:
            del params[key]
            logger.warning(
                "%s doesn't support %r -- dropped from the request instead of "
                "sending it and getting a 400 (o-series reasoning models reject "
                "temperature/top_p entirely).",
                model_name, key,
            )


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
    output_schema: Any = None,
    response_mime_type: Optional[str] = None,
    text_verbosity: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build the text parameter dict for the Responses API.

    Supports:
        - JSON schema output via output_schema
        - JSON mode via response_mime_type="application/json"
        - text verbosity hint
    """
    config: Dict[str, Any] = {}

    if output_schema is not None:
        schema_fn = getattr(output_schema, "model_json_schema", None)
        if callable(schema_fn):
            config["format"] = {
                "type": "json_schema",
                "name": getattr(output_schema, "__name__", "response"),
                "schema": enforce_additional_properties_false(schema_fn()),
                "strict": True,
            }
        elif isinstance(output_schema, dict):
            config["format"] = {
                "type": "json_schema",
                "name": "response",
                "schema": enforce_additional_properties_false(dict(output_schema)),
                "strict": True,
            }
    elif response_mime_type and "json" in response_mime_type.lower():
        config["format"] = {"type": "json_object"}

    if text_verbosity is not None:
        config["verbosity"] = text_verbosity

    return config if config else None


# ── Multi-modal prompt building ───────────────────────────────────────────────

# The only image formats OpenAI vision (Chat Completions and Responses API
# alike) actually supports -- everything else the provider will reject
# regardless of what MIME type we send. Kept in sync with the identical
# table in autourgos_openaichat.core (not imported from there, to avoid
# entangling this package's image encoding with openaichat's beyond what's
# already shared -- see that module's docstring for the full rationale).
_IMAGE_EXTENSION_MIME_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _guess_image_mime_type(file_path: str, ext: str) -> str:
    """
    Map a file extension to its correct image MIME type for vision input.
    See autourgos_openaichat.core._guess_image_mime_type for the full
    rationale (same fix, mirrored here).
    """
    mime = _IMAGE_EXTENSION_MIME_TYPES.get(ext)
    if mime is not None:
        return mime
    logger.warning(
        "_encode_file_part: %r has extension %r, which isn't a recognized "
        "image type (supported: png, jpg/jpeg, gif, webp). Sending as "
        "image/%s anyway, but the provider will likely reject it.",
        file_path, ext, ext,
    )
    return f"image/{ext}"


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
            mime = _guess_image_mime_type(file, ext)
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
    - With files → returns a list containing one user message item, since the
      Responses API's ``input`` expects message objects (``{"role", "content"}``),
      not bare content parts.
    """
    if not files:
        return text

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": text}]
    for f in files:
        part = _encode_file_part(f)
        if part is not None:
            content.append(part)

    return [{"role": "user", "content": content}]


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


# ── Tool/function calling ─────────────────────────────────────────────────────

def build_responses_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Autourgos tool dicts to the Responses API tool schema.

    Unlike Chat Completions (``{"type": "function", "function": {...}}``), the
    Responses API tool schema is flat: ``{"type": "function", "name", ...}``.
    """
    result: List[Dict[str, Any]] = []
    for i, t in enumerate(tools):
        if t.get("type") == "function" and "name" in t:
            result.append(t)
            continue
        if not t.get("name"):
            raise ValueError(f"Tool at index {i} is missing a 'name' key: {t!r}")
        result.append({
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
        })
    return result


def normalize_native_tool_calling_input(prompt: Any) -> Any:
    """
    Convert a Chat-Completions-shaped native-tool-calling message list into
    the item shapes the Responses API's ``input`` array actually expects.

    autourgos-agent's native tool-calling loop (``tool_calling_mode="native"``)
    is provider-agnostic and builds ONE canonical message format -- the same
    one Chat Completions (autourgos-openaichat) natively consumes:

        {"role": "assistant", "tool_calls": [{"id", "type": "function",
                                                "function": {"name", "arguments"}}]}
        {"role": "tool", "tool_call_id": ..., "content": ...}

    The Responses API has no such shapes at all -- it expects a flat
    ``function_call``/``function_call_output`` item per tool call/result
    instead of them being embedded in an "assistant"/"tool" message. Passed
    straight through unconverted, every one of these messages is rejected
    by the real API. This walks the list and rewrites exactly those two
    patterns into the Responses API's equivalents; a plain
    ``{"role": "user"/"system"/..., "content": ...}`` message (or anything
    already Responses-API-shaped, e.g. a caller building its own
    conversation directly against ``invoke_with_tools()``) passes through
    unchanged, since those ARE already valid Responses API input items --
    making this safe to apply unconditionally to any list-shaped prompt,
    not just ones that came from autourgos-agent's native loop.

    Non-list input (a plain string, or None) is returned unchanged.
    """
    if not isinstance(prompt, list):
        return prompt

    converted: List[Any] = []
    for msg in prompt:
        if not isinstance(msg, dict):
            converted.append(msg)
            continue

        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                converted.append({
                    "type": "function_call",
                    "call_id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments", "{}"),
                })
            continue

        if msg.get("role") == "tool":
            converted.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id"),
                "output": msg.get("content", ""),
            })
            continue

        converted.append(msg)

    return converted


def extract_tool_calls_from_response(resp: Any) -> List[Dict[str, Any]]:
    """
    Extract raw tool-call dicts from a Responses API response.

    Walks ``resp.output[]`` for items with ``type == "function_call"``, each
    carrying ``name``, ``arguments`` (a JSON string), and ``call_id``.
    Returns a list of ``{"name", "arguments", "call_id",
    "arguments_parse_error"}`` dicts with ``arguments`` already parsed from
    JSON (falling back to ``{}`` on a decode error -- ``arguments_parse_error``
    is set to the error message in that case so a mismatched/missing-args
    tool call doesn't silently look identical to a genuinely empty-args one).
    """
    calls: List[Dict[str, Any]] = []
    output = resp.get("output") if isinstance(resp, dict) else getattr(resp, "output", None)
    if not output:
        return calls
    for item in (output if isinstance(output, list) else []):
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "function_call":
            continue
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        raw_args = item.get("arguments") if isinstance(item, dict) else getattr(item, "arguments", None)
        call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
        parse_error = None
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError) as exc:
            arguments = {}
            parse_error = str(exc)
            logger.warning(
                "Malformed tool-call arguments for %r; treating as {}: %s",
                name or "<unknown>", exc,
            )
        calls.append({
            "name": name,
            "arguments": arguments,
            "call_id": call_id,
            "arguments_parse_error": parse_error,
        })
    return calls


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

    if isinstance(event, dict):
        delta = event.get("delta")
        if isinstance(delta, str):
            return delta

    return None


def extract_final_response_from_stream_event(event: Any) -> Optional[Any]:
    """
    Return the embedded, fully-populated ``Response`` object (the one that
    carries ``.usage``) from a terminal Responses API streaming event, or
    ``None`` for any other event.

    Unlike Chat Completions, the Responses API's ``response.completed``
    (and ``response.incomplete``, e.g. on a max_output_tokens truncation)
    event already carries the complete ``Response`` object -- no
    ``stream_options`` opt-in needed. This is how a streaming caller
    (``invoke(streaming=True)``) recovers token/cost data, since no
    delta-only event carries usage.
    """
    event_type = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)
    if event_type in ("response.completed", "response.incomplete"):
        response = getattr(event, "response", None) or (event.get("response") if isinstance(event, dict) else None)
        if response is not None:
            return response
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
    "strip_unsupported_sampling_params",
    "build_reasoning_config",
    "build_text_config",
    "build_multimodal_prompt",
    "build_response_create_params",
    "extract_text_delta_from_event",
    "extract_final_response_from_stream_event",
    "build_responses_tools",
    "extract_tool_calls_from_response",
    "normalize_native_tool_calling_input",
]
