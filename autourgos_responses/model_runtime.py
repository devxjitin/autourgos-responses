"""
Shared model runtime helpers for autourgos-responses.

``track_latency``, ``extract_template_fields``, ``coerce_prompt_variable``,
and ``configure_runtime_environment`` are byte-for-byte identical to the
autourgos-openaichat versions and are re-exported from there instead of
being duplicated. ``extract_usage_metadata``, ``extract_text_from_response``,
and ``build_structured_output`` stay local: the first two have different
field-priority ordering for the Responses API, and ``build_structured_output``
internally calls the local ``extract_usage_metadata``, so importing it from
openaichat would silently change its behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from autourgos_openaichat.model_runtime import (
    coerce_prompt_variable,
    configure_runtime_environment,
    extract_template_fields,
    track_latency,
)

# ── Token/usage extraction ────────────────────────────────────────────────────

def extract_usage_metadata(resp: Any) -> Dict[str, Optional[int]]:
    """Extract token counts from an OpenAI Responses API object."""
    if resp is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    usage = None
    if isinstance(resp, dict):
        usage = resp.get("usage") or resp.get("usage_metadata")
    else:
        usage = getattr(resp, "usage", None) or getattr(resp, "usage_metadata", None)

    if usage is not None:
        def _get(obj: Any, *keys: str) -> Optional[int]:
            for k in keys:
                v = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
                if v is not None:
                    return int(v)
            return None

        # Responses API uses input_tokens / output_tokens directly
        return {
            "input_tokens": _get(usage, "input_tokens", "prompt_tokens", "prompt_token_count"),
            "output_tokens": _get(usage, "output_tokens", "completion_tokens", "candidates_token_count"),
            "total_tokens": _get(usage, "total_tokens", "total_token_count"),
        }

    return {"input_tokens": None, "output_tokens": None, "total_tokens": None}


# ── Structured response payload ───────────────────────────────────────────────

def build_structured_output(
    *,
    model_name: str,
    response_text: str,
    raw_response: Any,
    latency_ms: Optional[float] = None,
    input_pricing: Optional[float] = None,
    output_pricing: Optional[float] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a normalized dict with usage metadata and optional cost fields."""
    usage = extract_usage_metadata(raw_response)

    payload: Dict[str, Any] = {
        "model": model_name,
        "response": response_text,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
    }

    if input_pricing is not None and usage["input_tokens"] is not None:
        payload["input_cost"] = (usage["input_tokens"] / 1_000_000) * input_pricing
    if output_pricing is not None and usage["output_tokens"] is not None:
        payload["output_cost"] = (usage["output_tokens"] / 1_000_000) * output_pricing
    if "input_cost" in payload and "output_cost" in payload:
        payload["total_cost"] = payload["input_cost"] + payload["output_cost"]
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if extra_fields:
        payload.update(extra_fields)

    return payload


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_response(resp: Any) -> Optional[str]:
    """Extract generated text from an OpenAI Responses API response."""
    if resp is None:
        return None
    if isinstance(resp, str) and resp.strip():
        return resp

    # Responses API primary path: output[] list
    output = resp.get("output") if isinstance(resp, dict) else getattr(resp, "output", None)
    if output:
        for item in (output if isinstance(output, list) else []):
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type == "message":
                content_list = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if content_list:
                    for part in (content_list if isinstance(content_list, list) else []):
                        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            return text

    # Shortcut: output_text attribute
    output_text = resp.get("output_text") if isinstance(resp, dict) else getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Fallback: Chat-style choices[] (should not appear in Responses API, but defensive)
    choices = resp.get("choices") if isinstance(resp, dict) else getattr(resp, "choices", None)
    if choices:
        first = choices[0]
        msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        if msg:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content

    if isinstance(resp, dict):
        for key in ("text", "delta", "content"):
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                return val

    return None


__all__ = [
    "track_latency",
    "extract_usage_metadata",
    "build_structured_output",
    "extract_text_from_response",
    "extract_template_fields",
    "coerce_prompt_variable",
    "configure_runtime_environment",
]
