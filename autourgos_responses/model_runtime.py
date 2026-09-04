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

from typing import Any, Dict, List, Optional

from autourgos_openaichat.model_runtime import (
    coerce_prompt_variable,
    configure_runtime_environment,
    extract_template_fields,
    track_latency,
)
from autourgos_openaichat.model_runtime import (
    build_structured_output as _shared_build_structured_output,
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
    """Return a normalized dict with usage metadata and optional cost fields.

    Shared implementation in autourgos-openaichat's ``model_runtime.py``;
    binds this package's own ``extract_usage_metadata`` (Responses-API field
    layout) via ``usage_fn`` so payload-building logic isn't duplicated.
    """
    return _shared_build_structured_output(
        model_name=model_name,
        response_text=response_text,
        raw_response=raw_response,
        latency_ms=latency_ms,
        input_pricing=input_pricing,
        output_pricing=output_pricing,
        extra_fields=extra_fields,
        usage_fn=extract_usage_metadata,
    )


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_response(resp: Any) -> Optional[str]:
    """Extract generated text from an OpenAI Responses API response."""
    if resp is None:
        return None
    if isinstance(resp, str) and resp.strip():
        return resp

    # Responses API primary path: output[] list. A response can carry more than
    # one "message" item and each item's content[] can carry more than one text
    # part -- all of them belong to the final text, so every part is collected
    # and joined (matching how the streaming path joins deltas), instead of
    # returning only the first fragment found.
    output = resp.get("output") if isinstance(resp, dict) else getattr(resp, "output", None)
    if output:
        collected: List[str] = []
        for item in (output if isinstance(output, list) else []):
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type == "message":
                content_list = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if content_list:
                    for part in (content_list if isinstance(content_list, list) else []):
                        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            collected.append(text)
        if collected:
            return "".join(collected)

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
