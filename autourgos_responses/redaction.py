"""
Best-effort PII / secret redaction for autourgos-responses.

Re-exports the (API-agnostic) redaction implementation from
autourgos-openaichat instead of duplicating it.
"""

from __future__ import annotations

from autourgos_openaichat.redaction import (
    DEFAULT_PATTERNS,
    compile_patterns,
    load_patterns_file,
    redact_text,
    redact_value,
    restore_text,
    terms_to_patterns,
)

__all__ = [
    "DEFAULT_PATTERNS",
    "compile_patterns",
    "terms_to_patterns",
    "load_patterns_file",
    "redact_text",
    "redact_value",
    "restore_text",
]
