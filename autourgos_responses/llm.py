"""
BaseLLM — base interface for all autourgos-responses model wrappers.

This module no longer duplicates the shared LLM base layer. It re-exports
BaseLLM, FunctionCall, ToolCallResponse, and CircuitBreakerOpenException
from autourgos-openaichat, which owns the canonical implementation
(including the circuit-breaker wrapping logic). This keeps
`from autourgos_responses.llm import BaseLLM` (and similar) working
unchanged for existing callers, while eliminating the maintenance burden
of keeping two identical copies in sync.
"""

from __future__ import annotations

from autourgos_openaichat import BaseLLM, CircuitBreakerOpenException, FunctionCall, ToolCallResponse

__all__ = [
    "BaseLLM",
    "FunctionCall",
    "ToolCallResponse",
    "CircuitBreakerOpenException",
]
