"""
autourgos-responses
===================
Self-contained LLM wrapper for the OpenAI Responses API.

Quick start::

    from autourgos_responses import OpenAIResponse

    llm = OpenAIResponse(model="gpt-4o", api_key="sk-...")
    reply = llm.invoke("What is the capital of France?")

    # With reasoning model
    llm = OpenAIResponse(model="o3-mini", reasoning_effort="medium")
    reply = llm.invoke("Solve step by step: 2x + 5 = 13")

    # Async streaming
    async for chunk in llm.astream("Explain quantum computing."):
        print(chunk, end="", flush=True)

    # Multi-turn chat
    messages = [{"role": "user", "content": "Hello"}]
    reply = llm.chat(messages)
"""

from .response import (
    OpenAIResponse,
    OpenAIResponseAPIError,
    OpenAIResponseConfigError,
    OpenAIResponseError,
    OpenAIResponseImportError,
    OpenAIResponseResponseError,
)
from .llm import BaseLLM, CircuitBreakerOpenException, FunctionCall, ToolCallResponse
from .model_runtime import (
    build_structured_output,
    configure_runtime_environment,
    extract_text_from_response,
    extract_usage_metadata,
    track_latency,
)

try:
    from importlib.metadata import version as _v, PackageNotFoundError
    __version__ = _v("autourgos-responses")
except Exception:
    __version__ = "1.0.0"

__all__ = [
    # Main class
    "OpenAIResponse",
    # Exceptions
    "OpenAIResponseError",
    "OpenAIResponseAPIError",
    "OpenAIResponseImportError",
    "OpenAIResponseResponseError",
    "OpenAIResponseConfigError",
    # Base types
    "BaseLLM",
    "FunctionCall",
    "ToolCallResponse",
    "CircuitBreakerOpenException",
    # Runtime helpers
    "track_latency",
    "extract_usage_metadata",
    "build_structured_output",
    "extract_text_from_response",
    "configure_runtime_environment",
]
