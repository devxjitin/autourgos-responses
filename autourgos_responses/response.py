"""
OpenAIResponse — LLM wrapper for the OpenAI Responses API.

Self-contained: no autourgos-core dependency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .llm import BaseLLM, FunctionCall, ToolCallResponse
from .model_runtime import (
    build_structured_output,
    coerce_prompt_variable,
    configure_runtime_environment,
    extract_template_fields,
    extract_text_from_response,
    track_latency,
)
from .core import (
    build_multimodal_prompt,
    build_reasoning_config,
    build_response_create_params,
    build_text_config,
    configure_async_openai_client,
    configure_openai_client,
    extract_text_delta_from_event,
    load_openai_module,
    logger,
    normalize_model_name,
    normalize_reasoning_effort,
    normalize_text_verbosity,
    release_async_openai_client,
    release_openai_client,
    resolve_api_key,
    resolve_base_url,
)

configure_runtime_environment()
_OPENAI_AVAILABLE, openai_cls, async_openai_cls, _OPENAI_IMPORT_ERROR = load_openai_module()


# ── Custom exceptions ─────────────────────────────────────────────────────────

class OpenAIResponseError(Exception):
    """Base exception for OpenAIResponse errors."""


class OpenAIResponseImportError(OpenAIResponseError):
    """Raised when the openai SDK cannot be imported."""


class OpenAIResponseAPIError(OpenAIResponseError):
    """Raised when an API request fails after all retries."""


class OpenAIResponseResponseError(OpenAIResponseError):
    """Raised when the API response cannot be interpreted."""


class OpenAIResponseConfigError(OpenAIResponseError):
    """Raised for incompatible configuration options."""


# ── Main class ────────────────────────────────────────────────────────────────

class OpenAIResponse(BaseLLM):
    """
    LLM wrapper for the OpenAI Responses API (client.responses.create).

    The Responses API is OpenAI's stateful, multi-turn, and reasoning-capable
    endpoint. This wrapper supports text generation, multi-modal input (images),
    streaming, structured output, reasoning configuration, and automatic retries.

    Example::

        from autourgos_responses import OpenAIResponse

        llm = OpenAIResponse(model="gpt-4o", api_key="sk-...")
        reply = llm.invoke("What is the capital of France?")

    With reasoning::

        llm = OpenAIResponse(model="o3-mini", reasoning_effort="medium")
        reply = llm.invoke("Solve this step by step: 2x + 5 = 13")

    Async streaming::

        async for chunk in llm.astream("Explain quantum entanglement."):
            print(chunk, end="", flush=True)

    Multi-turn chat::

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's 2+2?"},
        ]
        reply = llm.chat(messages)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        system_instruction: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        text_verbosity: Optional[str] = None,
        response_schema: Any = None,
        response_mime_type: Optional[str] = None,
        structured_output: bool = False,
        streaming: bool = False,
        max_retries: int = 3,
        timeout: Optional[float] = 60.0,
        backoff_factor: float = 0.5,
        input_pricing: Optional[float] = None,
        output_pricing: Optional[float] = None,
        circuit_failure_threshold: int = 5,
        circuit_cooldown_time: float = 30.0,
    ) -> None:
        """
        Args:
            model: OpenAI model name, e.g. "gpt-4o", "o3-mini", "o1".
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Override the API base URL (e.g. for proxies).
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            system_instruction: System prompt sent as the 'instructions' field.
            prompt_template: Optional template string with {variable} placeholders.
            temperature: Sampling temperature (0–2).
            top_p: Nucleus sampling probability (0–1).
            max_tokens: Maximum output tokens (maps to max_output_tokens).
            reasoning_effort: Reasoning model effort — "low", "medium", or "high".
            reasoning_summary: Whether to include reasoning summary in output.
            text_verbosity: Output verbosity hint — "concise", "detailed", or "auto".
            response_schema: Pydantic model or dict for structured JSON output.
            response_mime_type: e.g. "application/json" to enable json_object mode.
            structured_output: If True, invoke() returns a metadata dict.
            streaming: If True, invoke()/ainvoke() internally stream and join text.
            max_retries: Number of retry attempts on transient API errors.
            timeout: Request timeout in seconds.
            backoff_factor: Base multiplier for exponential back-off.
            input_pricing: USD per 1 million input tokens (for cost tracking).
            output_pricing: USD per 1 million output tokens (for cost tracking).
            circuit_failure_threshold: Consecutive failures before the circuit opens.
            circuit_cooldown_time: Seconds the circuit stays open before a probe.
        """
        super().__init__(
            input_pricing=input_pricing,
            output_pricing=output_pricing,
            circuit_failure_threshold=circuit_failure_threshold,
            circuit_cooldown_time=circuit_cooldown_time,
        )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.system_instruction = system_instruction
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.reasoning_summary = reasoning_summary
        self.text_verbosity = normalize_text_verbosity(text_verbosity)
        self.response_schema = response_schema
        self.response_mime_type = response_mime_type
        self.structured_output = structured_output
        self.streaming = streaming
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

        if self.structured_output and self.streaming:
            raise OpenAIResponseConfigError(
                "structured_output=True is incompatible with streaming=True."
            )

        self._model_name = normalize_model_name(self.model)
        self._client: Any = None
        self._async_client: Any = None
        self._init_clients()

    # ── Client init ───────────────────────────────────────────────────────────

    def _init_clients(self) -> None:
        if not _OPENAI_AVAILABLE or openai_cls is None or async_openai_cls is None:
            detail = f" Details: {_OPENAI_IMPORT_ERROR}" if _OPENAI_IMPORT_ERROR else ""
            raise OpenAIResponseImportError(
                "Failed to import openai SDK. Install it with: pip install openai" + detail
            )
        key = resolve_api_key(self.api_key)
        url = resolve_base_url(self.base_url)
        self._client = configure_openai_client(
            openai_cls,
            api_key=key,
            base_url=url,
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
        )
        self._async_client = configure_async_openai_client(
            async_openai_cls,
            api_key=key,
            base_url=url,
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
        )

    # ── Context managers ──────────────────────────────────────────────────────

    def __enter__(self) -> "OpenAIResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None

    async def __aenter__(self) -> "OpenAIResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._async_client is not None:
            await release_async_openai_client(self._async_client)
            self._async_client = None
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_prompt(
        self,
        prompt: Any,
        prompt_variables: Optional[Dict[str, Any]],
        files: Optional[Any] = None,
    ) -> Any:
        if prompt is not None:
            if isinstance(prompt, str) and not prompt.strip():
                raise ValueError("prompt must be a non-empty string or list when provided")
            if isinstance(prompt, list) and not prompt:
                raise ValueError("prompt must be a non-empty list when provided")
            if not isinstance(prompt, (str, list)):
                raise ValueError("prompt must be a string or list")
            if files and isinstance(prompt, list):
                raise OpenAIResponseConfigError(
                    "Cannot combine files with a pre-formatted list prompt."
                )
            return build_multimodal_prompt(prompt, files) if (files and isinstance(prompt, str)) else prompt

        if self.prompt_template is None:
            raise ValueError("prompt is required when prompt_template is not configured")

        merged = dict(prompt_variables or {})
        required = extract_template_fields(self.prompt_template)
        missing = sorted(f for f in required if f not in merged or not str(merged[f]).strip())
        if missing:
            raise ValueError(f"Missing prompt template variables: {', '.join(missing)}")
        rendered = self.prompt_template.format(**{k: coerce_prompt_variable(v) for k, v in merged.items()})
        if not rendered.strip():
            raise ValueError("Rendered prompt template is empty")
        return build_multimodal_prompt(rendered, files) if files else rendered

    def _build_base_params(self, *, input_data: Any, stream: bool) -> Dict[str, Any]:
        text_config = build_text_config(
            response_schema=self.response_schema,
            response_mime_type=self.response_mime_type,
            text_verbosity=self.text_verbosity,
        )
        reasoning = build_reasoning_config(
            effort=self.reasoning_effort,
            summary=self.reasoning_summary,
        )
        return build_response_create_params(
            self._model_name,
            input_data,
            instructions=self.system_instruction,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            reasoning=reasoning,
            text=text_config,
            stream=stream,
        )

    # ── Raw API calls ─────────────────────────────────────────────────────────

    def _create_raw(self, params: Dict[str, Any]) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._client.responses.create(**params)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise OpenAIResponseAPIError(
                        f"Responses API request failed after {self.max_retries} attempts. "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIResponseAPIError("Unexpected retry exhaustion") from last_exc

    async def _acreate_raw(self, params: Dict[str, Any]) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._async_client.responses.create(**params)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise OpenAIResponseAPIError(
                        f"Async Responses API request failed after {self.max_retries} attempts. "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIResponseAPIError("Unexpected async retry exhaustion") from last_exc

    # ── Non-stream invocation ─────────────────────────────────────────────────

    def _invoke_non_stream(self, *, input_data: Any) -> Any:
        params = self._build_base_params(input_data=input_data, stream=False)
        resp = self._create_raw(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp
        raise OpenAIResponseResponseError(
            "No text could be extracted from the Responses API response."
        )

    async def _ainvoke_non_stream(self, *, input_data: Any) -> Any:
        params = self._build_base_params(input_data=input_data, stream=False)
        resp = await self._acreate_raw(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp
        raise OpenAIResponseResponseError(
            "No text could be extracted from the async Responses API response."
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    def _invoke_stream_mode(self, *, input_data: Any) -> Iterator[str]:
        params = self._build_base_params(input_data=input_data, stream=True)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            emitted = False
            try:
                stream = self._client.responses.create(**params)
                for event in stream:
                    delta = extract_text_delta_from_event(event)
                    if delta:
                        emitted = True
                        yield delta
                if emitted:
                    return
                raise OpenAIResponseResponseError("No text deltas in streaming response")
            except (OpenAIResponseResponseError, OpenAIResponseAPIError):
                raise
            except Exception as exc:
                last_exc = exc
                if emitted or attempt == self.max_retries:
                    raise OpenAIResponseAPIError(
                        f"Streaming failed after {attempt} attempt(s). "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIResponseAPIError("Streaming failed unexpectedly") from last_exc

    async def _ainvoke_stream_mode(self, *, input_data: Any) -> AsyncIterator[str]:
        params = self._build_base_params(input_data=input_data, stream=True)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            emitted = False
            try:
                stream = await self._async_client.responses.create(**params)
                async for event in stream:
                    delta = extract_text_delta_from_event(event)
                    if delta:
                        emitted = True
                        yield delta
                if emitted:
                    return
                raise OpenAIResponseResponseError("No text deltas in async streaming response")
            except (OpenAIResponseResponseError, OpenAIResponseAPIError):
                raise
            except Exception as exc:
                last_exc = exc
                if emitted or attempt == self.max_retries:
                    raise OpenAIResponseAPIError(
                        f"Async streaming failed after {attempt} attempt(s). "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIResponseAPIError("Async streaming failed unexpectedly") from last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
    ) -> Any:
        """
        Generate a response synchronously.

        Args:
            prompt: Text string, content list, or None to use prompt_template.
            prompt_variables: Variables to fill prompt_template placeholders.
            files: Image file paths, bytes, or dicts to include as vision input.

        Returns:
            Generated text string, or a metadata dict if structured_output=True.
        """
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        if self.streaming:
            return "".join(self._invoke_stream_mode(input_data=resolved))
        with track_latency() as timing:
            response_text, raw_response = self._invoke_non_stream(input_data=resolved)
        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=response_text,
            raw_response=raw_response,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
        )
        self.last_metadata = metadata
        return metadata if self.structured_output else response_text

    async def ainvoke(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
    ) -> Any:
        """Async version of invoke()."""
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        if self.streaming:
            chunks: List[str] = []
            async for delta in self._ainvoke_stream_mode(input_data=resolved):
                chunks.append(delta)
            return "".join(chunks)
        with track_latency() as timing:
            response_text, raw_response = await self._ainvoke_non_stream(input_data=resolved)
        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=response_text,
            raw_response=raw_response,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
        )
        self.last_metadata = metadata
        return metadata if self.structured_output else response_text

    def stream(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
    ) -> Iterator[str]:
        """Stream text chunks synchronously."""
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        return self._invoke_stream_mode(input_data=resolved)

    async def astream(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
    ) -> AsyncIterator[str]:
        """Stream text chunks asynchronously."""
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        async for chunk in self._ainvoke_stream_mode(input_data=resolved):
            yield chunk

    # ── Low-level create() / acreate() ───────────────────────────────────────

    def create(self, input_data: Any = None, **overrides: Any) -> Any:
        """Direct access to client.responses.create() with managed retries."""
        if input_data is None:
            input_data = overrides.pop("input", None)
        if input_data is None:
            raise ValueError("input_data is required")
        params = self._build_base_params(input_data=input_data, stream=False)
        params.update(overrides)
        return self._create_raw(params)

    async def acreate(self, input_data: Any = None, **overrides: Any) -> Any:
        """Async version of create()."""
        if input_data is None:
            input_data = overrides.pop("input", None)
        if input_data is None:
            raise ValueError("input_data is required")
        params = self._build_base_params(input_data=input_data, stream=False)
        params.update(overrides)
        return await self._acreate_raw(params)

    # ── Multi-turn chat ───────────────────────────────────────────────────────

    def chat(self, messages: List[Dict[str, Any]], **overrides: Any) -> Any:
        """
        Send a multi-turn messages list to the Responses API.

        Args:
            messages: List of {role, content} dicts.
            **overrides: Extra params forwarded to create().

        Returns:
            Generated text or metadata dict.
        """
        params = self._build_base_params(input_data=messages, stream=False)
        params.update(overrides)
        with track_latency() as timing:
            resp = self._create_raw(params)
            text = extract_text_from_response(resp)
        if not text:
            raise OpenAIResponseResponseError("No text could be extracted from chat response")
        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=text,
            raw_response=resp,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
        )
        self.last_metadata = metadata
        return metadata if self.structured_output else text

    async def achat(self, messages: List[Dict[str, Any]], **overrides: Any) -> Any:
        """Async version of chat()."""
        params = self._build_base_params(input_data=messages, stream=False)
        params.update(overrides)
        with track_latency() as timing:
            resp = await self._acreate_raw(params)
            text = extract_text_from_response(resp)
        if not text:
            raise OpenAIResponseResponseError("No text could be extracted from async chat response")
        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=text,
            raw_response=resp,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
        )
        self.last_metadata = metadata
        return metadata if self.structured_output else text

    # ── Batch ─────────────────────────────────────────────────────────────────

    def batch_invoke(self, prompts: List[Any]) -> List[Any]:
        """Run invoke() for each prompt sequentially."""
        return [self.invoke(prompt=p) for p in prompts]

    async def abatch_invoke(self, prompts: List[Any]) -> List[Any]:
        """Run ainvoke() for each prompt concurrently."""
        return list(await asyncio.gather(*[self.ainvoke(prompt=p) for p in prompts]))

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"OpenAIResponse(model={self.model!r}, "
            f"reasoning_effort={self.reasoning_effort!r}, "
            f"streaming={self.streaming})"
        )


__all__ = [
    "OpenAIResponse",
    "OpenAIResponseError",
    "OpenAIResponseAPIError",
    "OpenAIResponseImportError",
    "OpenAIResponseResponseError",
    "OpenAIResponseConfigError",
]
