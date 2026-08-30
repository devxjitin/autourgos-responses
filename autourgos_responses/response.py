"""
OpenAIResponse — LLM wrapper for the OpenAI Responses API.

Self-contained: no autourgos-core dependency.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from .ledger import close_ledger, open_ledger, write_ledger_entry, write_shadow_ledger_entry
from .llm import BaseLLM, FunctionCall, ToolCallResponse
from .redaction import compile_patterns, redact_value, restore_text
from .shadow import compute_similarity
from .model_runtime import (
    build_structured_output,
    coerce_prompt_variable,
    configure_runtime_environment,
    extract_template_fields,
    extract_text_from_response,
    extract_usage_metadata,
    track_latency,
)
from .core import (
    build_multimodal_prompt,
    build_reasoning_config,
    build_response_create_params,
    build_responses_tools,
    build_text_config,
    configure_async_openai_client,
    configure_openai_client,
    extract_text_delta_from_event,
    extract_tool_calls_from_response,
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

# Client errors that will never succeed on retry — fail fast instead of
# burning the retry budget and adding latency.
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}

# Sentinel distinguishing "no override passed" from "override explicitly None".
_UNSET = object()


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


class OpenAIResponseValidationError(OpenAIResponseResponseError):
    """
    Raised by ``invoke_structured()``/``ainvoke_structured()`` when the model's
    output still fails Pydantic validation against ``output_schema`` after all
    validation retries are exhausted.

    ``.raw_text`` holds the last raw (invalid) response text.
    ``.validation_error`` holds the last Pydantic ``ValidationError`` (or JSON
    decode error) that caused the failure.
    """

    def __init__(self, message: str, *, raw_text: Optional[str], validation_error: Exception) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.validation_error = validation_error


class OpenAIResponseRedactionBlockedError(OpenAIResponseError):
    """
    Raised when ``redact_mode="block"`` and the resolved prompt matched one or
    more redaction patterns. ``.categories_found`` lists which categories
    (e.g. "email", "api_key") triggered the block.
    """

    def __init__(self, message: str, *, categories_found: List[str]) -> None:
        super().__init__(message)
        self.categories_found = categories_found


class OpenAIResponseAllProvidersFailedError(OpenAIResponseAPIError):
    """
    Raised when the primary provider and every configured fallback provider
    have all failed for the same request.

    ``.attempts`` holds one (label, exception) pair per provider that was
    tried, in the order they were attempted.
    """

    def __init__(self, message: str, attempts: List[Any]) -> None:
        super().__init__(message)
        self.attempts = attempts


# ── Main class ────────────────────────────────────────────────────────────────

class OpenAIResponse(BaseLLM):
    """
    LLM wrapper for the OpenAI Responses API (client.responses.create).

    The Responses API is OpenAI's stateful, multi-turn, and reasoning-capable
    endpoint. This wrapper supports text generation, multi-modal input (images),
    streaming, structured output, reasoning configuration, native tool-calling,
    provider fallback, a call ledger, PII redaction, shadow-mode dual dispatch,
    a session budget cap, and automatic retries.

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

    Native tool-calling::

        tools = [{"name": "get_weather", "description": "...", "parameters": {...}}]
        response = llm.invoke_with_tools("What's the weather in Paris?", tools)
        if response.has_tool_calls:
            print(response.tool_calls)
    """

    supports_tool_calling: bool = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        text_verbosity: Optional[str] = None,
        output_schema: Any = None,
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
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
        ledger_path: Optional[str] = None,
        ledger_store_content: bool = True,
        max_session_cost: Optional[float] = None,
        redact_pii: bool = False,
        redact_categories: Optional[List[str]] = None,
        redact_mode: str = "mask",
        redact_custom_patterns: Optional[Dict[str, str]] = None,
        redact_custom_terms: Optional[Dict[str, List[str]]] = None,
        redact_patterns_file: Optional[str] = None,
        redact_restore_in_response: bool = False,
        shadow_providers: Optional[List[Dict[str, Any]]] = None,
        on_shadow_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model: OpenAI model name, e.g. "gpt-4o", "o3-mini", "o1".
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Override the API base URL (e.g. for proxies).
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            system_prompt: System prompt sent as the 'instructions' field.
            prompt_template: Optional template string with {variable} placeholders.
            temperature: Sampling temperature (0–2).
            top_p: Nucleus sampling probability (0–1).
            max_tokens: Maximum output tokens (maps to max_output_tokens).
            reasoning_effort: Reasoning model effort — "low", "medium", or "high".
            reasoning_summary: Whether to include reasoning summary in output.
            text_verbosity: Output verbosity hint — "low", "medium", or "high".
            output_schema: Pydantic model or dict for structured JSON output.
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
            fallback_providers: Ordered list of backup providers to try, each a dict with
                "model" (required) and optional "api_key" / "base_url" / "organization" /
                "project". Tried in order after the primary provider exhausts its retries.
            ledger_path: If set, every invoke()/ainvoke()/invoke_structured()/
                ainvoke_structured() call is recorded to a local SQLite file at this path
                (created if it doesn't exist). None (default) disables the ledger entirely.
            ledger_store_content: If True (default), prompt and response text are stored in
                the ledger. Set False to log only tokens/cost/latency/provider metadata.
            max_session_cost: If set, blocks further invoke()/ainvoke()/invoke_structured()/
                ainvoke_structured() calls once accumulated cost (llm.session_cost_used)
                reaches this cap, raising BudgetExceededException. Requires both
                input_pricing and output_pricing to be set. Call reset_session_budget() to
                unblock.
            redact_pii: If True, the resolved prompt (string or input-item list) is scanned
                for likely secrets/PII before it is sent. A heuristic, best-effort scrubber
                — not a compliance-grade DLP solution. Default False (no scanning).
            redact_categories: Which built-in categories to scan for ("email", "credit_card",
                "ssn", "phone", "api_key"). Defaults to all of them when redact_pii=True.
            redact_mode: "mask" (default) replaces matches with "[REDACTED:<category>]" and
                the call proceeds; "block" raises OpenAIResponseRedactionBlockedError instead
                of sending anything.
            redact_custom_patterns: Extra {name: regex} entries merged in alongside the
                built-in categories.
            redact_custom_terms: Bring-your-own dictionary of exact/literal values to redact,
                as {category: ["value one", "value two", ...]} — no regex needed.
            redact_patterns_file: Path to a JSON file with "patterns" ({name: regex}) and/or
                "terms" ({name: [values]}) keys.
            redact_restore_in_response: If True (default False), the final text returned to
                the caller has any echoed placeholders swapped back for their original
                values. Requires redact_pii=True and redact_mode="mask". The ledger always
                records the still-masked text regardless of this setting.
            shadow_providers: Providers to dispatch the same prompt to concurrently with the
                primary, for comparison only — invoke()/ainvoke() always return the primary's
                result. Same entry shape as fallback_providers. Results land in
                llm.last_shadow_results and, if a ledger is configured, in the shadow_calls
                table. Each shadow call costs real money that is NOT counted toward
                max_session_cost.
            on_shadow_result: Optional callback invoked with each shadow result dict as it
                completes.
            extra_body: Raw provider-specific request fields merged into every request
                (primary, fallback, and shadow) — e.g. vLLM's guided_json/guided_regex for
                constrained decoding. None (default) adds nothing.
        """
        super().__init__(
            input_pricing=input_pricing,
            output_pricing=output_pricing,
            circuit_failure_threshold=circuit_failure_threshold,
            circuit_cooldown_time=circuit_cooldown_time,
            max_session_cost=max_session_cost,
        )
        if max_session_cost is not None and (input_pricing is None or output_pricing is None):
            raise OpenAIResponseConfigError(
                "max_session_cost requires both input_pricing and output_pricing to be set "
                "— otherwise cost is never computed and the budget cap would never trigger."
            )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.reasoning_summary = reasoning_summary
        self.text_verbosity = normalize_text_verbosity(text_verbosity)
        self.output_schema = output_schema
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

        for i, entry in enumerate(fallback_providers or []):
            if not entry.get("model"):
                raise OpenAIResponseConfigError(
                    f"fallback_providers[{i}] is missing required key 'model'."
                )
        self.fallback_providers: List[Dict[str, Any]] = list(fallback_providers or [])
        self._fallback_sync_clients: Dict[int, Any] = {}
        self._fallback_async_clients: Dict[int, Any] = {}

        self.ledger_path = ledger_path
        self.ledger_store_content = ledger_store_content
        self._ledger_conn = open_ledger(ledger_path) if ledger_path else None
        self._ledger_lock = threading.Lock()

        if redact_mode not in ("mask", "block"):
            raise OpenAIResponseConfigError(
                f"redact_mode must be 'mask' or 'block', got {redact_mode!r}."
            )
        self.redact_pii = redact_pii
        self.redact_mode = redact_mode
        self.last_redacted_categories: List[str] = []
        self._last_redaction_map: Dict[str, str] = {}
        try:
            self._redact_patterns = (
                compile_patterns(
                    redact_categories, redact_custom_patterns, redact_custom_terms, redact_patterns_file
                )
                if redact_pii else {}
            )
        except ValueError as exc:
            raise OpenAIResponseConfigError(str(exc)) from exc

        if redact_restore_in_response:
            if not redact_pii:
                raise OpenAIResponseConfigError(
                    "redact_restore_in_response requires redact_pii=True."
                )
            if redact_mode != "mask":
                raise OpenAIResponseConfigError(
                    "redact_restore_in_response requires redact_mode='mask' — with "
                    "redact_mode='block' the call never reaches the model, so there is "
                    "nothing to restore."
                )
        self.redact_restore_in_response = redact_restore_in_response

        for i, entry in enumerate(shadow_providers or []):
            if not entry.get("model"):
                raise OpenAIResponseConfigError(
                    f"shadow_providers[{i}] is missing required key 'model'."
                )
        self.shadow_providers: List[Dict[str, Any]] = list(shadow_providers or [])
        self.on_shadow_result = on_shadow_result
        self.last_shadow_results: List[Dict[str, Any]] = []
        self._shadow_sync_clients: Dict[int, Any] = {}
        self._shadow_async_clients: Dict[int, Any] = {}

        self.extra_body = extra_body

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

    def _get_fallback_sync_client(self, index: int) -> Any:
        """Lazily create and cache the sync client for fallback_providers[index]."""
        if index not in self._fallback_sync_clients:
            cfg = self.fallback_providers[index]
            self._fallback_sync_clients[index] = configure_openai_client(
                openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._fallback_sync_clients[index]

    def _get_fallback_async_client(self, index: int) -> Any:
        """Lazily create and cache the async client for fallback_providers[index]."""
        if index not in self._fallback_async_clients:
            cfg = self.fallback_providers[index]
            self._fallback_async_clients[index] = configure_async_openai_client(
                async_openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._fallback_async_clients[index]

    def _sync_targets(self) -> Iterator[tuple]:
        """Yield (label, model_name, client) for the primary, then each fallback in order."""
        yield "primary", self._model_name, self._client
        for i, cfg in enumerate(self.fallback_providers):
            yield (
                f"fallback[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_fallback_sync_client(i),
            )

    def _async_targets(self) -> Iterator[tuple]:
        """Yield (label, model_name, client) for the primary, then each fallback in order."""
        yield "primary", self._model_name, self._async_client
        for i, cfg in enumerate(self.fallback_providers):
            yield (
                f"fallback[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_fallback_async_client(i),
            )

    def _get_shadow_sync_client(self, index: int) -> Any:
        """Lazily create and cache the sync client for shadow_providers[index]."""
        if index not in self._shadow_sync_clients:
            cfg = self.shadow_providers[index]
            self._shadow_sync_clients[index] = configure_openai_client(
                openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._shadow_sync_clients[index]

    def _get_shadow_async_client(self, index: int) -> Any:
        """Lazily create and cache the async client for shadow_providers[index]."""
        if index not in self._shadow_async_clients:
            cfg = self.shadow_providers[index]
            self._shadow_async_clients[index] = configure_async_openai_client(
                async_openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._shadow_async_clients[index]

    def _shadow_targets(self) -> Iterator[tuple]:
        """Yield (label, model_name, client) for each configured shadow provider."""
        for i, cfg in enumerate(self.shadow_providers):
            yield (
                f"shadow[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_shadow_sync_client(i),
            )

    def _async_shadow_targets(self) -> Iterator[tuple]:
        """Yield (label, model_name, client) for each configured shadow provider."""
        for i, cfg in enumerate(self.shadow_providers):
            yield (
                f"shadow[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_shadow_async_client(i),
            )

    # ── Context managers ──────────────────────────────────────────────────────

    def __enter__(self) -> "OpenAIResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None
        for client in self._fallback_sync_clients.values():
            release_openai_client(client)
        self._fallback_sync_clients = {}
        for client in self._shadow_sync_clients.values():
            release_openai_client(client)
        self._shadow_sync_clients = {}
        close_ledger(self._ledger_conn)
        self._ledger_conn = None

    async def __aenter__(self) -> "OpenAIResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._async_client is not None:
            await release_async_openai_client(self._async_client)
            self._async_client = None
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None
        for client in self._fallback_async_clients.values():
            await release_async_openai_client(client)
        self._fallback_async_clients = {}
        for client in self._fallback_sync_clients.values():
            release_openai_client(client)
        self._fallback_sync_clients = {}
        for client in self._shadow_async_clients.values():
            await release_async_openai_client(client)
        self._shadow_async_clients = {}
        for client in self._shadow_sync_clients.values():
            release_openai_client(client)
        self._shadow_sync_clients = {}
        close_ledger(self._ledger_conn)
        self._ledger_conn = None

    # ── Call ledger ───────────────────────────────────────────────────────────

    def _log_to_ledger(
        self,
        *,
        call_type: str,
        prompt: Any,
        metadata: Dict[str, Any],
        response_override: Any = _UNSET,
    ) -> None:
        if self._ledger_conn is None:
            return
        prompt_text = str(prompt) if (self.ledger_store_content and prompt is not None) else None
        raw_response = metadata.get("response") if response_override is _UNSET else response_override
        response_text = raw_response if self.ledger_store_content else None
        write_ledger_entry(
            self._ledger_conn,
            self._ledger_lock,
            model=metadata.get("model"),
            provider_used=metadata.get("provider_used"),
            call_type=call_type,
            prompt=prompt_text,
            response=response_text,
            metadata=metadata,
            redacted_categories=self.last_redacted_categories,
        )

    # ── Shadow-mode dual dispatch ────────────────────────────────────────────
    # Runs concurrently with (not after) the primary call. invoke()/ainvoke()
    # always return the primary's result; shadow results are observation-only.

    def _build_shadow_result(
        self,
        label: str,
        response_text: Optional[str],
        raw_response: Any,
        primary_text: Optional[str],
        latency_ms: float,
        error: Optional[str],
    ) -> Dict[str, Any]:
        if error is not None:
            return {
                "provider_used": label, "response": None, "similarity": None,
                "input_tokens": None, "output_tokens": None, "total_cost": None,
                "latency_ms": latency_ms, "error": error,
            }
        usage = extract_usage_metadata(raw_response)
        total_cost = None
        if (
            self.input_pricing is not None and self.output_pricing is not None
            and usage["input_tokens"] is not None and usage["output_tokens"] is not None
        ):
            total_cost = (
                (usage["input_tokens"] / 1_000_000) * self.input_pricing
                + (usage["output_tokens"] / 1_000_000) * self.output_pricing
            )
        return {
            "provider_used": label,
            "response": response_text,
            "similarity": compute_similarity(primary_text, response_text),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_cost": total_cost,
            "latency_ms": latency_ms,
            "error": None,
        }

    def _log_shadow_to_ledger(self, result: Dict[str, Any]) -> None:
        if self._ledger_conn is None:
            return
        write_shadow_ledger_entry(
            self._ledger_conn,
            self._ledger_lock,
            provider_used=result["provider_used"],
            response=result["response"] if self.ledger_store_content else None,
            similarity=result["similarity"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_cost=result["total_cost"],
            latency_ms=result["latency_ms"],
            error=result["error"],
        )

    def _finalize_shadow_results(self, raw_results: List[tuple], primary_text: Optional[str]) -> None:
        results = []
        for label, text, raw, latency_ms, error in raw_results:
            result = self._build_shadow_result(label, text, raw, primary_text, latency_ms, error)
            results.append(result)
            self._log_shadow_to_ledger(result)
            if self.on_shadow_result is not None:
                try:
                    self.on_shadow_result(result)
                except Exception:
                    logger.warning("on_shadow_result callback raised", exc_info=True)
        self.last_shadow_results = results

    def _execute_shadow_attempt_sync(
        self, label: str, model_name: str, client: Any, input_data: Any
    ) -> tuple:
        params = dict(self._build_base_params(input_data=input_data, stream=False))
        params["model"] = model_name
        start = time.perf_counter()
        try:
            raw = client.responses.create(**params)
            text = extract_text_from_response(raw)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, text, raw, latency_ms, None
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, None, None, latency_ms, f"{type(exc).__name__}: {exc}"

    async def _execute_shadow_attempt_async(
        self, label: str, model_name: str, client: Any, input_data: Any
    ) -> tuple:
        params = dict(self._build_base_params(input_data=input_data, stream=False))
        params["model"] = model_name
        start = time.perf_counter()
        try:
            raw = await client.responses.create(**params)
            text = extract_text_from_response(raw)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, text, raw, latency_ms, None
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, None, None, latency_ms, f"{type(exc).__name__}: {exc}"

    def _dispatch_shadow_sync(self, input_data: Any, primary_text: Optional[str]) -> None:
        if not self.shadow_providers:
            self.last_shadow_results = []
            return
        targets = list(self._shadow_targets())
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = [
                executor.submit(self._execute_shadow_attempt_sync, label, model_name, client, input_data)
                for label, model_name, client in targets
            ]
            raw_results = [f.result() for f in futures]
        self._finalize_shadow_results(raw_results, primary_text)

    async def _dispatch_shadow_async(self, input_data: Any, primary_text: Optional[str]) -> None:
        if not self.shadow_providers:
            self.last_shadow_results = []
            return
        targets = list(self._async_shadow_targets())
        raw_results = list(await asyncio.gather(*[
            self._execute_shadow_attempt_async(label, model_name, client, input_data)
            for label, model_name, client in targets
        ]))
        self._finalize_shadow_results(raw_results, primary_text)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_prompt(
        self,
        prompt: Any,
        prompt_variables: Optional[Dict[str, Any]],
        files: Optional[Any] = None,
    ) -> Any:
        """Resolve prompt or render from template, then apply redaction if enabled."""
        resolved = self._resolve_prompt_raw(prompt, prompt_variables, files)
        return self._apply_redaction(resolved)

    def _apply_redaction(self, resolved: Any) -> Any:
        self.last_redacted_categories = []
        self._last_redaction_map = {}
        if not self.redact_pii:
            return resolved
        redacted, found, mapping = redact_value(
            resolved, self._redact_patterns, track_mapping=self.redact_restore_in_response
        )
        self.last_redacted_categories = found
        self._last_redaction_map = mapping
        if not found:
            return resolved
        if self.redact_mode == "block":
            raise OpenAIResponseRedactionBlockedError(
                f"Prompt blocked: matched redaction categories {found}.",
                categories_found=found,
            )
        return redacted

    def _resolve_prompt_raw(
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
            output_schema=self.output_schema,
            response_mime_type=self.response_mime_type,
            text_verbosity=self.text_verbosity,
        )
        reasoning = build_reasoning_config(
            effort=self.reasoning_effort,
            summary=self.reasoning_summary,
        )
        params = build_response_create_params(
            self._model_name,
            input_data,
            instructions=self.system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            reasoning=reasoning,
            text=text_config,
            stream=stream,
        )
        if self.extra_body:
            params["extra_body"] = dict(self.extra_body)
        return params

    # ── Raw API calls (single client, with retry/back-off) ──────────────────────

    def _attempt_sync_create(self, client: Any, params: Dict[str, Any], label: str) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return client.responses.create(**params)
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    raise OpenAIResponseAPIError(
                        f"[{label}] Responses API request failed with non-retryable status "
                        f"{status_code}. Error: {type(exc).__name__}: {exc}"
                    ) from exc
                if attempt == self.max_retries:
                    raise OpenAIResponseAPIError(
                        f"[{label}] Responses API request failed after {self.max_retries} "
                        f"attempts. Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIResponseAPIError(f"[{label}] Unexpected retry exhaustion") from last_exc

    async def _attempt_async_create(self, client: Any, params: Dict[str, Any], label: str) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await client.responses.create(**params)
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    raise OpenAIResponseAPIError(
                        f"[{label}] Async Responses API request failed with non-retryable "
                        f"status {status_code}. Error: {type(exc).__name__}: {exc}"
                    ) from exc
                if attempt == self.max_retries:
                    raise OpenAIResponseAPIError(
                        f"[{label}] Async Responses API request failed after "
                        f"{self.max_retries} attempts. Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIResponseAPIError(f"[{label}] Unexpected async retry exhaustion") from last_exc

    def _create_raw(self, params: Dict[str, Any]) -> Any:
        return self._attempt_sync_create(self._client, params, "primary")

    async def _acreate_raw(self, params: Dict[str, Any]) -> Any:
        return await self._attempt_async_create(self._async_client, params, "primary")

    # ── Raw API calls across primary + fallback providers ───────────────────────

    def _create_across_providers(self, params: Dict[str, Any]) -> tuple:
        """Try the primary, then each fallback provider in order. Returns (raw, label)."""
        attempts: List[Any] = []
        for label, model_name, client in self._sync_targets():
            attempt_params = dict(params)
            attempt_params["model"] = model_name
            try:
                return self._attempt_sync_create(client, attempt_params, label), label
            except OpenAIResponseAPIError as exc:
                attempts.append((label, exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise OpenAIResponseAllProvidersFailedError(
            f"All {len(attempts)} provider(s) failed: "
            + "; ".join(f"{label}: {exc}" for label, exc in attempts),
            attempts=attempts,
        )

    async def _acreate_across_providers(self, params: Dict[str, Any]) -> tuple:
        """Try the primary, then each fallback provider in order. Returns (raw, label)."""
        attempts: List[Any] = []
        for label, model_name, client in self._async_targets():
            attempt_params = dict(params)
            attempt_params["model"] = model_name
            try:
                return await self._attempt_async_create(client, attempt_params, label), label
            except OpenAIResponseAPIError as exc:
                attempts.append((label, exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise OpenAIResponseAllProvidersFailedError(
            f"All {len(attempts)} provider(s) failed: "
            + "; ".join(f"{label}: {exc}" for label, exc in attempts),
            attempts=attempts,
        )

    # ── Non-stream invocation ─────────────────────────────────────────────────

    def _invoke_non_stream(self, *, input_data: Any) -> Any:
        params = self._build_base_params(input_data=input_data, stream=False)
        resp, provider_label = self._create_across_providers(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp, provider_label
        raise OpenAIResponseResponseError(
            "No text could be extracted from the Responses API response."
        )

    async def _ainvoke_non_stream(self, *, input_data: Any) -> Any:
        params = self._build_base_params(input_data=input_data, stream=False)
        resp, provider_label = await self._acreate_across_providers(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp, provider_label
        raise OpenAIResponseResponseError(
            "No text could be extracted from the async Responses API response."
        )

    # ── Streaming ─────────────────────────────────────────────────────────────
    # Fallback only kicks in if a target fails before it has emitted any chunk —
    # once partial text has reached the caller, switching providers mid-stream
    # would duplicate or corrupt output, so the error is raised as-is instead.

    def _invoke_stream_mode(self, *, input_data: Any) -> Iterator[str]:
        base_params = self._build_base_params(input_data=input_data, stream=True)
        attempts: List[Any] = []
        for label, model_name, client in self._sync_targets():
            params = dict(base_params)
            params["model"] = model_name
            last_exc: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                emitted = False
                try:
                    stream = client.responses.create(**params)
                    for event in stream:
                        delta = extract_text_delta_from_event(event)
                        if delta:
                            emitted = True
                            yield delta
                    if emitted:
                        return
                    raise OpenAIResponseResponseError(f"[{label}] No text deltas in streaming response")
                except OpenAIResponseResponseError as exc:
                    if emitted:
                        raise
                    last_exc = exc
                    break
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    if emitted:
                        raise OpenAIResponseAPIError(
                            f"[{label}] Streaming failed mid-response after emitting output: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if status_code in _NON_RETRYABLE_STATUS_CODES:
                        last_exc = OpenAIResponseAPIError(
                            f"[{label}] Streaming failed with non-retryable status {status_code}. "
                            f"Error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    if attempt == self.max_retries:
                        last_exc = OpenAIResponseAPIError(
                            f"[{label}] Streaming failed after {attempt} attempt(s). "
                            f"Last error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
                    continue
                break
            attempts.append((label, last_exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise OpenAIResponseAllProvidersFailedError(
            f"Streaming failed on all {len(attempts)} provider(s): "
            + "; ".join(f"{lbl}: {exc}" for lbl, exc in attempts),
            attempts=attempts,
        )

    async def _ainvoke_stream_mode(self, *, input_data: Any) -> AsyncIterator[str]:
        base_params = self._build_base_params(input_data=input_data, stream=True)
        attempts: List[Any] = []
        for label, model_name, client in self._async_targets():
            params = dict(base_params)
            params["model"] = model_name
            last_exc: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                emitted = False
                try:
                    stream = await client.responses.create(**params)
                    async for event in stream:
                        delta = extract_text_delta_from_event(event)
                        if delta:
                            emitted = True
                            yield delta
                    if emitted:
                        return
                    raise OpenAIResponseResponseError(f"[{label}] No text deltas in async streaming response")
                except OpenAIResponseResponseError as exc:
                    if emitted:
                        raise
                    last_exc = exc
                    break
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    if emitted:
                        raise OpenAIResponseAPIError(
                            f"[{label}] Async streaming failed mid-response after emitting output: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if status_code in _NON_RETRYABLE_STATUS_CODES:
                        last_exc = OpenAIResponseAPIError(
                            f"[{label}] Async streaming failed with non-retryable status {status_code}. "
                            f"Error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    if attempt == self.max_retries:
                        last_exc = OpenAIResponseAPIError(
                            f"[{label}] Async streaming failed after {attempt} attempt(s). "
                            f"Last error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
                    continue
                break
            attempts.append((label, last_exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise OpenAIResponseAllProvidersFailedError(
            f"Async streaming failed on all {len(attempts)} provider(s): "
            + "; ".join(f"{lbl}: {exc}" for lbl, exc in attempts),
            attempts=attempts,
        )

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
        self._check_budget()
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        if self.streaming:
            return "".join(self._invoke_stream_mode(input_data=resolved))
        with track_latency() as timing:
            response_text, raw_response, provider_label = self._invoke_non_stream(input_data=resolved)

        masked_response_text = response_text
        if self.redact_restore_in_response:
            response_text = restore_text(response_text, self._last_redaction_map)

        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=response_text,
            raw_response=raw_response,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
            extra_fields={"provider_used": provider_label},
        )
        self.last_metadata = metadata
        self._record_session_cost(metadata.get("total_cost"))
        self._log_to_ledger(
            call_type="invoke", prompt=resolved, metadata=metadata, response_override=masked_response_text
        )
        self._dispatch_shadow_sync(resolved, response_text)
        return metadata if self.structured_output else response_text

    async def ainvoke(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
    ) -> Any:
        """Async version of invoke()."""
        self._check_budget()
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        if self.streaming:
            chunks: List[str] = []
            async for delta in self._ainvoke_stream_mode(input_data=resolved):
                chunks.append(delta)
            return "".join(chunks)

        with track_latency() as timing:
            response_text, raw_response, provider_label = await self._ainvoke_non_stream(input_data=resolved)

        masked_response_text = response_text
        if self.redact_restore_in_response:
            response_text = restore_text(response_text, self._last_redaction_map)

        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=response_text,
            raw_response=raw_response,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
            extra_fields={"provider_used": provider_label},
        )
        self.last_metadata = metadata
        self._record_session_cost(metadata.get("total_cost"))
        self._log_to_ledger(
            call_type="ainvoke", prompt=resolved, metadata=metadata, response_override=masked_response_text
        )
        await self._dispatch_shadow_async(resolved, response_text)
        return metadata if self.structured_output else response_text

    # ── Validated structured output ──────────────────────────────────────────
    # Server-side json_schema strict mode (build_text_config) already
    # constrains the shape of the JSON. This adds a feedback loop on top: if
    # the result still fails Pydantic validation, the error is sent back to
    # the model and it gets another chance to correct itself.

    @staticmethod
    def _is_pydantic_model_class(obj: Any) -> bool:
        return isinstance(obj, type) and hasattr(obj, "model_validate_json")

    def _require_structured_schema(self) -> None:
        if not self._is_pydantic_model_class(self.output_schema):
            raise OpenAIResponseConfigError(
                "invoke_structured()/ainvoke_structured() require output_schema= to be a "
                "Pydantic BaseModel class (not a plain dict or None)."
            )
        if self.streaming:
            raise OpenAIResponseConfigError(
                "invoke_structured()/ainvoke_structured() are incompatible with streaming=True."
            )

    @staticmethod
    def _correction_input_items(bad_text: str, error: Exception) -> List[Dict[str, Any]]:
        return [
            {"role": "assistant", "content": bad_text},
            {
                "role": "user",
                "content": (
                    "Your last response failed schema validation with this error:\n"
                    f"{error}\n\nReturn corrected JSON that matches the schema exactly."
                ),
            },
        ]

    def invoke_structured(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        max_validation_retries: int = 2,
    ) -> Any:
        """
        Generate a response and validate it against ``output_schema`` (a Pydantic
        BaseModel class), retrying with the validation error fed back to the model
        on failure.

        Returns:
            A validated instance of ``output_schema``.

        Raises:
            OpenAIResponseValidationError: if the response still fails validation
                after ``max_validation_retries`` correction attempts.
        """
        self._require_structured_schema()
        self._check_budget()
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        input_data = resolved if isinstance(resolved, list) else [{"role": "user", "content": resolved}]

        last_error: Optional[Exception] = None
        last_text: Optional[str] = None
        for attempt in range(max_validation_retries + 1):
            with track_latency() as timing:
                response_text, raw_response, provider_label = self._invoke_non_stream(input_data=input_data)
            masked_response_text = response_text
            text_to_validate = response_text
            if self.redact_restore_in_response:
                text_to_validate = restore_text(response_text, self._last_redaction_map)
            last_text = text_to_validate
            try:
                validated = self.output_schema.model_validate_json(text_to_validate)
            except Exception as exc:
                last_error = exc
                # Feed back the model's own (still-masked) output — never the restored
                # text, which would leak the real secret into the model's own context.
                input_data = input_data + self._correction_input_items(masked_response_text, exc)
                continue
            self.last_metadata = build_structured_output(
                model_name=self._model_name,
                response_text=text_to_validate,
                raw_response=raw_response,
                latency_ms=timing["latency_ms"],
                input_pricing=self.input_pricing,
                output_pricing=self.output_pricing,
                extra_fields={"provider_used": provider_label, "validation_retries": attempt},
            )
            self._record_session_cost(self.last_metadata.get("total_cost"))
            self._log_to_ledger(
                call_type="invoke_structured", prompt=resolved, metadata=self.last_metadata,
                response_override=masked_response_text,
            )
            return validated

        raise OpenAIResponseValidationError(
            f"Output failed schema validation after {max_validation_retries + 1} attempt(s). "
            f"Last error: {last_error}",
            raw_text=last_text,
            validation_error=last_error,
        )

    async def ainvoke_structured(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        max_validation_retries: int = 2,
    ) -> Any:
        """Async version of invoke_structured()."""
        self._require_structured_schema()
        self._check_budget()
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        input_data = resolved if isinstance(resolved, list) else [{"role": "user", "content": resolved}]

        last_error: Optional[Exception] = None
        last_text: Optional[str] = None
        for attempt in range(max_validation_retries + 1):
            with track_latency() as timing:
                response_text, raw_response, provider_label = await self._ainvoke_non_stream(input_data=input_data)
            masked_response_text = response_text
            text_to_validate = response_text
            if self.redact_restore_in_response:
                text_to_validate = restore_text(response_text, self._last_redaction_map)
            last_text = text_to_validate
            try:
                validated = self.output_schema.model_validate_json(text_to_validate)
            except Exception as exc:
                last_error = exc
                input_data = input_data + self._correction_input_items(masked_response_text, exc)
                continue
            self.last_metadata = build_structured_output(
                model_name=self._model_name,
                response_text=text_to_validate,
                raw_response=raw_response,
                latency_ms=timing["latency_ms"],
                input_pricing=self.input_pricing,
                output_pricing=self.output_pricing,
                extra_fields={"provider_used": provider_label, "validation_retries": attempt},
            )
            self._record_session_cost(self.last_metadata.get("total_cost"))
            self._log_to_ledger(
                call_type="ainvoke_structured", prompt=resolved, metadata=self.last_metadata,
                response_override=masked_response_text,
            )
            return validated

        raise OpenAIResponseValidationError(
            f"Output failed schema validation after {max_validation_retries + 1} attempt(s). "
            f"Last error: {last_error}",
            raw_text=last_text,
            validation_error=last_error,
        )

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

    # ── Native function-calling ───────────────────────────────────────────────

    def invoke_with_tools(
        self,
        prompt: Any,
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolCallResponse:
        """
        Call the Responses API with native function-calling tools.

        Args:
            prompt: User message string, content list, or pre-built input-item list.
            tools: List of tool dicts with keys name/description/parameters.
            **kwargs: Optional overrides such as tool_choice, files.

        Returns:
            ToolCallResponse with tool_calls (if the model called tools)
            or text (if the model gave a final answer).
        """
        files = kwargs.pop("files", None)
        resolved = self._resolve_prompt(prompt, None, files)
        responses_tools = build_responses_tools(tools)
        params = self._build_base_params(input_data=resolved, stream=False)
        if responses_tools:
            params["tools"] = responses_tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        raw, _provider_label = self._create_across_providers(params)
        raw_calls = extract_tool_calls_from_response(raw) if responses_tools else []
        if raw_calls:
            tool_calls = [
                FunctionCall(name=c["name"], arguments=c["arguments"], call_id=c["call_id"])
                for c in raw_calls
            ]
            return ToolCallResponse(tool_calls=tool_calls, raw=raw)
        text = extract_text_from_response(raw)
        return ToolCallResponse(text=text, raw=raw)

    async def ainvoke_with_tools(
        self,
        prompt: Any,
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolCallResponse:
        """Async version of invoke_with_tools()."""
        files = kwargs.pop("files", None)
        resolved = self._resolve_prompt(prompt, None, files)
        responses_tools = build_responses_tools(tools)
        params = self._build_base_params(input_data=resolved, stream=False)
        if responses_tools:
            params["tools"] = responses_tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        raw, _provider_label = await self._acreate_across_providers(params)
        raw_calls = extract_tool_calls_from_response(raw) if responses_tools else []
        if raw_calls:
            tool_calls = [
                FunctionCall(name=c["name"], arguments=c["arguments"], call_id=c["call_id"])
                for c in raw_calls
            ]
            return ToolCallResponse(tool_calls=tool_calls, raw=raw)
        text = extract_text_from_response(raw)
        return ToolCallResponse(text=text, raw=raw)

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
    "OpenAIResponseValidationError",
    "OpenAIResponseRedactionBlockedError",
    "OpenAIResponseAllProvidersFailedError",
]
