"""
OpenAIResponse — LLM wrapper for the OpenAI Responses API.

Self-contained: no autourgos-core dependency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from .ledger import close_ledger
from .llm import (
    _NON_RETRYABLE_STATUS_CODES,
    BaseProviderLLM,
    FunctionCall,
    NonTransientError,
    ToolCallResponse,
)
from .redaction import redact_value, restore_text
from .model_runtime import (
    build_structured_output,
    coerce_prompt_variable,
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
    extract_final_response_from_stream_event,
    extract_text_delta_from_event,
    extract_tool_calls_from_response,
    load_openai_module,
    logger,
    normalize_model_name,
    normalize_native_tool_calling_input,
    normalize_reasoning_effort,
    normalize_text_verbosity,
    release_async_openai_client,
    release_openai_client,
    resolve_api_key,
    resolve_base_url,
    strip_unsupported_sampling_params,
)

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


class OpenAIResponseConfigError(OpenAIResponseError, NonTransientError):
    """
    Raised for incompatible configuration options.

    A caller/config mistake, not a sign the provider is unhealthy -- mixes in
    NonTransientError so it never counts toward the circuit breaker's
    consecutive-failure threshold (see NonTransientError's docstring).
    """


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


class OpenAIResponseRedactionBlockedError(OpenAIResponseError, NonTransientError):
    """
    Raised when ``redact_mode="block"`` and the resolved prompt matched one or
    more redaction patterns. ``.categories_found`` lists which categories
    (e.g. "email", "api_key") triggered the block.

    This is the redaction policy working as designed, not a provider failure
    -- mixes in NonTransientError so it never counts toward the circuit
    breaker's consecutive-failure threshold (see NonTransientError's
    docstring). Without this, a burst of legitimately-blocked prompts would
    trip the breaker and block every other call on the same instance,
    including clean ones.
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


class OpenAIResponseDeadlineExceededError(OpenAIResponseAPIError):
    """
    Raised when ``max_call_duration`` is set and the aggregate wall-clock
    time for one logical call (across all retries and, if configured, every
    fallback provider) has been exceeded.

    Without an aggregate deadline, retries and fallback each get their own
    full retry budget independently -- a synchronous caller can be left
    waiting for up to roughly ``providers x max_retries x timeout`` (plus
    backoff sleeps) before finally getting an answer or an error. This is
    checked between attempts/providers, not by cancelling an in-flight HTTP
    request already sent (that's still bounded by the per-request
    ``timeout``) -- so total wall-clock time can slightly exceed
    ``max_call_duration`` by up to one in-flight request's duration, but
    never by a further full retry/fallback cycle.
    """


# ── Main class ────────────────────────────────────────────────────────────────

class OpenAIResponse(BaseProviderLLM):
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

    _config_error_cls = OpenAIResponseConfigError
    _deadline_exceeded_cls = OpenAIResponseDeadlineExceededError
    _api_error_cls = OpenAIResponseAPIError
    _all_providers_failed_cls = OpenAIResponseAllProvidersFailedError
    _api_name = "Responses API"
    supports_tool_calling: bool = True

    def _do_sync_create(self, client: Any, params: Dict[str, Any]) -> Any:
        return client.responses.create(**params)

    async def _do_async_create(self, client: Any, params: Dict[str, Any]) -> Any:
        return await client.responses.create(**params)

    def _apply_per_target_param_guards(self, params: Dict[str, Any], model_name: str) -> None:
        strip_unsupported_sampling_params(params, model_name)

    _logger = logger

    def _extract_response_text(self, raw: Any) -> Optional[str]:
        return extract_text_from_response(raw)

    def _extract_usage(self, raw: Any) -> Dict[str, Optional[int]]:
        return extract_usage_metadata(raw)

    def _build_base_params_for_call(
        self, prompt_input: Any, *, stream: bool, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._build_base_params(input_data=prompt_input, stream=stream, overrides=overrides)

    _response_error_cls = OpenAIResponseResponseError

    def _extract_text_delta(self, event: Any) -> Optional[str]:
        return extract_text_delta_from_event(event)

    def _extract_usage_event(self, event: Any) -> Optional[Any]:
        return extract_final_response_from_stream_event(event)

    def _invoke_non_stream_for_call(self, prepared_input: Any, *, overrides: Optional[Dict[str, Any]]) -> Any:
        return self._invoke_non_stream(input_data=prepared_input, overrides=overrides)

    async def _ainvoke_non_stream_for_call(self, prepared_input: Any, *, overrides: Optional[Dict[str, Any]]) -> Any:
        return await self._ainvoke_non_stream(input_data=prepared_input, overrides=overrides)

    _validation_error_cls = OpenAIResponseValidationError
    _create_input_key = "input"
    _create_missing_input_message = "input_data is required"

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
        max_call_duration: Optional[float] = None,
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
        store: Optional[bool] = None,
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
            max_retries: Total attempts per provider on transient API errors, including
                the first try (e.g. 3 means 1 initial attempt + up to 2 retries). Must
                be >= 1 -- there is no "0 attempts" mode.
            timeout: Request timeout in seconds, per HTTP attempt.
            backoff_factor: Base multiplier for exponential back-off.
            max_call_duration: Optional aggregate wall-clock budget in seconds for one
                logical invoke()/ainvoke()/etc. call, covering every retry attempt and
                (if configured) every fallback provider. Without this, retries and
                fallback each get their own full retry budget independently — worst
                case, a call can take roughly providers x max_retries x timeout (plus
                backoff sleeps) before finally returning or raising. When set, the
                deadline is checked before each retry attempt and before moving to the
                next fallback provider, raising OpenAIResponseDeadlineExceededError
                once exceeded — it does not cancel an in-flight HTTP request already
                sent (that stays bounded by timeout=), so total wall-clock time can
                exceed max_call_duration by up to one in-flight request's duration, but
                never by a further full retry/fallback cycle. None (default) disables
                this cap — retries and fallback behave exactly as before.
            input_pricing: USD per 1 million input tokens (for cost tracking).
            output_pricing: USD per 1 million output tokens (for cost tracking).
            circuit_failure_threshold: Consecutive failures before the circuit opens.
            circuit_cooldown_time: Seconds the circuit stays open before a probe.
            fallback_providers: Ordered list of backup providers to try, each a dict with
                "model" (required) and optional "api_key" / "base_url" / "organization" /
                "project" / "input_pricing" / "output_pricing". Tried in order after the
                primary provider exhausts its retries. When a fallback actually answers,
                llm.last_metadata/the ledger report *that* fallback's own model name, and
                cost is computed from its own "input_pricing"/"output_pricing" (not the
                primary's — a fallback is typically a different, differently-priced model).
                If a fallback entry doesn't set them, cost fields are simply omitted for
                that call rather than computed with the wrong (primary's) price.
            ledger_path: If set, every invoke()/ainvoke()/invoke_structured()/
                ainvoke_structured() call is recorded to a local SQLite file at this path
                (created if it doesn't exist). None (default) disables the ledger entirely.
            ledger_store_content: If True (default), prompt and response text are stored in
                the ledger. Set False to log only tokens/cost/latency/provider metadata.
            max_session_cost: If set, blocks further invoke()/ainvoke()/invoke_structured()/
                ainvoke_structured() calls once accumulated cost (llm.session_cost_used)
                reaches this cap, raising BudgetExceededException. Requires both
                input_pricing and output_pricing to be set. Call reset_session_budget() to
                unblock. Concurrent calls sharing a capped instance (e.g. via
                abatch_invoke()) are admitted one at a time — a call's budget check, its
                API call, and recording its cost all happen before the next concurrent
                call's check is allowed to proceed — so the cap actually holds instead of
                being overshoot-able by N concurrent calls each passing the check before
                any of them records cost. This trades away concurrency for calls sharing a
                capped instance; an uncapped instance (max_session_cost=None, the default)
                pays no serialization cost.
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
                result. Same entry shape as fallback_providers, including its own optional
                "input_pricing"/"output_pricing" — a shadow result's cost is computed from
                that entry's own pricing (never the primary's), and left unset if the entry
                doesn't provide it. Results land in llm.last_shadow_results and, if a ledger
                is configured, in the shadow_calls table. Each shadow call costs real money
                that is NOT counted toward max_session_cost.
            on_shadow_result: Optional callback invoked with each shadow result dict as it
                completes.
            extra_body: Raw provider-specific request fields merged into every request
                (primary, fallback, and shadow) — e.g. vLLM's guided_json/guided_regex for
                constrained decoding. None (default) adds nothing.
            store: Whether OpenAI should persist the response server-side (retrievable by
                ID, used for evals/distillation). None (default) omits the param entirely,
                so the API's own default applies. A per-call invoke(store=...)/etc. override
                takes precedence over this constructor default for that one call.
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
        self.timeout = timeout
        self.backoff_factor = backoff_factor
        self.max_call_duration = max_call_duration

        if self.structured_output and self.streaming:
            raise OpenAIResponseConfigError(
                "structured_output=True is incompatible with streaming=True."
            )

        self._init_provider_common(
            max_retries=max_retries,
            fallback_providers=fallback_providers,
            ledger_path=ledger_path,
            ledger_store_content=ledger_store_content,
            redact_pii=redact_pii,
            redact_mode=redact_mode,
            redact_categories=redact_categories,
            redact_custom_patterns=redact_custom_patterns,
            redact_custom_terms=redact_custom_terms,
            redact_patterns_file=redact_patterns_file,
            redact_restore_in_response=redact_restore_in_response,
            shadow_providers=shadow_providers,
            on_shadow_result=on_shadow_result,
            extra_body=extra_body,
            store=store,
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

    def close(self) -> None:
        """
        Release the underlying client(s) synchronously.

        Equivalent to ``__exit__()`` — lets callers that hold this LLM via
        composition (e.g. autourgos-agent's ``Agent``, whose context-manager
        cleanup calls ``llm.close()``/``llm.aclose()`` if present) release
        resources without needing to use ``with`` directly on this object.
        """
        self.__exit__()

    async def aclose(self) -> None:
        """Release the underlying client(s) asynchronously. Equivalent to ``__aexit__()``."""
        await self.__aexit__()

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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_redaction(self, resolved: Any) -> "tuple[Any, List[str], Dict[str, str]]":
        """
        Redact ``resolved`` if enabled and return (redacted_value,
        found_categories, mapping) as a call-local tuple.

        Deliberately does NOT write to ``self.last_redacted_categories``/
        ``self._last_redaction_map`` here -- those are best-effort
        convenience attributes set by the caller *after* its full call
        completes (mirroring ``self.last_metadata``), never used internally
        for restore_text()/ledger logging. Writing them here made every
        concurrent call on a shared instance race to overwrite the same
        instance attributes, so a slower call could restore/log using a
        faster, unrelated call's mapping/categories -- including splicing
        another caller's secret into this caller's returned text.
        """
        if not self.redact_pii:
            return resolved, [], {}
        redacted, found, mapping = redact_value(
            resolved, self._redact_patterns, track_mapping=self.redact_restore_in_response
        )
        if not found:
            return resolved, [], {}
        if self.redact_mode == "block":
            raise OpenAIResponseRedactionBlockedError(
                f"Prompt blocked: matched redaction categories {found}.",
                categories_found=found,
            )
        return redacted, found, mapping

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

    def _build_base_params(
        self,
        *,
        input_data: Any,
        stream: bool,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        if self.store is not None:
            params["store"] = self.store
        if overrides:
            # "input"/"model"/"stream" stay structurally managed (per-target
            # model swap on fallback, non-stream vs. stream dispatch) — a
            # caller override can't hijack those, only request params like
            # temperature, top_p, max_output_tokens, instructions.
            params.update(
                {k: v for k, v in overrides.items() if k not in ("input", "model", "stream")}
            )
        return params

    # ── Aggregate call deadline ──────────────────────────────────────────────
    # Opt-in (max_call_duration=None disables this entirely): without it,
    # retries and fallback providers each get their own full retry budget
    # independently, with no cap on total wall-clock time for one logical
    # call. Checked between attempts/providers only -- an in-flight HTTP
    # request already sent is never cancelled, so it stays bounded by
    # timeout= instead.

    # ── Non-stream invocation ─────────────────────────────────────────────────

    def _invoke_non_stream(
        self, *, input_data: Any, overrides: Optional[Dict[str, Any]] = None
    ) -> Any:
        params = self._build_base_params(input_data=input_data, stream=False, overrides=overrides)
        resp, provider_label, provider_model, provider_pricing = self._create_across_providers(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp, provider_label, provider_model, provider_pricing
        raise OpenAIResponseResponseError(
            "No text could be extracted from the Responses API response."
        )

    async def _ainvoke_non_stream(
        self, *, input_data: Any, overrides: Optional[Dict[str, Any]] = None
    ) -> Any:
        params = self._build_base_params(input_data=input_data, stream=False, overrides=overrides)
        resp, provider_label, provider_model, provider_pricing = await self._acreate_across_providers(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp, provider_label, provider_model, provider_pricing
        raise OpenAIResponseResponseError(
            "No text could be extracted from the async Responses API response."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        **overrides: Any,
    ) -> Any:
        """
        Generate a response synchronously.

        Args:
            prompt: Text string, content list, or None to use prompt_template.
            prompt_variables: Variables to fill prompt_template placeholders.
            files: Image file paths, bytes, or dicts to include as vision input.
            **overrides: Per-call request params merged over the constructor's
                defaults for this call only — e.g. temperature=, top_p=,
                max_tokens=. Applied to every provider in the fallback chain;
                "input"/"model"/"stream" are structurally managed and cannot
                be overridden this way.

        Returns:
            Generated text string, or a metadata dict if structured_output=True.
        """
        self._check_budget()
        resolved, redacted_categories, redaction_map = self._resolve_prompt(prompt, prompt_variables, files)
        self.last_redacted_categories = redacted_categories
        return self._run_invoke(
            prepared_input=resolved,
            resolved=resolved,
            redacted_categories=redacted_categories,
            redaction_map=redaction_map,
            overrides=overrides,
        )

    async def ainvoke(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        **overrides: Any,
    ) -> Any:
        """Async version of invoke(). See invoke() for **overrides semantics."""
        self._check_budget()
        resolved, redacted_categories, redaction_map = self._resolve_prompt(prompt, prompt_variables, files)
        self.last_redacted_categories = redacted_categories
        return await self._arun_invoke(
            prepared_input=resolved,
            resolved=resolved,
            redacted_categories=redacted_categories,
            redaction_map=redaction_map,
            overrides=overrides,
        )

    # ── Validated structured output ──────────────────────────────────────────
    # Server-side json_schema strict mode (build_text_config) already
    # constrains the shape of the JSON. This adds a feedback loop on top: if
    # the result still fails Pydantic validation, the error is sent back to
    # the model and it gets another chance to correct itself.

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
        resolved, redacted_categories, redaction_map = self._resolve_prompt(prompt, prompt_variables, files)
        self.last_redacted_categories = redacted_categories
        input_data = resolved if isinstance(resolved, list) else [{"role": "user", "content": resolved}]
        return self._run_invoke_structured(
            prepared_input=input_data,
            resolved=resolved,
            redacted_categories=redacted_categories,
            redaction_map=redaction_map,
            max_validation_retries=max_validation_retries,
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
        resolved, redacted_categories, redaction_map = self._resolve_prompt(prompt, prompt_variables, files)
        self.last_redacted_categories = redacted_categories
        input_data = resolved if isinstance(resolved, list) else [{"role": "user", "content": resolved}]
        return await self._arun_invoke_structured(
            prepared_input=input_data,
            resolved=resolved,
            redacted_categories=redacted_categories,
            redaction_map=redaction_map,
            max_validation_retries=max_validation_retries,
        )

    def stream(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        **overrides: Any,
    ) -> Iterator[str]:
        """Stream text chunks synchronously. See invoke() for **overrides semantics."""
        resolved, redacted_categories, _redaction_map = self._resolve_prompt(prompt, prompt_variables, files)
        self.last_redacted_categories = redacted_categories
        return self._invoke_stream_mode(prompt_input=resolved, overrides=overrides)

    async def astream(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        **overrides: Any,
    ) -> AsyncIterator[str]:
        """Stream text chunks asynchronously. See invoke() for **overrides semantics."""
        resolved, redacted_categories, _redaction_map = self._resolve_prompt(prompt, prompt_variables, files)
        self.last_redacted_categories = redacted_categories
        async for chunk in self._ainvoke_stream_mode(prompt_input=resolved, overrides=overrides):
            yield chunk

    # ── Low-level create() / acreate() ───────────────────────────────────────

    def create(self, input_data: Any = None, **overrides: Any) -> Any:
        """Direct access to client.responses.create() with managed retries."""
        params = self._prepare_create_params(input_data, overrides)
        return self._create_raw(params)

    async def acreate(self, input_data: Any = None, **overrides: Any) -> Any:
        """Async version of create()."""
        params = self._prepare_create_params(input_data, overrides)
        return await self._acreate_raw(params)

    # ── Multi-turn chat ───────────────────────────────────────────────────────

    def chat(self, messages: List[Dict[str, Any]], **overrides: Any) -> Any:
        """
        Send a multi-turn messages list to the Responses API.

        Goes through the same pipeline as invoke() — redaction, provider
        fallback, budget cap, ledger, and shadow dispatch — just with a
        pre-built messages list instead of a single prompt.

        Args:
            messages: List of {role, content} dicts.
            **overrides: Per-call request params merged over the constructor's
                defaults for this call only. See invoke() for **overrides
                semantics.

        Returns:
            Generated text or metadata dict.
        """
        self._check_budget()
        resolved, redacted_categories, redaction_map = self._apply_redaction(messages)
        self.last_redacted_categories = redacted_categories
        return self._run_invoke(
            prepared_input=resolved,
            resolved=resolved,
            redacted_categories=redacted_categories,
            redaction_map=redaction_map,
            overrides=overrides,
            call_type="chat",
            force_non_stream=True,
        )

    async def achat(self, messages: List[Dict[str, Any]], **overrides: Any) -> Any:
        """Async version of chat(). See chat() for pipeline/**overrides semantics."""
        self._check_budget()
        resolved, redacted_categories, redaction_map = self._apply_redaction(messages)
        self.last_redacted_categories = redacted_categories
        return await self._arun_invoke(
            prepared_input=resolved,
            resolved=resolved,
            redacted_categories=redacted_categories,
            redaction_map=redaction_map,
            overrides=overrides,
            call_type="achat",
            force_non_stream=True,
        )

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
            **kwargs: tool_choice, files, plus any other per-call request
                param override (e.g. temperature=, max_output_tokens=) —
                merged over the constructor's defaults for this call only,
                the same as invoke()'s **overrides.

        Returns:
            ToolCallResponse with tool_calls (if the model called tools)
            or text (if the model gave a final answer).
        """
        files = kwargs.pop("files", None)
        tool_choice = kwargs.pop("tool_choice", "auto")
        # autourgos-agent's native loop (and any other Chat-Completions-
        # shaped caller) builds "assistant"+tool_calls / "role":"tool"
        # messages -- the Responses API has no such shapes and rejects them
        # outright. Convert BEFORE _resolve_prompt/redaction so both see
        # the already-Responses-shaped list; a no-op for a plain string or
        # an already-Responses-shaped list. See its docstring for detail.
        prompt = normalize_native_tool_calling_input(prompt)
        resolved, redacted_categories, redaction_map = self._resolve_prompt(prompt, None, files)
        self.last_redacted_categories = redacted_categories
        try:
            responses_tools = build_responses_tools(tools)
        except ValueError as exc:
            raise OpenAIResponseConfigError(str(exc)) from exc
        params = self._build_base_params(input_data=resolved, stream=False, overrides=kwargs)
        if responses_tools:
            params["tools"] = responses_tools
            params["tool_choice"] = tool_choice
        with self._budget_admission():
            with track_latency() as timing:
                raw, provider_label, provider_model, provider_pricing = self._create_across_providers(params)
            raw_calls = extract_tool_calls_from_response(raw) if responses_tools else []
            tool_calls = [
                FunctionCall(
                    name=c["name"], arguments=c["arguments"], call_id=c["call_id"],
                    arguments_parse_error=c.get("arguments_parse_error"),
                )
                for c in raw_calls
            ]
            masked_text = extract_text_from_response(raw) if not tool_calls else None
            text = restore_text(masked_text, redaction_map) if (
                masked_text and self.redact_restore_in_response
            ) else masked_text
            metadata = build_structured_output(
                model_name=provider_model,
                response_text=masked_text,
                raw_response=raw,
                latency_ms=timing["latency_ms"],
                input_pricing=provider_pricing[0],
                output_pricing=provider_pricing[1],
                extra_fields={"provider_used": provider_label},
            )
            self._record_session_cost(metadata.get("total_cost"))
            self._log_to_ledger(
                call_type="invoke_with_tools", prompt=resolved, metadata=metadata,
                redacted_categories=redacted_categories, response_override=masked_text,
            )
        if tool_calls:
            return ToolCallResponse(tool_calls=tool_calls, raw=raw)
        return ToolCallResponse(text=text, raw=raw)

    async def ainvoke_with_tools(
        self,
        prompt: Any,
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolCallResponse:
        """Async version of invoke_with_tools()."""
        files = kwargs.pop("files", None)
        tool_choice = kwargs.pop("tool_choice", "auto")
        # See invoke_with_tools()'s identical comment.
        prompt = normalize_native_tool_calling_input(prompt)
        resolved, redacted_categories, redaction_map = self._resolve_prompt(prompt, None, files)
        self.last_redacted_categories = redacted_categories
        try:
            responses_tools = build_responses_tools(tools)
        except ValueError as exc:
            raise OpenAIResponseConfigError(str(exc)) from exc
        params = self._build_base_params(input_data=resolved, stream=False, overrides=kwargs)
        if responses_tools:
            params["tools"] = responses_tools
            params["tool_choice"] = tool_choice
        async with self._async_budget_admission():
            with track_latency() as timing:
                raw, provider_label, provider_model, provider_pricing = await self._acreate_across_providers(params)
            raw_calls = extract_tool_calls_from_response(raw) if responses_tools else []
            tool_calls = [
                FunctionCall(
                    name=c["name"], arguments=c["arguments"], call_id=c["call_id"],
                    arguments_parse_error=c.get("arguments_parse_error"),
                )
                for c in raw_calls
            ]
            masked_text = extract_text_from_response(raw) if not tool_calls else None
            text = restore_text(masked_text, redaction_map) if (
                masked_text and self.redact_restore_in_response
            ) else masked_text
            metadata = build_structured_output(
                model_name=provider_model,
                response_text=masked_text,
                raw_response=raw,
                latency_ms=timing["latency_ms"],
                input_pricing=provider_pricing[0],
                output_pricing=provider_pricing[1],
                extra_fields={"provider_used": provider_label},
            )
            self._record_session_cost(metadata.get("total_cost"))
            self._log_to_ledger(
                call_type="ainvoke_with_tools", prompt=resolved, metadata=metadata,
                redacted_categories=redacted_categories, response_override=masked_text,
            )
        if tool_calls:
            return ToolCallResponse(tool_calls=tool_calls, raw=raw)
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
    "OpenAIResponseDeadlineExceededError",
]
