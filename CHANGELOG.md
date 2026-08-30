# Changelog

## [2.1.0] - 2026-08-30

Feature-parity pass against `autourgos-openaichat`, ported and adapted for the Responses API's different wire format. All new features are opt-in via new constructor params — no change to existing default behavior.

- Added: native tool/function calling — `invoke_with_tools()`/`ainvoke_with_tools()` returning `ToolCallResponse`. Uses the Responses API's flat tool schema (`{"type": "function", "name", ...}`) and parses `function_call` items from `response.output[]`, rather than Chat Completions' nested `tool_calls`.
- Added: provider fallback chain — `fallback_providers=`, tried in order after the primary exhausts its retries; raises `OpenAIResponseAllProvidersFailedError` if every target fails. `provider_used` is now included in returned metadata.
- Added: session budget governor — `max_session_cost=` (requires `input_pricing`/`output_pricing`), raises `BudgetExceededException` once accumulated cost hits the cap; `reset_session_budget()` to unblock.
- Added: local SQLite call ledger — `ledger_path=`/`ledger_store_content=`, records every `invoke`/`ainvoke`/`invoke_structured`/`ainvoke_structured` call. Reuses `autourgos_openaichat.ledger` (schema is API-agnostic).
- Added: PII/secret redaction — `redact_pii=`, `redact_categories=`, `redact_mode=` ("mask"/"block"), `redact_custom_patterns=`, `redact_custom_terms=`, `redact_patterns_file=`, `redact_restore_in_response=`. Reuses `autourgos_openaichat.redaction`.
- Added: shadow-mode dual dispatch — `shadow_providers=`, `on_shadow_result=`; concurrent comparison calls with `difflib`-based similarity scoring, results in `last_shadow_results` and (if a ledger is configured) the `shadow_calls` table.
- Added: validated structured output — `invoke_structured()`/`ainvoke_structured()` return a validated Pydantic instance directly, retrying with the validation error fed back to the model on failure (`max_validation_retries=`). Raises `OpenAIResponseValidationError` on exhaustion.
- Added: `extra_body=` constrained-decoding passthrough, merged into every request.
- Changed: `autourgos-openaichat` dependency floor raised to `>=2.2.0` (needed for `ledger.py`/`redaction.py`/`shadow.py`/`BudgetExceededException`).
- Internal: retry logic refactored to `_attempt_sync_create`/`_attempt_async_create` (per-target) with `_create_across_providers`/`_acreate_across_providers` iterating fallback targets, matching `autourgos-openaichat`'s architecture.

## [2.0.3] - 2026-08-29

- Docs: added contributor badges (Sonia, Vishwanil Suman) to README, matching `autourgos-openaichat`. No functional changes.

## [2.0.2] - 2026-08-29

- Changed (breaking): relicensed from MIT to Apache License 2.0 — adds an explicit patent grant and patent-retaliation clause, matching `autourgos-openaichat`. `LICENSE` and `pyproject.toml` classifiers updated accordingly.
- Changed (breaking): `OpenAIResponse` constructor params renamed to match `autourgos-openaichat` — `system_instruction` -> `system_prompt`, `response_schema` -> `output_schema`. No backward-compat aliases; callers on the old names must update.
- Fixed: retry logic no longer retries on non-retryable client errors (HTTP 400/401/403/404/422) in the request and streaming paths (sync + async) — these now fail immediately instead of burning the full retry budget with exponential backoff.
- Fixed: `output_schema` was broken for structured output — `text.format` was nested under a `json_schema` key (the Chat Completions shape); the Responses API expects it flat. Also now enforces `additionalProperties: false` for strict json_schema mode, same as `autourgos-openaichat`.
- Fixed: `text_verbosity` accepted `{"concise","detailed","auto"}`, which the real API always rejected with a 400; corrected to the actual accepted values `{"low","medium","high"}`.
- Fixed: multi-modal `files=` input was structurally invalid — content parts were passed directly as `input` instead of wrapped in a `{"role": "user", "content": [...]}` message item, which the Responses API requires. Vision input was previously unusable.
- Docs: rewrote README.md to match `autourgos-openaichat`'s structure — badges, Features summary, Supported Providers table, full per-provider examples (added Google Gemini, xAI Grok, and OpenRouter, previously missing), reorganized under one Core Usage section with a matching Table of Contents.
- All fixes verified live against a real Azure OpenAI deployment.

## [2.0.1] - 2026-07-27

- Fixed: standardized logger to logging.getLogger(__name__). Docs: Quick Start now notes OPENAI_API_KEY.

## [2.0.0] - 2026-07-27

BREAKING: autourgos-responses now depends on autourgos-openaichat>=1.0.1 (previously only depended on openai). `BaseLLM`, `FunctionCall`, `ToolCallResponse`, `CircuitBreakerOpenException`, and the following core/model_runtime helpers are now re-exported from autourgos-openaichat instead of being duplicated locally: `load_openai_module`, `resolve_api_key`, `resolve_base_url`, `configure_openai_client`, `configure_async_openai_client`, `release_openai_client`, `release_async_openai_client`, `normalize_model_name`, `track_latency`, `extract_template_fields`, `coerce_prompt_variable`, `configure_runtime_environment`. This eliminates a maintenance burden where circuit-breaker bugfixes had to be applied in two places. No public API/behavior change — `OpenAIResponse`'s constructor, methods, and Responses-API-specific behavior (reasoning_effort, text_verbosity, output[] parsing) are unchanged. Responses-API-specific helpers (`extract_usage_metadata`, `extract_text_from_response`, `build_structured_output`, `build_reasoning_config`, `build_text_config`, `build_multimodal_prompt`, `build_response_create_params`, `extract_text_delta_from_event`, `normalize_reasoning_effort`, `normalize_text_verbosity`) remain local because their logic differs from the Chat Completions equivalents.

## [1.0.1] - 2026-06-16

- Update Documentation
