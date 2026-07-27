# Changelog

## [2.0.1] - 2026-07-27

- Fixed: standardized logger to logging.getLogger(__name__). Docs: Quick Start now notes OPENAI_API_KEY.

## [2.0.0] - 2026-07-27

BREAKING: autourgos-responses now depends on autourgos-openaichat>=1.0.1 (previously only depended on openai). `BaseLLM`, `FunctionCall`, `ToolCallResponse`, `CircuitBreakerOpenException`, and the following core/model_runtime helpers are now re-exported from autourgos-openaichat instead of being duplicated locally: `load_openai_module`, `resolve_api_key`, `resolve_base_url`, `configure_openai_client`, `configure_async_openai_client`, `release_openai_client`, `release_async_openai_client`, `normalize_model_name`, `track_latency`, `extract_template_fields`, `coerce_prompt_variable`, `configure_runtime_environment`. This eliminates a maintenance burden where circuit-breaker bugfixes had to be applied in two places. No public API/behavior change — `OpenAIResponse`'s constructor, methods, and Responses-API-specific behavior (reasoning_effort, text_verbosity, output[] parsing) are unchanged. Responses-API-specific helpers (`extract_usage_metadata`, `extract_text_from_response`, `build_structured_output`, `build_reasoning_config`, `build_text_config`, `build_multimodal_prompt`, `build_response_create_params`, `extract_text_delta_from_event`, `normalize_reasoning_effort`, `normalize_text_verbosity`) remain local because their logic differs from the Chat Completions equivalents.

## [1.0.1] - 2026-06-16

- Update Documentation
