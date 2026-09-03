# autourgos-responses — Features

A self-contained Python wrapper around the **OpenAI Responses API** (`client.responses.create`), and by extension any provider that speaks the same protocol (Azure, Groq, Gemini, Mistral, DeepSeek, Ollama, LM Studio, vLLM, OpenRouter, Together AI, Perplexity, xAI). Builds on `autourgos-openaichat`'s shared base layer (`BaseLLM`/`BaseProviderLLM`, circuit breaker) in addition to `openai`.

## Full Feature List

### Core generation
- Sync (`invoke`) and async (`ainvoke`) text generation
- Sync (`stream`) and async (`astream`) streaming, including streaming with full cost/usage recovery via `invoke(streaming=True)`
- Multi-turn conversations via `chat()`/`achat()`, or a plain message-item list passed directly as input
- Native **reasoning models** (`o3`, `o3-mini`, `o1`) with configurable `reasoning_effort` and `reasoning_summary`
- `text_verbosity` control
- System prompts (via the Responses API's `instructions` field) and `{placeholder}` prompt templates
- Multi-modal vision input — image file paths, raw bytes, or URLs, with correct MIME-type detection
- Batch invocation — `batch_invoke()` (sequential) / `abatch_invoke()` (concurrent via `asyncio.gather`)
- Native function/tool calling (`invoke_with_tools`/`ainvoke_with_tools`)
- Structured output: plain JSON mode, or a Pydantic-validated result with an automatic validation-retry loop that feeds the schema error back to the model (`invoke_structured`/`ainvoke_structured`)

### Reliability
- Automatic retries with exponential back-off, skipping non-retryable 4xx status codes
- Circuit breaker — opens after N consecutive failures, cools down for a configurable window
- Automatic provider fallback chain — an ordered list of backup providers (each with its own API key/base URL/pricing), tried in order with no external proxy/gateway required
- Optional aggregate call deadline (`max_call_duration`) capping total wall-clock time across every retry and fallback attempt for one logical call

### Cost & budget
- Built-in per-call cost and latency tracking (`last_metadata`), computed from configurable per-1M-token input/output pricing
- Session budget governor (`max_session_cost`) that hard-stops further calls once a USD cap is reached, with concurrency-safe admission

### Observability
- Optional local call ledger — SQLite file, no external service, records every call's prompt/response/tokens/cost/latency/provider
- Shadow-mode dual dispatch — send the same prompt concurrently to one or more "shadow" providers purely for comparison (similarity score, cost, latency), without ever changing what the caller receives

### Security
- Optional PII/secret redaction — heuristic pre-flight scrubber for emails, API keys, credit cards, SSNs, phone numbers, with custom regex/literal-term dictionaries and an external patterns file
- `mask` (replace and proceed) or `block` (raise instead of sending) modes, with optional reversible restore-in-response

### Advanced / escape hatches
- `extra_body=` passthrough for provider-specific fields (e.g. vLLM's `guided_json`/`guided_regex`) for constrained decoding
- `store=` passthrough for OpenAI's server-side response persistence
- Low-level `create()`/`acreate()` for direct, retry-managed access to the raw response object
- Sync/async context manager support for deterministic client cleanup
- Fully typed (`py.typed`)

## Relationship to autourgos-openaichat

Both packages implement the same reliability/cost/observability/security feature set (fallback chain, circuit breaker, budget governor, call ledger, shadow-mode dispatch, PII redaction, native tool calling, validated structured output) through a shared `BaseProviderLLM` base class — the difference is purely which endpoint each targets (`chat.completions.create` vs `responses.create`) and the handful of things unique to the Responses API (reasoning models, `reasoning_effort`, `text_verbosity`, `instructions`-based system prompts, `chat()`/`achat()`).

> Note: reasoning models and `reasoning_effort`/`text_verbosity` are OpenAI-only features of the Responses API — other OpenAI-compatible providers accept the same calls but ignore or reject those params.

---

## Competitor Comparison

Landscape research (LLM gateways, orchestration frameworks, and structured-output libraries), current as of the search date.

| Capability | **autourgos-responses** | Raw `openai` Python SDK (Responses API) | [LiteLLM](https://docs.litellm.ai/) | [LangChain](https://www.langchain.com/) | [Instructor](https://python.useinstructor.com/) | [Portkey AI Gateway](https://github.com/portkey-ai/gateway) |
|---|---|---|---|---|---|---|
| Scope | In-process Python library, no separate service | In-process Python library | Library **or** a self-hosted proxy/gateway | In-process orchestration framework | In-process library, wraps an LLM client | Hosted or self-hosted **gateway/proxy** (separate service) |
| Targets the Responses API specifically | Yes, natively (`responses.create`) | Yes | Via its OpenAI-compatible passthrough | Via a separate integration, Chat Completions is the more mature path | Depends on the wrapped client | Passthrough |
| Reasoning-model support (`o3`/`o1`, `reasoning_effort`) | Yes, first-class constructor params | Yes, raw API surface only | Passthrough only | Passthrough only | Passthrough only | Passthrough |
| Multi-provider via one interface | Yes, any OpenAI-compatible `base_url` | No (OpenAI/Azure only) | Yes, 100+ providers | Yes, via separate integration packages per provider | Depends on the wrapped client | Yes, 1,600+ models |
| Automatic retries | Yes, exponential back-off, skips non-retryable 4xx | Yes, basic exponential back-off (`max_retries`, default 2) | Yes, configurable per model/group | Not native — needs external tooling | Yes, via Tenacity, largely for validation-failure retries | Yes |
| Provider fallback chain | Yes, built into the library, no proxy needed | No | Yes — a core LiteLLM feature | Not native | No | Yes — a core feature |
| Circuit breaker | Yes, built-in | No | Not a standard built-in primitive | No | No | Partial (gateway-level health/routing) |
| Aggregate call deadline across retries+fallback | Yes (`max_call_duration`) | No (per-request `timeout` only) | Not as a single explicit cross-attempt budget | No | No | Partial (gateway request timeouts) |
| Cost tracking | Yes, per-call, configurable pricing | No | Yes, with real-time provider pricing lookup | Only via LangSmith (external service) | No | Yes, with per-team/app dashboards |
| Session budget hard-cap | Yes (`max_session_cost`), concurrency-safe | No | Yes, via virtual-key budget routing (server-side) | No native | Yes, but scoped to validation-retry token budget only | Yes, via budget/limits on virtual keys |
| Local audit ledger (no external service) | Yes, SQLite, opt-in | No | No | No — needs LangSmith/Langfuse | No | No |
| Shadow-mode dual dispatch (compare providers on live traffic, in-process) | Yes, built-in, with similarity scoring | No | Not as an in-library primitive | No | No | Not as a code-level primitive |
| PII/secret redaction | Yes, built-in, mask/block, custom dictionaries | No | No | No | No | Yes, via 50+ built-in guardrails (hosted feature) |
| Native tool/function calling | Yes | Yes (raw API surface only) | Yes (passthrough) | Yes | Not its focus | Passthrough |
| Structured output + validation-retry loop | Yes, Pydantic, automatic retry with schema-error feedback | No (raw `text.format` only) | Passthrough only | Via separate output-parser abstractions | Yes — this is Instructor's core specialty | No |
| Requires infrastructure/ops | No — pure library | No | Optional (proxy mode) or none (SDK mode) | No | No | Yes for self-hosted; hosted plan is a paid SaaS |
| Pricing | Free, open source | Free | Free, open source (self-hosted) | Free, open source (LangSmith is paid) | Free, open source | Free tier + paid plans (~$49/mo+ for production features) |

### How to read this

- **vs. the raw OpenAI SDK**: this library keeps the Responses API's native strengths (reasoning models, `instructions`, stateful-friendly shape) while adding everything the raw SDK doesn't — fallback, circuit breaking, budget caps, ledger, shadow mode, redaction, validated structured output.
- **vs. LiteLLM**: LiteLLM's specialty is provider breadth and gateway deployment, with strong fallback/cost tracking, but it treats the Responses API as one more passthrough shape rather than a first-class target with dedicated reasoning-model ergonomics; it also lacks in-process shadow-mode dispatch, a built-in circuit breaker, and PII redaction.
- **vs. LangChain**: Chat Completions-style chat models are LangChain's more mature integration path; reliability/cost/observability are largely delegated to LangSmith rather than built into the client.
- **vs. Instructor**: deepest tool for Pydantic-validated structured output specifically, but no fallback chain, circuit breaker, budget governor, ledger, or redaction, and no particular Responses-API/reasoning-model specialization.
- **vs. Portkey**: a hosted/self-hosted gateway service with broad guardrails and dashboards, but it's infrastructure to run (or a paid product at scale); autourgos-responses is a plain importable library with no service to operate, and with reasoning-model support built directly into the constructor.

Sources:
- [Portkey vs LiteLLM: Routing, Fallbacks, Cost Tracking, and Control](https://medium.com/@adnanmasood/portkey-vs-litellm-routing-fallbacks-cost-tracking-and-control-the-llm-gateway-playbook-part-195855dc25c3)
- [Fallbacks (Provider Failover) | LiteLLM docs](https://docs.litellm.ai/docs/proxy/reliability)
- [Reliability - Retries, Fallbacks | LiteLLM docs](https://docs.litellm.ai/docs/completion/reliable_completions)
- [LangChain Observability: Monitoring Guide for Production Apps](https://uptrace.dev/blog/langchain-observability)
- [Retry Logic with Tenacity - Instructor docs](https://python.useinstructor.com/concepts/retrying/)
- [Retry Mechanisms - Instructor docs](https://python.useinstructor.com/learning/validation/retry_mechanisms/)
- [Portkey AI Gateway GitHub](https://github.com/portkey-ai/gateway)
- [Best LiteLLM Alternatives in 2026](https://www.getmaxim.ai/articles/best-litellm-alternatives-in-2026/)
- [Best LLM Routing Platforms Compared (2026)](https://www.requesty.ai/blog/best-llm-routing-platforms-compared-2026-requesty-portkey-litellm-openrouter)
- [Retries - OpenAI Python SDK docs](https://openai-openai-python-73.mintlify.app/concepts/retries)
