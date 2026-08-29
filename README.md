# autourgos-responses

LLM wrapper for the **OpenAI Responses API** (`client.responses.create`), part of the [Autourgos](https://github.com/devxjitin) framework.

No `autourgos-core` dependency required. As of v2.0.0, this package depends on `autourgos-openaichat` (which brings in the shared LLM base layer — `BaseLLM`, circuit breaker, etc.) in addition to `openai`. Just `pip install autourgos-responses` and you are ready.

The Responses API is OpenAI's newer, stateful endpoint that supports reasoning models (`o3`, `o3-mini`, `o1`), built-in tools, and multi-turn input natively.

---

## Why use this?

Almost every major LLM provider today — Groq, Together AI, Mistral, Perplexity, DeepSeek, Ollama, LM Studio, vLLM, Azure OpenAI — exposes an **OpenAI-compatible API**. This means they all accept the same request format.

`autourgos-responses` takes advantage of this. You set `base_url` to any provider's endpoint and `model` to whatever model they offer. **One package, any LLM.** You never have to learn a new SDK or rewrite your code when you switch providers.

The Responses API gives you extra power on top: native reasoning models (`o3`, `o3-mini`, `o1`) with configurable thinking effort, text verbosity control, and cleaner multi-turn conversation handling.

```
OpenAI (gpt-4o, o3, o3-mini, o1) ──────────┐
Groq (Llama, Mixtral, Gemma) ───────────────┤
Together AI (70B, 8x7B, ...) ───────────────┤  autourgos-responses
Mistral AI (mistral-large, ...) ────────────┤  (one interface)
DeepSeek (deepseek-chat, ...) ──────────────┤
Perplexity (sonar models) ──────────────────┤
Ollama — any local model ───────────────────┤
LM Studio — any local model ────────────────┤
vLLM — self-hosted ─────────────────────────┤
Azure OpenAI ───────────────────────────────┘
```

---

## Table of Contents

- [Install](#install)
- [Works With Any LLM](#works-with-any-llm)
- [Quick Start](#quick-start)
- [Basic Text Generation](#basic-text-generation)
- [Async Generation](#async-generation)
- [Streaming](#streaming)
- [Async Streaming](#async-streaming)
- [Batch Invocation](#batch-invocation)
- [System Instruction](#system-instruction)
- [Prompt Templates](#prompt-templates)
- [Reasoning Models](#reasoning-models)
- [Multi-Modal Vision Input](#multi-modal-vision-input)
- [Structured Output](#structured-output)
- [JSON Mode](#json-mode)
- [Multi-Turn Chat](#multi-turn-chat)
- [Cost Tracking](#cost-tracking)
- [Context Manager](#context-manager)
- [Circuit Breaker](#circuit-breaker)
- [Error Handling](#error-handling)
- [Constructor Reference](#constructor-reference)
- [What Each Method Returns](#what-each-method-returns)
- [Differences vs autourgos-openaichat](#differences-vs-autourgos-openaichat)

---

## Install

```bash
pip install autourgos-responses
```

Requires Python 3.10+ and `openai>=1.0.0`.

---

## Works With Any LLM

All you need to switch providers is `base_url` and the right model name. Your API key comes from the provider you choose.

### OpenAI (default)

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    api_key="sk-...",           # or set OPENAI_API_KEY env var
)
reply = llm.invoke("What is the capital of France?")
print(reply)
# Paris
```

### OpenAI reasoning models

These are special to OpenAI's Responses API. They support `reasoning_effort` to control how long the model thinks before answering.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="o3-mini",
    api_key="sk-...",
    reasoning_effort="high",   # "low", "medium", or "high"
)
reply = llm.invoke("Prove that the square root of 2 is irrational.")
print(reply)
# Assume for contradiction that √2 = p/q in lowest terms...
```

### Groq — fastest inference, free tier available

Groq runs open-source models (Llama 3, Mixtral, Gemma) at extremely high speed. Get your key at https://console.groq.com.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="llama3-70b-8192",
    api_key="gsk_...",          # Groq API key
    base_url="https://api.groq.com/openai/v1",
)
reply = llm.invoke("Explain quantum entanglement simply.")
print(reply)
# Quantum entanglement is when two particles become linked so that
# the state of one instantly affects the other, no matter how far apart they are.
```

Other Groq models: `llama3-8b-8192`, `mixtral-8x7b-32768`, `gemma2-9b-it`

### Together AI — wide model selection

Together AI hosts hundreds of open-source models. Get your key at https://api.together.xyz.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="...",              # Together AI key
    base_url="https://api.together.xyz/v1",
)
reply = llm.invoke("Write a Python function to check if a number is prime.")
print(reply)
# def is_prime(n: int) -> bool:
#     if n < 2:
#         return False
#     for i in range(2, int(n**0.5) + 1):
#         if n % i == 0:
#             return False
#     return True
```

Other Together AI models: `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen2-72B-Instruct`

### Mistral AI

Get your key at https://console.mistral.ai.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="mistral-large-latest",
    api_key="...",              # Mistral API key
    base_url="https://api.mistral.ai/v1",
)
reply = llm.invoke("What are the benefits of test-driven development?")
print(reply)
# TDD helps you write cleaner code, catch bugs early, and gives
# you confidence to refactor without breaking existing behaviour.
```

Other Mistral models: `mistral-medium-latest`, `mistral-small-latest`, `open-mixtral-8x7b`

### DeepSeek

Get your key at https://platform.deepseek.com.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="deepseek-chat",
    api_key="...",              # DeepSeek API key
    base_url="https://api.deepseek.com/v1",
)
reply = llm.invoke("What is a transformer neural network?")
print(reply)
# A transformer is a neural network architecture that uses self-attention
# to process input sequences in parallel, making it highly effective for
# NLP tasks like translation, summarisation, and text generation.
```

Other DeepSeek models: `deepseek-reasoner`

### Perplexity — web-connected models

Perplexity's Sonar models can search the web in real time. Get your key at https://www.perplexity.ai/settings/api.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="llama-3.1-sonar-large-128k-online",
    api_key="pplx-...",        # Perplexity API key
    base_url="https://api.perplexity.ai",
)
reply = llm.invoke("What are the top AI news stories today?")
print(reply)
# Today's top AI stories include...
```

### Ollama — run any model locally, no internet needed

Ollama runs models entirely on your machine. Install from https://ollama.com, then pull a model:

```bash
ollama pull llama3
```

No API key needed for local use.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="llama3",
    api_key="ollama",           # can be any string — Ollama ignores it
    base_url="http://localhost:11434/v1",
)
reply = llm.invoke("What is the difference between RAM and ROM?")
print(reply)
# RAM (Random Access Memory) is fast, temporary storage your computer uses
# while running programs. ROM (Read-Only Memory) is permanent storage that
# holds firmware your computer needs to boot up.
```

Other Ollama models: `mistral`, `phi3`, `gemma2`, `codellama`, `qwen2` — anything you pull with `ollama pull`.

### LM Studio — local models with a GUI

LM Studio lets you download and run GGUF models locally. Start the local server in LM Studio, then:

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="local-model",        # use whatever model name LM Studio shows
    api_key="lm-studio",        # any string — ignored locally
    base_url="http://localhost:1234/v1",
)
reply = llm.invoke("Explain recursion in simple terms.")
print(reply)
# Recursion is when a function calls itself to solve a smaller version
# of the same problem, until it reaches a base case that stops the loop.
```

### vLLM — self-hosted high-throughput serving

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="EMPTY",            # vLLM default when no auth is set
    base_url="http://your-server:8000/v1",
)
reply = llm.invoke("What is the capital of Japan?")
print(reply)
# Tokyo
```

### Azure OpenAI

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",             # your deployment name in Azure
    api_key="...",              # Azure OpenAI key
    base_url="https://<your-resource>.openai.azure.com/openai/deployments/gpt-4o",
)
reply = llm.invoke("What is cloud computing?")
print(reply)
# Cloud computing is the delivery of computing services over the internet
# on a pay-as-you-go basis.
```

### Switching providers at runtime

```python
from autourgos_responses import OpenAIResponse

PROVIDERS = {
    "openai": {
        "model": "gpt-4o-mini",
        "api_key": "sk-...",
        "base_url": None,
    },
    "groq": {
        "model": "llama3-8b-8192",
        "api_key": "gsk_...",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "ollama": {
        "model": "llama3",
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
    },
}

for name, cfg in PROVIDERS.items():
    llm = OpenAIResponse(**cfg)
    reply = llm.invoke("Say hello in one word.")
    print(f"{name}: {reply}")

# openai: Hello!
# groq:   Hello!
# ollama: Hello!
```

---

## Quick Start

Set the `OPENAI_API_KEY` environment variable first, or pass `api_key=` directly.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")
reply = llm.invoke("What is the capital of France?")
print(reply)
# Paris
```

---

## Basic Text Generation

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    api_key="sk-...",        # or set OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=256,
)

reply = llm.invoke("Explain machine learning in one sentence.")
print(reply)
# Machine learning is a branch of AI where systems learn from data
# to make predictions or decisions without being explicitly programmed.
```

---

## Async Generation

```python
import asyncio
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")

async def main():
    reply = await llm.ainvoke("What is the speed of light?")
    print(reply)
    # The speed of light in a vacuum is approximately 299,792,458 metres per second.

asyncio.run(main())
```

---

## Streaming

Stream the response token by token synchronously.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")

for chunk in llm.stream("Write a haiku about mountains."):
    print(chunk, end="", flush=True)

# Silent peaks above,
# Clouds drift through the ancient stone,
# Eagles trace the wind.
```

You can also enable streaming at construction time so `invoke()` internally streams and returns the full joined text:

```python
llm = OpenAIResponse(model="gpt-4o", streaming=True)
reply = llm.invoke("Tell me a fun fact about space.")
print(reply)
# A day on Venus is longer than a year on Venus — it takes 243 Earth days
# to rotate once but only 225 Earth days to orbit the Sun.
```

---

## Async Streaming

```python
import asyncio
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")

async def main():
    async for chunk in llm.astream("Count prime numbers up to 20."):
        print(chunk, end="", flush=True)
    # 2, 3, 5, 7, 11, 13, 17, 19

asyncio.run(main())
```

---

## Batch Invocation

### Synchronous (sequential)

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o-mini")

prompts = [
    "Capital of Japan?",
    "Capital of Germany?",
    "Capital of Brazil?",
]

results = llm.batch_invoke(prompts)
for prompt, result in zip(prompts, results):
    print(f"{prompt} -> {result}")

# Capital of Japan?   -> Tokyo
# Capital of Germany? -> Berlin
# Capital of Brazil?  -> Brasilia
```

### Async (concurrent)

```python
import asyncio
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o-mini")

async def main():
    results = await llm.abatch_invoke([
        "Capital of Japan?",
        "Capital of Germany?",
        "Capital of Brazil?",
    ])
    print(results)
    # ['Tokyo', 'Berlin', 'Brasilia']

asyncio.run(main())
```

---

## System Instruction

Set a persistent system prompt sent as the `instructions` field of every request.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    system_prompt="You are a concise assistant. Always reply in exactly one sentence.",
)

reply = llm.invoke("What is photosynthesis?")
print(reply)
# Photosynthesis is the process by which plants use sunlight, water, and CO2
# to produce glucose and oxygen.
```

---

## Prompt Templates

Define a reusable template with `{placeholders}` and fill them at call time.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    prompt_template="Summarise the following {topic} in {num_words} words:\n\n{content}",
)

reply = llm.invoke(prompt_variables={
    "topic": "article",
    "num_words": "30",
    "content": "Quantum computing uses quantum bits (qubits) that can exist in superposition...",
})
print(reply)
# Quantum computing uses qubits in superposition to perform many calculations
# simultaneously, offering vastly superior speeds for specific complex problems
# like cryptography and molecular simulation.
```

Missing variables raise a clear error:

```python
llm.invoke(prompt_variables={"topic": "article"})
# ValueError: Missing prompt template variables: content, num_words
```

---

## Reasoning Models

`o3`, `o3-mini`, and `o1` are OpenAI's reasoning models. They support `reasoning_effort` to control how long the model thinks before answering. Higher effort produces better answers for hard problems but takes longer and costs more.

> Reasoning models are only available from OpenAI. When using other providers, omit `reasoning_effort`.

### reasoning_effort

Valid values: `"low"`, `"medium"`, `"high"`.

```python
from autourgos_responses import OpenAIResponse

# Low effort — fast, cheaper
llm = OpenAIResponse(model="o3-mini", reasoning_effort="low")
reply = llm.invoke("What is 17 × 23?")
print(reply)
# 391

# Medium effort — balanced
llm = OpenAIResponse(model="o3-mini", reasoning_effort="medium")
reply = llm.invoke("Solve: if a train travels at 80 km/h for 2.5 hours, how far does it go?")
print(reply)
# The train travels 200 km. (80 km/h × 2.5 h = 200 km)

# High effort — most thorough, best for hard problems
llm = OpenAIResponse(model="o3", reasoning_effort="high")
reply = llm.invoke(
    "Prove that the square root of 2 is irrational."
)
print(reply)
# Assume for contradiction that √2 = p/q where p and q are integers with no common factors...
```

### When to use each level

| effort | Use for | Speed | Cost |
|---|---|---|---|
| `"low"` | Simple maths, factual Q&A, quick summaries | Very fast | Lowest |
| `"medium"` | Multi-step reasoning, code generation | Moderate | Medium |
| `"high"` | Hard proofs, complex analysis, frontier research | Slow | Highest |

### Invalid effort raises immediately

```python
OpenAIResponse(model="o3-mini", reasoning_effort="ultra")
# ValueError: Invalid reasoning_effort 'ultra'. Must be one of: ['high', 'low', 'medium']
```

---

## Multi-Modal Vision Input

Pass image files, URLs, or raw bytes alongside text.

> Note: vision support depends on the provider and model. GPT-4o, LLaVA (Ollama), and several others support it.

### From a file path

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")
reply = llm.invoke("What objects are in this image?", files=["photo.jpg"])
print(reply)
# The image shows a wooden desk with a laptop, a coffee mug, and an open notebook.
```

### From a URL

```python
reply = llm.invoke(
    "Describe this chart in detail.",
    files=["https://example.com/sales-chart.png"],
)
print(reply)
# The chart is a bar graph comparing quarterly revenue across four product lines.
# Q3 shows the highest sales at approximately $2.4M for Product A...
```

### From raw bytes

```python
with open("diagram.png", "rb") as f:
    image_bytes = f.read()

reply = llm.invoke("Explain this architecture diagram.", files=[image_bytes])
print(reply)
# The diagram shows a microservices architecture with an API gateway at the top
# routing requests to three downstream services: Auth, Orders, and Payments...
```

### Multiple images

```python
reply = llm.invoke(
    "Which image shows more people?",
    files=["crowd1.jpg", "crowd2.jpg"],
)
print(reply)
# The first image shows more people — it appears to be a large outdoor concert
# with thousands of attendees, while the second shows a small group of around 20.
```

---

## Structured Output

Return data that matches a Pydantic model automatically.

```python
from pydantic import BaseModel, Field
from autourgos_responses import OpenAIResponse

class WeatherReport(BaseModel):
    city: str = Field(description="Name of the city")
    temperature_celsius: float = Field(description="Current temperature in Celsius")
    condition: str = Field(description="Weather condition e.g. Sunny, Rainy")
    humidity_percent: int = Field(description="Humidity percentage 0-100")

llm = OpenAIResponse(model="gpt-4o", output_schema=WeatherReport)
result = llm.invoke("Describe a typical summer day in London.")

import json
data = json.loads(result["response"])
print(data)
# {
#   "city": "London",
#   "temperature_celsius": 22.0,
#   "condition": "Partly Cloudy",
#   "humidity_percent": 65
# }
```

Use a plain dict schema instead of Pydantic:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
    },
    "required": ["name", "age"],
}

llm = OpenAIResponse(model="gpt-4o", output_schema=schema)
result = llm.invoke("Invent a fictional person.")
print(result["response"])
# {"name": "Mira Caldwell", "age": 34}
```

---

## JSON Mode

Force valid JSON output without defining a schema.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    response_mime_type="application/json",
    system_prompt="Always respond with valid JSON only.",
)

reply = llm.invoke("List three programming languages with their year of creation.")
print(reply)
# {
#   "languages": [
#     {"name": "Python",     "year": 1991},
#     {"name": "JavaScript", "year": 1995},
#     {"name": "Rust",       "year": 2010}
#   ]
# }
```

---

## Multi-Turn Chat

Pass a list of role-tagged messages directly to carry conversation history.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")

messages = [
    {"role": "user",      "content": "My favourite colour is blue."},
    {"role": "assistant", "content": "That is a great choice! Blue is calming and versatile."},
    {"role": "user",      "content": "What is my favourite colour?"},
]

reply = llm.chat(messages)
print(reply)
# Your favourite colour is blue!
```

### Async multi-turn

```python
import asyncio
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")

async def main():
    messages = [
        {"role": "user",      "content": "I work as a data scientist."},
        {"role": "assistant", "content": "That is a fascinating field!"},
        {"role": "user",      "content": "What is my job?"},
    ]
    reply = await llm.achat(messages)
    print(reply)
    # You work as a data scientist.

asyncio.run(main())
```

### Building a conversation loop

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")
history = []

def chat(user_message: str) -> str:
    history.append({"role": "user", "content": user_message})
    reply = llm.chat(history)
    history.append({"role": "assistant", "content": reply})
    return reply

print(chat("My name is Jitin."))
# Nice to meet you, Jitin!

print(chat("I am building an AI framework called Autourgos."))
# That sounds exciting! What does Autourgos focus on?

print(chat("What is my name and what am I building?"))
# Your name is Jitin, and you are building an AI framework called Autourgos.
```

---

## Cost Tracking

Pass pricing (USD per 1 million tokens) to get cost breakdowns.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    input_pricing=2.50,    # $2.50 per 1M input tokens
    output_pricing=10.00,  # $10.00 per 1M output tokens
    structured_output=True,
)

result = llm.invoke("Summarise the history of the internet in 3 sentences.")
print(result["model"])          # gpt-4o
print(result["response"])       # The internet began as ARPANET in the 1960s...
print(result["input_tokens"])   # 21
print(result["output_tokens"])  # 68
print(result["total_tokens"])   # 89
print(result["input_cost"])     # 0.0000525
print(result["output_cost"])    # 0.00068
print(result["total_cost"])     # 0.0007325
print(result["latency_ms"])     # 1102.4
```

Access the last call metadata without `structured_output=True`:

```python
llm = OpenAIResponse(model="gpt-4o", input_pricing=2.50, output_pricing=10.00)
reply = llm.invoke("Hello!")
print(llm.last_metadata)
# {
#   "model": "gpt-4o",
#   "response": "Hello! How can I help you today?",
#   "input_tokens": 9,
#   "output_tokens": 10,
#   "total_tokens": 19,
#   "input_cost": 0.0000225,
#   "output_cost": 0.0001,
#   "total_cost": 0.0001225,
#   "latency_ms": 921.7
# }
```

---

## Context Manager

Automatically closes the HTTP client when done.

```python
from autourgos_responses import OpenAIResponse

with OpenAIResponse(model="gpt-4o") as llm:
    reply = llm.invoke("Quick question: what is 2 + 2?")
    print(reply)
    # 4
# Client is closed automatically here
```

Async context manager:

```python
import asyncio
from autourgos_responses import OpenAIResponse

async def main():
    async with OpenAIResponse(model="gpt-4o") as llm:
        reply = await llm.ainvoke("What year did the Berlin Wall fall?")
        print(reply)
        # The Berlin Wall fell in 1989.

asyncio.run(main())
```

---

## Circuit Breaker

Protects against cascading failures. After `circuit_failure_threshold` consecutive API errors, all calls are blocked for `circuit_cooldown_time` seconds, then automatically reset.

This is useful when you are using a local model (Ollama, LM Studio) or a rate-limited API — if the server goes down, the circuit breaker stops your code from hammering it with failed requests.

```python
from autourgos_responses import OpenAIResponse, CircuitBreakerOpenException

llm = OpenAIResponse(
    model="gpt-4o",
    circuit_failure_threshold=3,   # open after 3 consecutive failures
    circuit_cooldown_time=60.0,    # block calls for 60 seconds
)

try:
    reply = llm.invoke("Hello!")
    print(reply)
except CircuitBreakerOpenException as e:
    print(f"Circuit is open, skipping call: {e}")
    # Circuit breaker OPEN for OpenAIResponse — 3 consecutive failures.
    # Blocked until 1718500060.0.
```

After the cooldown expires, the next call is allowed through as a probe. If it succeeds, the circuit resets to closed. If it fails again, the cooldown restarts.

---

## Low-Level Access

Direct access to the raw Responses API object when you need full control.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")

raw = llm.create("Explain gravity briefly.")
print(raw.output_text)
print(raw.usage.input_tokens)
print(raw.usage.output_tokens)
```

Async:

```python
raw = await llm.acreate("Explain gravity briefly.")
print(raw.output_text)
```

With overrides:

```python
raw = llm.create(
    "Summarise this.",
    temperature=0.3,
    max_output_tokens=50,
)
```

---

## Error Handling

```python
from autourgos_responses import (
    OpenAIResponse,
    OpenAIResponseAPIError,
    OpenAIResponseResponseError,
    OpenAIResponseConfigError,
    OpenAIResponseImportError,
    CircuitBreakerOpenException,
)

llm = OpenAIResponse(model="gpt-4o")

try:
    reply = llm.invoke("Hello!")
    print(reply)
except OpenAIResponseAPIError as e:
    # All retries exhausted — network issue or rate limit
    print(f"API error after retries: {e}")
except OpenAIResponseResponseError as e:
    # Response was received but no text could be extracted
    print(f"Could not parse response: {e}")
except OpenAIResponseConfigError as e:
    # Incompatible options e.g. streaming=True + structured_output=True
    print(f"Configuration error: {e}")
except OpenAIResponseImportError as e:
    # openai package is not installed
    print(f"openai not installed: {e}")
except CircuitBreakerOpenException as e:
    # Too many recent failures — circuit is open
    print(f"Circuit breaker is open: {e}")
```

### Retry behaviour

By default the wrapper retries up to 3 times with exponential back-off:

| Attempt | Wait before retry |
|---|---|
| 1st failure | 0.5 s |
| 2nd failure | 1.0 s |
| 3rd failure | 2.0 s |
| 4th failure | raises `OpenAIResponseAPIError` |

Change with `max_retries` and `backoff_factor`:

```python
llm = OpenAIResponse(
    model="gpt-4o",
    max_retries=5,
    backoff_factor=1.0,   # waits: 1s, 2s, 4s, 8s then raises
)
```

---

## Constructor Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Model name. e.g. `"gpt-4o"`, `"o3-mini"`, `"llama3-70b-8192"` |
| `api_key` | `str` | `OPENAI_API_KEY` env | API key for the provider you are using |
| `base_url` | `str` | `OPENAI_BASE_URL` env | Provider endpoint. e.g. `"https://api.groq.com/openai/v1"` or `"http://localhost:11434/v1"` |
| `organization` | `str` | `None` | OpenAI organization ID (OpenAI only) |
| `project` | `str` | `None` | OpenAI project ID (OpenAI only) |
| `system_prompt` | `str` | `None` | System prompt sent as `instructions` field |
| `prompt_template` | `str` | `None` | Template with `{variable}` placeholders |
| `temperature` | `float` | `None` | Sampling temperature 0–2 |
| `top_p` | `float` | `None` | Nucleus sampling 0–1 |
| `max_tokens` | `int` | `None` | Maximum output tokens (maps to `max_output_tokens`) |
| `reasoning_effort` | `str` | `None` | `"low"`, `"medium"`, or `"high"` — for o3, o3-mini, o1 only |
| `reasoning_summary` | `str` | `None` | Include reasoning summary in output (OpenAI only) |
| `text_verbosity` | `str` | `None` | `"low"`, `"medium"`, or `"high"` |
| `output_schema` | `BaseModel` / `dict` | `None` | Pydantic model or JSON schema for structured output |
| `response_mime_type` | `str` | `None` | `"application/json"` enables JSON object mode |
| `structured_output` | `bool` | `False` | If `True`, `invoke()` returns a metadata dict |
| `streaming` | `bool` | `False` | If `True`, `invoke()` streams internally and joins |
| `max_retries` | `int` | `3` | Retry attempts on transient API errors |
| `timeout` | `float` | `60.0` | Request timeout in seconds |
| `backoff_factor` | `float` | `0.5` | Exponential back-off base (wait = factor × 2^attempt) |
| `input_pricing` | `float` | `None` | USD per 1 million input tokens |
| `output_pricing` | `float` | `None` | USD per 1 million output tokens |
| `circuit_failure_threshold` | `int` | `5` | Consecutive failures before the circuit opens |
| `circuit_cooldown_time` | `float` | `30.0` | Seconds the circuit stays open before probing |

---

## What Each Method Returns

| Method | Returns |
|---|---|
| `invoke(prompt)` | `str` — generated text (or `dict` if `structured_output=True`) |
| `ainvoke(prompt)` | same as `invoke`, async |
| `stream(prompt)` | `Iterator[str]` — text chunks |
| `astream(prompt)` | `AsyncIterator[str]` — text chunks |
| `batch_invoke(prompts)` | `list[str]` — one result per prompt, sequential |
| `abatch_invoke(prompts)` | `list[str]` — concurrent results |
| `chat(messages)` | `str` — generated text (or `dict` if `structured_output=True`) |
| `achat(messages)` | same as `chat`, async |
| `create(input_data)` | Raw OpenAI Responses API response object |
| `acreate(input_data)` | same as `create`, async |

### Metadata dict keys (when `structured_output=True` or via `llm.last_metadata`)

| Key | Type | Description |
|---|---|---|
| `"model"` | `str` | Model name used |
| `"response"` | `str` | Generated text |
| `"input_tokens"` | `int \| None` | Input token count |
| `"output_tokens"` | `int \| None` | Output token count |
| `"total_tokens"` | `int \| None` | Total token count |
| `"input_cost"` | `float` | Input cost in USD (only if `input_pricing` set) |
| `"output_cost"` | `float` | Output cost in USD (only if `output_pricing` set) |
| `"total_cost"` | `float` | Total cost in USD (only if both pricing values set) |
| `"latency_ms"` | `float` | Request round-trip time in milliseconds |

---

## Supported Providers (quick reference)

| Provider | base_url | Notes |
|---|---|---|
| OpenAI | (default) | GPT-4o, o3, o3-mini, o1, GPT-4o-mini |
| Groq | `https://api.groq.com/openai/v1` | Llama 3, Mixtral, Gemma — very fast |
| Together AI | `https://api.together.xyz/v1` | 100+ open-source models |
| Mistral AI | `https://api.mistral.ai/v1` | mistral-large, mixtral, codestral |
| DeepSeek | `https://api.deepseek.com/v1` | deepseek-chat, deepseek-reasoner |
| Perplexity | `https://api.perplexity.ai` | Web-connected sonar models |
| Ollama | `http://localhost:11434/v1` | Runs locally, no API key needed |
| LM Studio | `http://localhost:1234/v1` | Runs locally, GUI-based |
| vLLM | `http://your-server:8000/v1` | Self-hosted, high throughput |
| Azure OpenAI | `https://<resource>.openai.azure.com/...` | Enterprise OpenAI |

---

## Differences vs autourgos-openaichat

| Feature | autourgos-openaichat | autourgos-responses |
|---|---|---|
| API endpoint | `chat.completions.create` | `responses.create` |
| System prompt field | `messages[0].role = "system"` | `instructions` parameter |
| Reasoning models | Not supported | `reasoning_effort` param for o3/o1 |
| Text verbosity control | Not supported | `text_verbosity` param |
| Multi-turn input | Messages list | Messages list or plain string |
| Native tool calling | Supported | Not yet in Responses API |
| Use when | Building chat agents, tool-calling | Using reasoning models, simple generation |

Both packages support the same providers via `base_url`. Choose based on the API endpoint your use case needs.

---

## License

MIT — Copyright (c) 2026 Jitin Kumar Sengar
