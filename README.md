# autourgos-responses

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/autourgos-responses/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-responses/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)
[![Contributor](https://img.shields.io/badge/Contributor-Sonia-blueviolet.svg)]()
[![Contributor](https://img.shields.io/badge/Contributor-Vishwanil%20Suman-blueviolet.svg)]()

A single, self-contained LLM wrapper for the **OpenAI Responses API** (`client.responses.create`), and by extension every provider that speaks the same protocol (Groq, Gemini, Azure, Ollama, and more). Part of the [Autourgos](https://github.com/devxjitin) agentic-AI framework. Depends on `autourgos-openaichat` for the shared base layer (`BaseLLM`, circuit breaker) in addition to `openai`.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")           # reads OPENAI_API_KEY
reply = llm.invoke("What is the capital of France?")
print(reply)
# Paris
```

---

## Features

- **One interface, any OpenAI-compatible provider**: OpenAI, Azure, Groq, Gemini, Mistral, DeepSeek, Ollama, and more, switched with just `base_url` + `model`
- Native reasoning models (`o3`, `o3-mini`, `o1`) with configurable `reasoning_effort` and `reasoning_summary`
- Text verbosity control (`text_verbosity`)
- Sync and async generation, plus streaming for both
- Structured output validated against a Pydantic model, or plain JSON mode
- Multi-modal vision input: file paths, URLs, or raw bytes
- Prompt templates with `{placeholder}` variables
- Multi-turn conversations via `chat()` / `achat()`, or a plain message list as input
- Automatic retries with exponential back-off (skips non-retryable 4xx errors), plus a circuit breaker for cascading-failure protection
- Built-in cost and latency tracking
- Fully typed (`py.typed`), sync/async context managers, low-level raw-response access

---

## Table of Contents

- [Install](#install)
- [Supported Providers](#supported-providers)
- [Provider Examples](#provider-examples)
  - [OpenAI](#openai)
  - [OpenAI Reasoning Models](#openai-reasoning-models)
  - [Azure OpenAI](#azure-openai)
  - [Google Gemini](#google-gemini)
  - [Groq](#groq-fastest-inference-free-tier-available)
  - [xAI (Grok)](#xai-grok)
  - [OpenRouter](#openrouter-one-key-hundreds-of-models)
  - [Together AI](#together-ai-wide-model-selection)
  - [Mistral AI](#mistral-ai)
  - [DeepSeek](#deepseek)
  - [Perplexity](#perplexity-web-connected-models)
  - [Ollama](#ollama-run-any-model-locally-no-internet-needed)
  - [LM Studio](#lm-studio-local-models-with-a-gui)
  - [vLLM](#vllm-self-hosted-high-throughput-serving)
  - [Switching providers at runtime](#switching-providers-at-runtime)
- [Core Usage](#core-usage)
  - [Text Generation](#text-generation)
  - [Async Generation](#async-generation)
  - [Streaming](#streaming)
  - [Async Streaming](#async-streaming)
  - [Batch Invocation](#batch-invocation)
  - [System Prompt](#system-prompt)
  - [Prompt Templates](#prompt-templates)
  - [Reasoning Models](#reasoning-models)
  - [Vision Input](#vision-input)
  - [Structured Output](#structured-output)
  - [JSON Mode](#json-mode)
  - [Multi-Turn Chat](#multi-turn-chat)
  - [Cost Tracking](#cost-tracking)
  - [Context Manager](#context-manager)
  - [Circuit Breaker](#circuit-breaker)
  - [Low-Level Access](#low-level-access)
  - [Error Handling](#error-handling)
- [Constructor Reference](#constructor-reference)
- [API Reference](#api-reference)
- [Differences vs autourgos-openaichat](#differences-vs-autourgos-openaichat)
- [License](#license)

---

## Install

```bash
pip install autourgos-responses
```

Requires Python 3.10+ and `openai>=1.0.0`. Structured output (`output_schema=`) additionally needs `pydantic>=2.0` if you use it.

---

## Supported Providers

Almost every major LLM provider exposes an **OpenAI-compatible API**: same request format as OpenAI's Responses endpoint. Point `base_url` at the provider and `model` at whatever they offer; nothing else changes.

| Provider | `base_url` | Get a key |
|---|---|---|
| OpenAI | *(default, omit)* | https://platform.openai.com/api-keys |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<deployment>` | Azure Portal |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | https://aistudio.google.com/apikey |
| Groq | `https://api.groq.com/openai/v1` | https://console.groq.com |
| xAI (Grok) | `https://api.x.ai/v1` | https://console.x.ai |
| OpenRouter | `https://openrouter.ai/api/v1` | https://openrouter.ai/keys |
| Together AI | `https://api.together.xyz/v1` | https://api.together.xyz |
| Mistral AI | `https://api.mistral.ai/v1` | https://console.mistral.ai |
| DeepSeek | `https://api.deepseek.com/v1` | https://platform.deepseek.com |
| Perplexity | `https://api.perplexity.ai` | https://www.perplexity.ai/settings/api |
| Ollama (local) | `http://localhost:11434/v1` | none, runs on your machine |
| LM Studio (local) | `http://localhost:1234/v1` | none, runs on your machine |
| vLLM (self-hosted) | `http://your-server:8000/v1` | none, you host it |

> Note: reasoning models (`o3`, `o3-mini`, `o1`) and `reasoning_effort`/`text_verbosity` are OpenAI-only features of the Responses API. Other providers accept the same `invoke`/`stream`/`chat` calls but ignore or reject those params.

---

## Provider Examples

Every example below is the full, runnable snippet. Swap in your own key and go.

### OpenAI

The default provider. No `base_url` needed.

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

### OpenAI Reasoning Models

`o3`, `o3-mini`, and `o1` support `reasoning_effort` to control how long the model thinks before answering.

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

### Azure OpenAI

Azure hosts OpenAI models in your own subscription. `model` is your **deployment name** in Azure, not the base model name. Get your endpoint and key from the Azure Portal.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",              # your deployment name in Azure
    api_key="...",               # Azure OpenAI key
    base_url="https://<your-resource>.openai.azure.com/openai/deployments/gpt-4o",
)
reply = llm.invoke("What is cloud computing?")
print(reply)
# Cloud computing is the delivery of computing services over the internet
# (servers, storage, databases, networking, software) on a pay-as-you-go basis.
```

### Google Gemini

Gemini exposes an OpenAI-compatible endpoint, so no separate Google SDK is needed. Get your key at https://aistudio.google.com/apikey.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gemini-2.0-flash",
    api_key="...",               # Gemini API key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
reply = llm.invoke("Explain photosynthesis in one sentence.")
print(reply)
# Photosynthesis is the process by which plants convert sunlight, water, and
# carbon dioxide into glucose and oxygen.
```

Other Gemini models: `gemini-2.0-flash-lite`, `gemini-1.5-pro`, `gemini-1.5-flash`.

### Groq (fastest inference, free tier available)

Groq runs open-source models (Llama 3, Mixtral, Gemma) at extremely high speed. Get your key at https://console.groq.com.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="llama3-70b-8192",
    api_key="gsk_...",           # Groq API key
    base_url="https://api.groq.com/openai/v1",
)
reply = llm.invoke("Explain quantum entanglement simply.")
print(reply)
# Quantum entanglement is when two particles become linked so that
# the state of one instantly affects the other, no matter how far apart they are.
```

Other Groq models: `llama3-8b-8192`, `mixtral-8x7b-32768`, `gemma2-9b-it`.

### xAI (Grok)

Get your key at https://console.x.ai.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="grok-2-latest",
    api_key="xai-...",           # xAI API key
    base_url="https://api.x.ai/v1",
)
reply = llm.invoke("What makes Mars red?")
print(reply)
# Mars appears red because its surface is covered in iron oxide (rust),
# formed when iron in the soil reacted with trace oxygen long ago.
```

### OpenRouter (one key, hundreds of models)

OpenRouter proxies dozens of providers (including Anthropic Claude and Google Gemini) behind a single OpenAI-compatible API and one API key. Get your key at https://openrouter.ai/keys.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="anthropic/claude-3.5-sonnet",   # or "google/gemini-2.0-flash-001", "openai/gpt-4o", ...
    api_key="sk-or-...",         # OpenRouter API key
    base_url="https://openrouter.ai/api/v1",
)
reply = llm.invoke("Write a Python one-liner to reverse a string.")
print(reply)
# s[::-1]
```

### Together AI (wide model selection)

Together AI hosts hundreds of open-source models. Get your key at https://api.together.xyz.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="...",                # Together AI key
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

Other Together AI models: `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen2-72B-Instruct`.

### Mistral AI

Get your key at https://console.mistral.ai.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="mistral-large-latest",
    api_key="...",                # Mistral API key
    base_url="https://api.mistral.ai/v1",
)
reply = llm.invoke("What are the benefits of test-driven development?")
print(reply)
# TDD helps you write cleaner code, catch bugs early, and gives
# you confidence to refactor without breaking existing behaviour.
```

Other Mistral models: `mistral-medium-latest`, `mistral-small-latest`, `open-mixtral-8x7b`.

### DeepSeek

Get your key at https://platform.deepseek.com.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="deepseek-chat",
    api_key="...",                # DeepSeek API key
    base_url="https://api.deepseek.com/v1",
)
reply = llm.invoke("What is a transformer neural network?")
print(reply)
# A transformer is a neural network architecture that uses self-attention
# to process input sequences in parallel, making it highly effective for
# NLP tasks like translation, summarisation, and text generation.
```

Other DeepSeek models: `deepseek-reasoner`.

### Perplexity (web-connected models)

Perplexity's Sonar models can search the web in real time. Get your key at https://www.perplexity.ai/settings/api.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="llama-3.1-sonar-large-128k-online",
    api_key="pplx-...",           # Perplexity API key
    base_url="https://api.perplexity.ai",
)
reply = llm.invoke("What is the latest version of Python?")
print(reply)
# Python 3.13.x is the latest stable release as of 2025...
```

### Ollama (run any model locally, no internet needed)

Ollama runs models entirely on your machine. Install from https://ollama.com, then pull a model:

```bash
ollama pull llama3
```

No API key needed for local use.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="llama3",
    api_key="ollama",             # can be any string, Ollama ignores it
    base_url="http://localhost:11434/v1",
)
reply = llm.invoke("What is machine learning?")
print(reply)
# Machine learning is a subset of AI where algorithms learn patterns
# from data to make predictions or decisions without explicit programming.
```

Other Ollama models: `mistral`, `phi3`, `gemma2`, `codellama`, `qwen2`, and anything you pull with `ollama pull`.

### LM Studio (local models with a GUI)

LM Studio lets you download and run GGUF models locally. Start the local server in LM Studio, then:

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="local-model",          # use whatever model name LM Studio shows
    api_key="lm-studio",          # any string, ignored locally
    base_url="http://localhost:1234/v1",
)
reply = llm.invoke("Tell me a short joke.")
print(reply)
# Why do programmers prefer dark mode? Because light attracts bugs!
```

### vLLM (self-hosted high-throughput serving)

vLLM lets you host your own models with high throughput. After starting your vLLM server:

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="EMPTY",              # vLLM's default when no auth is configured
    base_url="http://your-server:8000/v1",
)
reply = llm.invoke("What is the capital of Japan?")
print(reply)
# Tokyo
```

### Switching providers at runtime

Because all these providers use the same interface, switching is trivial:

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
    "gemini": {
        "model": "gemini-2.0-flash",
        "api_key": "...",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

for name, cfg in PROVIDERS.items():
    llm = OpenAIResponse(**cfg)
    reply = llm.invoke("Say hello in one word.")
    print(f"{name}: {reply}")

# openai: Hello!
# groq:   Hello!
# gemini: Hello!
```

---

## Core Usage

### Text Generation

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    api_key="sk-...",             # or set OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=256,
)

reply = llm.invoke("Explain machine learning in one sentence.")
print(reply)
# Machine learning is a branch of AI where systems learn from data
# to make predictions or decisions without being explicitly programmed.
```

### Async Generation

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

### Streaming

Stream the response token by token, synchronously.

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

### Async Streaming

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

### Batch Invocation

Synchronous (sequential):

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

Async (concurrent):

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

### System Prompt

Set a persistent system prompt sent as the `instructions` field of every request.

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(
    model="gpt-4o",
    system_prompt="You are a pirate. Always respond in pirate speak.",
)

reply = llm.invoke("What time is it?")
print(reply)
# Arrr, I know not the exact hour, but the sun be high in the sky, matey!
```

### Prompt Templates

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

### Reasoning Models

`o3`, `o3-mini`, and `o1` are OpenAI's reasoning models. They support `reasoning_effort` to control how long the model thinks before answering. Higher effort produces better answers for hard problems but takes longer and costs more.

> Reasoning models and `reasoning_effort`/`reasoning_summary`/`text_verbosity` are OpenAI-only. When using other providers, omit these params.

```python
from autourgos_responses import OpenAIResponse

# Low effort — fast, cheaper
llm = OpenAIResponse(model="o3-mini", reasoning_effort="low")
reply = llm.invoke("What is 17 x 23?")
print(reply)
# 391

# Medium effort — balanced
llm = OpenAIResponse(model="o3-mini", reasoning_effort="medium")
reply = llm.invoke("Solve: if a train travels at 80 km/h for 2.5 hours, how far does it go?")
print(reply)
# The train travels 200 km. (80 km/h x 2.5 h = 200 km)

# High effort — most thorough, best for hard problems
llm = OpenAIResponse(model="o3", reasoning_effort="high")
reply = llm.invoke("Prove that the square root of 2 is irrational.")
print(reply)
# Assume for contradiction that √2 = p/q where p and q are integers with no common factors...
```

| effort | Use for | Speed | Cost |
|---|---|---|---|
| `"low"` | Simple maths, factual Q&A, quick summaries | Very fast | Lowest |
| `"medium"` | Multi-step reasoning, code generation | Moderate | Medium |
| `"high"` | Hard proofs, complex analysis, frontier research | Slow | Highest |

Text verbosity is controlled separately with `text_verbosity` (`"low"`, `"medium"`, or `"high"`):

```python
llm = OpenAIResponse(model="gpt-4o", text_verbosity="low")
reply = llm.invoke("Explain how a car engine works.")
```

Invalid values raise immediately:

```python
OpenAIResponse(model="o3-mini", reasoning_effort="ultra")
# ValueError: Invalid reasoning_effort 'ultra'. Must be one of: ['high', 'low', 'medium']
```

### Vision Input

Pass image files, URLs, or raw bytes alongside text.

> Note: vision support depends on the provider and model. GPT-4o, Gemini, LLaVA (on Ollama), and several others support it.

> **Warning:** the file-path branch reads whatever local path it's given and base64-embeds its contents into the outgoing API request, with no path validation. Do not pass LLM- or tool-controlled paths through unchecked. An unchecked path could be used to exfiltrate arbitrary local files.

From a file path:

```python
from autourgos_responses import OpenAIResponse

llm = OpenAIResponse(model="gpt-4o")
reply = llm.invoke("What objects are in this image?", files=["photo.jpg"])
print(reply)
# The image shows a wooden desk with a laptop, a coffee mug, and an open notebook.
```

From a URL:

```python
reply = llm.invoke(
    "Describe this chart in detail.",
    files=["https://example.com/sales-chart.png"],
)
print(reply)
# The chart is a bar graph comparing quarterly revenue across four product lines.
# Q3 shows the highest sales at approximately $2.4M for Product A...
```

From raw bytes:

```python
with open("diagram.png", "rb") as f:
    image_bytes = f.read()

reply = llm.invoke("Explain this architecture diagram.", files=[image_bytes])
print(reply)
# The diagram shows a microservices architecture with an API gateway at the top
# routing requests to three downstream services: Auth, Orders, and Payments...
```

Multiple images:

```python
reply = llm.invoke(
    "Which image shows more people?",
    files=["crowd1.jpg", "crowd2.jpg"],
)
print(reply)
# The first image shows more people — it appears to be a large outdoor concert
# with thousands of attendees, while the second shows a small group of around 20.
```

### Structured Output

Return a Pydantic model as JSON automatically.

```python
from pydantic import BaseModel, Field
from autourgos_responses import OpenAIResponse
import json

class WeatherReport(BaseModel):
    city: str = Field(description="Name of the city")
    temperature_celsius: float = Field(description="Current temperature in Celsius")
    condition: str = Field(description="Weather condition e.g. Sunny, Rainy")
    humidity_percent: int = Field(description="Humidity percentage 0-100")

llm = OpenAIResponse(model="gpt-4o", output_schema=WeatherReport)
result = llm.invoke("Describe a typical summer day in London.")

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

### JSON Mode

Force the model to return valid JSON without a schema.

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

### Multi-Turn Chat

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

Async multi-turn:

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

Building a conversation loop:

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

### Cost Tracking

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
print(result["response"])       # The internet began as ARPANET...
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

### Context Manager

Automatically closes the HTTP client when done.

```python
from autourgos_responses import OpenAIResponse

with OpenAIResponse(model="gpt-4o") as llm:
    reply = llm.invoke("Quick question: what is 2 + 2?")
    print(reply)
    # 4
# Client is closed here automatically
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

### Circuit Breaker

Protects against cascading failures. After `circuit_failure_threshold` consecutive API errors, all calls are blocked for `circuit_cooldown_time` seconds.

This is useful when you are using a local model (Ollama, LM Studio) or a rate-limited API. If the server goes down, the circuit breaker stops your code from hammering it with failed requests.

```python
from autourgos_responses import OpenAIResponse, CircuitBreakerOpenException

llm = OpenAIResponse(
    model="gpt-4o",
    circuit_failure_threshold=3,   # open after 3 consecutive failures
    circuit_cooldown_time=60.0,    # block for 60 seconds
)

try:
    reply = llm.invoke("Hello!")
except CircuitBreakerOpenException as e:
    print(f"Circuit is open: {e}")
    # Circuit breaker OPEN for OpenAIResponse: 3 consecutive failures.
    # Blocked until 1718500000.0.
```

The circuit automatically resets after the cooldown and allows one probe call through.

### Low-Level Access

Direct access to the raw Responses API response object when you need full control.

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

### Error Handling

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
except OpenAIResponseAPIError as e:
    # API request failed after all retries (or immediately on a non-retryable 4xx)
    print(f"API error: {e}")
except OpenAIResponseResponseError as e:
    # Response was received but text could not be extracted
    print(f"Response parse error: {e}")
except OpenAIResponseConfigError as e:
    # Incompatible options (e.g. streaming + structured_output)
    print(f"Config error: {e}")
except OpenAIResponseImportError as e:
    # openai SDK not installed
    print(f"Import error: {e}")
except CircuitBreakerOpenException as e:
    # Too many recent failures, circuit is open
    print(f"Circuit open: {e}")
```

Retry behaviour: by default the wrapper retries up to 3 times with exponential back-off, but fails immediately (no retry) on non-retryable client errors — HTTP 400, 401, 403, 404, 422.

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
| `model` | `str` | required | Model name. e.g. `"gpt-4o"`, `"o3-mini"`, `"llama3-70b-8192"`, `"gemini-2.0-flash"` |
| `api_key` | `str` | `OPENAI_API_KEY` env | API key for the provider you are using |
| `base_url` | `str` | `OPENAI_BASE_URL` env | Provider endpoint. e.g. `"https://api.groq.com/openai/v1"` or `"http://localhost:11434/v1"` |
| `organization` | `str` | `None` | OpenAI organization ID (OpenAI only) |
| `project` | `str` | `None` | OpenAI project ID (OpenAI only) |
| `system_prompt` | `str` | `None` | System prompt sent as the `instructions` field |
| `prompt_template` | `str` | `None` | Template with `{variable}` placeholders |
| `temperature` | `float` | `None` | Sampling temperature 0 to 2. Higher = more random |
| `top_p` | `float` | `None` | Nucleus sampling 0 to 1 |
| `max_tokens` | `int` | `None` | Maximum output tokens (maps to `max_output_tokens`) |
| `reasoning_effort` | `str` | `None` | `"low"`, `"medium"`, or `"high"` — for o3, o3-mini, o1 only |
| `reasoning_summary` | `str` | `None` | Include a reasoning summary in output (OpenAI only) |
| `text_verbosity` | `str` | `None` | `"low"`, `"medium"`, or `"high"` |
| `output_schema` | `BaseModel` / `dict` | `None` | Pydantic model or JSON schema for structured output |
| `response_mime_type` | `str` | `None` | `"application/json"` enables JSON object mode |
| `structured_output` | `bool` | `False` | If `True`, `invoke()` returns a metadata dict |
| `streaming` | `bool` | `False` | If `True`, `invoke()` streams internally and joins |
| `max_retries` | `int` | `3` | Retry attempts on transient API errors |
| `timeout` | `float` | `60.0` | Request timeout in seconds |
| `backoff_factor` | `float` | `0.5` | Exponential back-off base (wait = factor x 2^attempt) |
| `input_pricing` | `float` | `None` | USD per 1 million input tokens |
| `output_pricing` | `float` | `None` | USD per 1 million output tokens |
| `circuit_failure_threshold` | `int` | `5` | Consecutive failures before the circuit opens |
| `circuit_cooldown_time` | `float` | `30.0` | Seconds the circuit stays open before probing |

---

## API Reference

### What Each Method Returns

| Method | Returns |
|---|---|
| `invoke(prompt)` | `str`, generated text (or `dict` if `structured_output=True`) |
| `ainvoke(prompt)` | same as `invoke`, async |
| `stream(prompt)` | `Iterator[str]`, text chunks |
| `astream(prompt)` | `AsyncIterator[str]`, text chunks |
| `batch_invoke(prompts)` | `list[str]`, one result per prompt, sequential |
| `abatch_invoke(prompts)` | `list[str]`, concurrent results |
| `chat(messages)` | `str`, generated text (or `dict` if `structured_output=True`) |
| `achat(messages)` | same as `chat`, async |
| `create(input_data)` | Raw Responses API `Response` object |
| `acreate(input_data)` | same as `create`, async |

### Metadata dict (when `structured_output=True`, or via `llm.last_metadata`)

| Key | Type | Description |
|---|---|---|
| `"model"` | `str` | Model name used |
| `"response"` | `str` | Generated text |
| `"input_tokens"` | `int \| None` | Input token count |
| `"output_tokens"` | `int \| None` | Output token count |
| `"total_tokens"` | `int \| None` | Total token count |
| `"input_cost"` | `float` | Input cost in USD (only if `input_pricing` set) |
| `"output_cost"` | `float` | Output cost in USD (only if `output_pricing` set) |
| `"total_cost"` | `float` | Total cost in USD (only if both pricing set) |
| `"latency_ms"` | `float` | Request round-trip time in milliseconds |

---

## Differences vs autourgos-openaichat

| Feature | autourgos-openaichat | autourgos-responses |
|---|---|---|
| API endpoint | `chat.completions.create` | `responses.create` |
| System prompt field | `messages[0].role = "system"` | `instructions` parameter |
| Reasoning models | Not supported | `reasoning_effort` param for o3/o1 |
| Text verbosity control | Not supported | `text_verbosity` param |
| Multi-turn input | Messages list | Messages list (via `chat()`) or plain string |
| Native tool calling | Supported (`invoke_with_tools`) | Not yet in this wrapper |
| Use when | Building chat agents, tool-calling | Using reasoning models, simple generation |

Both packages support the same providers via `base_url`. Choose based on the API endpoint your use case needs.

---

## License

Apache License 2.0, Copyright (c) 2026 Jitin Kumar Sengar
