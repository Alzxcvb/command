"""API wrappers for model providers.

Four backends: Anthropic direct, OpenAI direct, local Ollama, OpenRouter (fallback).
Use `get_provider_for(model)` to pick the right one for a given `ModelInfo`; set
`COMMAND_PROVIDER` in the env to force a single backend ("anthropic", "openai",
"ollama", "openrouter"). If a direct backend's key is missing the dispatcher
falls back to OpenRouter so one env var is enough to start.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import httpx
from openai import OpenAI

from .types import ModelInfo


class Provider(Protocol):
    def call(
        self,
        model: ModelInfo,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> tuple[str, float]: ...

    def call_raw(
        self,
        model_id: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> tuple[str, float]: ...


class OpenRouterProvider:
    """Sends prompts via OpenRouter — universal fallback."""

    name = "openrouter"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY env var or pass api_key."
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def call(self, model, prompt, *, system_prompt=None, max_tokens=1024):
        return self.call_raw(model.id, prompt, system_prompt=system_prompt, max_tokens=max_tokens)

    def call_raw(self, model_id, prompt, *, system_prompt=None, max_tokens=1024):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""
        return content, latency_ms


class AnthropicProvider:
    """Direct Anthropic API. Strips the 'anthropic/' prefix for model IDs."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("No Anthropic API key. Set ANTHROPIC_API_KEY.")
        self.base_url = "https://api.anthropic.com/v1/messages"

    @staticmethod
    def _strip(model_id: str) -> str:
        return model_id.split("/", 1)[1] if model_id.startswith("anthropic/") else model_id

    def call(self, model, prompt, *, system_prompt=None, max_tokens=1024):
        return self.call_raw(model.id, prompt, system_prompt=system_prompt, max_tokens=max_tokens)

    def call_raw(self, model_id, prompt, *, system_prompt=None, max_tokens=1024):
        body = {
            "model": self._strip(model_id),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        start = time.perf_counter()
        resp = httpx.post(self.base_url, json=body, headers=headers, timeout=60.0)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
        blocks = data.get("content", [])
        content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return content, latency_ms


class OpenAIProvider:
    """Direct OpenAI API. Strips the 'openai/' prefix for model IDs."""

    name = "openai"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("No OpenAI API key. Set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=self.api_key)

    @staticmethod
    def _strip(model_id: str) -> str:
        return model_id.split("/", 1)[1] if model_id.startswith("openai/") else model_id

    def call(self, model, prompt, *, system_prompt=None, max_tokens=1024):
        return self.call_raw(model.id, prompt, system_prompt=system_prompt, max_tokens=max_tokens)

    def call_raw(self, model_id, prompt, *, system_prompt=None, max_tokens=1024):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self._strip(model_id), messages=messages, max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""
        return content, latency_ms


class OllamaProvider:
    """Local Ollama HTTP API. Defaults to http://localhost:11434.

    Model id is passed through as-is after stripping a 'local/' or 'ollama/'
    prefix. Ollama is $0 — always preferred when it can handle the task.
    """

    name = "ollama"

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL")
                         or "http://localhost:11434").rstrip("/")

    @staticmethod
    def _strip(model_id: str) -> str:
        for prefix in ("local/", "ollama/"):
            if model_id.startswith(prefix):
                return model_id[len(prefix):]
        return model_id

    def call(self, model, prompt, *, system_prompt=None, max_tokens=1024):
        return self.call_raw(model.id, prompt, system_prompt=system_prompt, max_tokens=max_tokens)

    def call_raw(self, model_id, prompt, *, system_prompt=None, max_tokens=1024):
        body = {
            "model": self._strip(model_id),
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if system_prompt:
            body["system"] = system_prompt
        start = time.perf_counter()
        resp = httpx.post(f"{self.base_url}/api/generate", json=body, timeout=120.0)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", ""), latency_ms


def _build_if_keyed(cls):
    try:
        return cls()
    except ValueError:
        return None


def get_provider_for(model: ModelInfo) -> Provider:
    """Pick the provider for a model.

    Priority:
      1. `COMMAND_PROVIDER` env override ("anthropic" | "openai" | "ollama" | "openrouter")
      2. Provider matching `model.provider` (Anthropic, OpenAI, local) if its key is set
      3. OpenRouter fallback
    """
    forced = os.environ.get("COMMAND_PROVIDER", "").strip().lower()
    if forced:
        if forced == "anthropic":
            return AnthropicProvider()
        if forced == "openai":
            return OpenAIProvider()
        if forced == "ollama":
            return OllamaProvider()
        if forced == "openrouter":
            return OpenRouterProvider()
        raise ValueError(f"Unknown COMMAND_PROVIDER: {forced!r}")

    provider_name = (model.provider or "").lower()
    if provider_name == "anthropic":
        direct = _build_if_keyed(AnthropicProvider)
        if direct:
            return direct
    if provider_name == "openai":
        direct = _build_if_keyed(OpenAIProvider)
        if direct:
            return direct
    if provider_name in ("local", "ollama"):
        return OllamaProvider()

    return OpenRouterProvider()
