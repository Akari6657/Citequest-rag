"""
LLM provider abstraction with OpenAI-compatible API support.

Two implementations:
- OpenAICompatibleProvider — calls DeepSeek / OpenAI / any compatible API
- MockLLMProvider           — returns canned responses for testing (no API key)

Configuration via environment variables:
    LLM_BASE_URL   — API base URL (default: https://api.deepseek.com/v1)
    LLM_API_KEY    — API key
    LLM_MODEL      — Model name (default: deepseek-chat)

Usage:
    from app.rag.llm_provider import create_provider

    llm = create_provider()            # auto-detect from env
    response = llm.generate(prompt)
    print(response.text)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Load .env from project root (if it exists) so env vars are available
# before create_provider() reads them.
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Result from an LLM generate() call."""

    text: str
    """The generated text."""

    model: str = ""
    """Model name used."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage: {'prompt_tokens': N, 'completion_tokens': M, 'total_tokens': T}"""

    latency_ms: float = 0.0
    """Time spent waiting for the API response."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMProvider:
    """Abstract interface for LLM backends."""

    def generate(
        self,
        system: str = "",
        user: str = "",
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock provider (no API key needed, for tests)
# ---------------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """Returns pre-defined answers.  For unit tests — never calls an API."""

    def __init__(self, fixed_response: str = ""):
        self._fixed = fixed_response

    def generate(self, system: str = "", user: str = "", **kwargs) -> LLMResponse:
        answer = self._fixed or (
            "这是一条基于检索证据的模拟回答 [1]。实际运行时将由 LLM 生成回答。"
        )
        return LLMResponse(
            text=answer,
            model="mock",
            usage={},
            latency_ms=0.0,
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (DeepSeek, OpenAI, etc.)
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(LLMProvider):
    """Calls any OpenAI-compatible chat completions API.

    Works with DeepSeek, OpenAI, vLLM, Ollama, and any endpoint that
    implements POST /chat/completions.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        short_task: bool = False,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.short_task = short_task

    def generate(
        self,
        system: str = "",
        user: str = "",
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request and return the response.

        Args:
            system: System prompt (instructions for the model).
            user: User message (evidence + question).
            **kwargs: Passed as extra fields to the API (temperature, max_tokens, etc.).

        Returns:
            LLMResponse with text, model, usage, and latency.
        """
        import httpx

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        # DeepSeek defaults to thinking mode. Keyword extraction needs the
        # small output budget for its answer, not a reasoning trace. Keep
        # provider-specific fields away from other compatible endpoints.
        if self.short_task and urlsplit(self.base_url).hostname == "api.deepseek.com":
            body.setdefault("thinking", {"type": "disabled"})

        logger.info("Calling %s model=%s ...", url, self.model)
        t0 = time.perf_counter()

        try:
            response = httpx.post(
                url,
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("LLM API call failed: %s", exc)
            raise

        elapsed = (time.perf_counter() - t0) * 1000

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content", "")

        usage_raw = data.get("usage", {}) or {}

        return LLMResponse(
            text=content.strip() if content else "",
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage_raw.get("prompt_tokens", 0),
                "completion_tokens": usage_raw.get("completion_tokens", 0),
                "total_tokens": usage_raw.get("total_tokens", 0),
            },
            latency_ms=round(elapsed, 2),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def is_llm_configured() -> bool:
    """Return whether a real OpenAI-compatible provider is configured."""
    return bool(os.getenv("LLM_API_KEY", "").strip())


def create_provider(*, timeout: float | None = None, short_task: bool = False) -> LLMProvider:
    """Create the appropriate LLM provider based on environment.

    If LLM_API_KEY is set, returns an OpenAICompatibleProvider (DeepSeek).
    Otherwise returns a MockLLMProvider for development.

    Args:
        timeout: Optional request timeout override. When omitted, the provider's
                 normal timeout is preserved.
        short_task: Disable thinking for short tasks on the official DeepSeek
                    endpoint. Other providers retain their normal behavior.
    """
    if is_llm_configured():
        logger.info("Using OpenAICompatibleProvider (model=%s)", os.getenv("LLM_MODEL", "deepseek-chat"))
        if timeout is None:
            return OpenAICompatibleProvider(short_task=short_task)
        return OpenAICompatibleProvider(timeout=timeout, short_task=short_task)
    else:
        logger.info("No LLM_API_KEY set — using MockLLMProvider")
        return MockLLMProvider()
