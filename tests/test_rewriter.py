"""Tests for bounded, optional Chinese query rewriting."""

from __future__ import annotations

import pytest

from app.rag import rewriter
from app.rag.llm_provider import LLMResponse


class FakeProvider:
    def __init__(self, text: str = "", error: Exception | None = None):
        self.text = text
        self.error = error
        self.observed = {}

    def generate(self, **kwargs):
        self.observed.update(kwargs)
        if self.error is not None:
            raise self.error
        return LLMResponse(text=self.text, model="fake")


def test_english_query_skips_provider(monkeypatch):
    monkeypatch.setattr(
        rewriter,
        "create_provider",
        lambda **kwargs: pytest.fail("English rewrite should not create a provider"),
    )

    assert rewriter.prepare_lexical_query("hybrid retrieval") == (
        "hybrid retrieval",
        "",
    )


def test_missing_api_key_skips_mock_provider(monkeypatch):
    monkeypatch.setattr(rewriter, "is_llm_configured", lambda: False)
    monkeypatch.setattr(
        rewriter,
        "create_provider",
        lambda **kwargs: pytest.fail("Mock provider must not supply rewrite keywords"),
    )

    query = "为什么 GAN 训练不稳定"
    assert rewriter.prepare_lexical_query(query) == (query, "")


def test_successful_rewrite_expands_original_query(monkeypatch):
    provider = FakeProvider("Keywords: GAN mode-collapse convergence")
    observed = {}
    monkeypatch.setattr(rewriter, "is_llm_configured", lambda: True)
    monkeypatch.setattr(rewriter, "get_rewrite_timeout_seconds", lambda: 0.75)

    def fake_create_provider(*, timeout, short_task):
        observed["timeout"] = timeout
        observed["short_task"] = short_task
        return provider

    monkeypatch.setattr(rewriter, "create_provider", fake_create_provider)

    query = "为什么 GAN 训练不稳定"
    lexical_query, keywords = rewriter.prepare_lexical_query(query)

    assert observed["timeout"] == 0.75
    assert observed["short_task"] is True
    assert keywords == "GAN mode-collapse convergence"
    assert lexical_query == f"{query} {keywords}"
    assert provider.observed["temperature"] == 0.0
    assert provider.observed["max_tokens"] == 128
    assert query in provider.observed["user"]


def test_provider_timeout_preserves_original_query(monkeypatch):
    monkeypatch.setattr(rewriter, "is_llm_configured", lambda: True)
    monkeypatch.setattr(rewriter, "get_rewrite_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr(
        rewriter,
        "create_provider",
        lambda **kwargs: FakeProvider(error=TimeoutError("slow provider")),
    )

    query = "如何评估检索增强生成"
    assert rewriter.prepare_lexical_query(query) == (query, "")


@pytest.mark.parametrize(
    "output",
    [
        "",
        "这是一条基于检索证据的模拟回答 [1]。",
        "123 456",
        "a" * 201,
        "one two three four five six seven eight nine ten eleven twelve thirteen",
    ],
)
def test_invalid_provider_output_preserves_original_query(monkeypatch, output):
    monkeypatch.setattr(rewriter, "is_llm_configured", lambda: True)
    monkeypatch.setattr(rewriter, "get_rewrite_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(
        rewriter,
        "create_provider",
        lambda **kwargs: FakeProvider(output),
    )

    query = "如何评估检索增强生成"
    assert rewriter.prepare_lexical_query(query) == (query, "")


def test_invalid_timeout_configuration_preserves_original_query(monkeypatch):
    monkeypatch.setattr(rewriter, "is_llm_configured", lambda: True)

    def invalid_timeout():
        raise ValueError("invalid timeout")

    monkeypatch.setattr(rewriter, "get_rewrite_timeout_seconds", invalid_timeout)

    query = "如何评估检索增强生成"
    assert rewriter.prepare_lexical_query(query) == (query, "")
