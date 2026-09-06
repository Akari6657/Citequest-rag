"""Tests for RAG modules — all use MockLLM, no real API calls."""

import pytest

from app.rag.citation import extract_citations, verify_citations
from app.rag.context_builder import _estimate_tokens
from app.rag.llm_provider import (
    MockLLMProvider,
    OpenAICompatibleProvider,
    create_provider,
)
from app.rag.prompt import build_prompts
from app.rag import answer as rag_answer
from app.rag.answer import (
    _dicts_to_search_results,
    _effective_hybrid_alpha,
    _retrieve_evidence,
)
from app.core.schemas import SearchResult


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_basic(self):
        assert extract_citations("结论 [1] 和 [2] 表明了这一点。") == [1, 2]

    def test_deduplicate(self):
        assert extract_citations("[1] [1] [2] [1]") == [1, 2]

    def test_no_citations(self):
        assert extract_citations("没有引用。") == []

    def test_large_numbers(self):
        assert extract_citations("参见 [1] 和 [123]") == [1, 123]

    def test_not_citation_brackets(self):
        # Python list syntax should not be caught
        assert extract_citations("list = [1, 2, 3]") == []


class TestVerifyCitations:
    def _map(self, ids):
        return [{"citation_id": i, "title": f"Paper {i}", "paper_id": f"P{i}"} for i in ids]

    def test_all_valid(self):
        r = verify_citations("如 [1][2] 所示。", self._map([1, 2, 3]))
        assert r.valid is True

    def test_invalid_id(self):
        r = verify_citations("如 [1] 和 [5] 所示。", self._map([1, 2]))
        assert r.valid is False
        assert 5 in r.invalid_ids

    def test_no_citations(self):
        r = verify_citations("没有引用。", self._map([1, 2]))
        assert r.valid is False
        assert len(r.warnings) > 0


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class TestBuildPrompts:
    def test_returns_tuple(self):
        system, user = build_prompts("[1] Test evidence", "测试问题")
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_chinese_content(self):
        system, user = build_prompts("[1] 证据", "问题？")
        assert "证据" in user
        assert "问题？" in user

    def test_evidence_in_user(self):
        _, user = build_prompts("[1] GAN paper abstract here", "什么是GAN？")
        assert "[1] GAN paper abstract here" in user
        assert "什么是GAN？" in user


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 0

    def test_short(self):
        assert _estimate_tokens("hello world") == 3

    def test_non_ascii_uses_a_conservative_allowance(self):
        assert _estimate_tokens("abcd中") == 3  # 1 ASCII token + 2 for the CJK character.


class TestRAGHybridAlpha:
    def test_direct_rag_default_is_canonical(self):
        assert _effective_hybrid_alpha("hybrid", None) == 0.5

    def test_non_hybrid_mode_does_not_report_alpha(self):
        assert _effective_hybrid_alpha("lexical", 0.2) is None

    def test_invalid_direct_alpha_is_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            _effective_hybrid_alpha("hybrid", 1.1)


class TestRAGRetrievalRewrite:
    def test_hybrid_uses_original_dense_and_expanded_lexical_query(
        self, monkeypatch, tmp_path
    ):
        observed = {}
        query = "如何评估 RAG"
        expanded = f"{query} retrieval augmented generation evaluation"
        monkeypatch.setattr(
            rag_answer,
            "prepare_lexical_query",
            lambda value: (expanded, "retrieval augmented generation evaluation"),
        )

        def fake_hybrid(*args, **kwargs):
            observed["query"] = args[0]
            observed.update(kwargs)
            return []

        monkeypatch.setattr(rag_answer, "search_hybrid", fake_hybrid)

        _retrieve_evidence(query, 5, "hybrid", 0.5, tmp_path / "db", tmp_path)

        assert observed["query"] == query
        assert observed["lexical_query"] == expanded

    def test_vector_mode_skips_rewrite(self, monkeypatch, tmp_path):
        observed = {}
        monkeypatch.setattr(
            rag_answer,
            "prepare_lexical_query",
            lambda query: pytest.fail("Vector RAG must not call rewrite"),
        )

        def fake_vector(query, **kwargs):
            observed["query"] = query
            return []

        monkeypatch.setattr("app.retrieval.vector_store.search_vector", fake_vector)

        _retrieve_evidence("中文问题", 5, "vector", None, tmp_path / "db", tmp_path)

        assert observed["query"] == "中文问题"

    def test_zero_alpha_hybrid_skips_rewrite(self, monkeypatch, tmp_path):
        observed = {}
        monkeypatch.setattr(
            rag_answer,
            "prepare_lexical_query",
            lambda query: pytest.fail("Pure-vector Hybrid must not call rewrite"),
        )

        def fake_hybrid(*args, **kwargs):
            observed.update(kwargs)
            return []

        monkeypatch.setattr(rag_answer, "search_hybrid", fake_hybrid)

        _retrieve_evidence("中文问题", 5, "hybrid", 0.0, tmp_path / "db", tmp_path)

        assert observed["lexical_query"] == "中文问题"

    def test_in_process_pre_retrieved_results_are_preserved(self):
        result = SearchResult(
            paper_id="P1",
            chunk_id="expanded_hit",
            title="Expanded hit",
            year=2024,
            venue=None,
            score=1.0,
        )

        normalized = _dicts_to_search_results([result])

        assert normalized == [result]


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


class TestMockLLM:
    def test_returns_fixed_response(self):
        llm = MockLLMProvider("固定回答 [1]。")
        r = llm.generate(system="sys", user="question")
        assert r.text == "固定回答 [1]。"
        assert r.model == "mock"
        assert r.latency_ms == 0.0

    def test_default_response(self):
        llm = MockLLMProvider()
        r = llm.generate(system="sys", user="question")
        assert len(r.text) > 0
        assert "[1]" in r.text


class TestCreateProvider:
    def test_mock_when_no_key(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        provider = create_provider()
        assert isinstance(provider, MockLLMProvider)

    def test_real_provider_accepts_timeout_override(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        provider = create_provider(timeout=0.75)

        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.timeout == 0.75
