"""Tests for app.api.routes_search readiness behavior."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api import routes_search
from app.core.schemas import SearchRequest, SearchResult


def _configure_hybrid_route(monkeypatch, tmp_path, observed):
    (tmp_path / "index.faiss").write_text("fake", encoding="utf-8")
    (tmp_path / "id_map.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(routes_search, "get_db_path", lambda: tmp_path / "metadata.sqlite")
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(
        routes_search,
        "route_query",
        lambda query: SimpleNamespace(should_rag=False, reason="test"),
    )

    def fake_search_hybrid(*args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(routes_search, "search_hybrid", fake_search_hybrid)


def test_faiss_ready_requires_index_and_id_map(tmp_path):
    assert routes_search._faiss_ready(tmp_path) is False

    (tmp_path / "index.faiss").write_text("fake", encoding="utf-8")
    assert routes_search._faiss_ready(tmp_path) is False

    (tmp_path / "id_map.json").write_text("[]", encoding="utf-8")
    assert routes_search._faiss_ready(tmp_path) is True


def test_vector_search_requires_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)

    request = SearchRequest(query="GAN image generation", mode="vector", top_k=3)

    try:
        routes_search.search_papers(request)
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail["error_code"] == "INDEX_NOT_READY"
    else:
        raise AssertionError("Expected vector search to fail when FAISS is missing")


def test_hybrid_search_requires_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)

    request = SearchRequest(query="GAN image generation", mode="hybrid", top_k=3)

    try:
        routes_search.search_papers(request)
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail["error_code"] == "INDEX_NOT_READY"
    else:
        raise AssertionError("Expected hybrid search to fail when FAISS is missing")


def test_lexical_search_does_not_require_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(routes_search, "search_lexical", lambda *args, **kwargs: [])

    request = SearchRequest(query="GAN image generation", mode="lexical", top_k=3)
    response = routes_search.search_papers(request)

    assert response.mode == "lexical"
    assert response.effective_alpha is None
    assert response.total_results == 0
    assert response.results == []


def test_search_request_rejects_reversed_year_range():
    with pytest.raises(ValidationError, match="year_from must be less than or equal"):
        SearchRequest(
            query="retrieval",
            mode="lexical",
            year_from=2024,
            year_to=2020,
        )


@pytest.mark.parametrize(
    ("mode", "retriever_name"),
    [
        ("lexical", "search_lexical"),
        ("vector", "search_vector"),
        ("hybrid", "search_hybrid"),
    ],
)
def test_search_route_forwards_year_range(
    monkeypatch, tmp_path, mode, retriever_name
):
    (tmp_path / "index.faiss").write_text("fake", encoding="utf-8")
    (tmp_path / "id_map.json").write_text("[]", encoding="utf-8")
    observed = {}

    monkeypatch.setattr(routes_search, "get_db_path", lambda: tmp_path / "metadata.sqlite")
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(
        routes_search,
        "prepare_lexical_query",
        lambda query: (query, ""),
    )
    monkeypatch.setattr(
        routes_search,
        "route_query",
        lambda query: SimpleNamespace(should_rag=False, reason="test"),
    )

    def fake_retriever(*args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(routes_search, retriever_name, fake_retriever)

    response = routes_search.search_papers(
        SearchRequest(
            query="filtered retrieval",
            mode=mode,
            top_k=3,
            year_from=2020,
            year_to=2024,
        )
    )

    assert observed["year_from"] == 2020
    assert observed["year_to"] == 2024
    assert response.total_results == 0


def test_hybrid_search_uses_environment_alpha_when_omitted(monkeypatch, tmp_path):
    observed = {}
    _configure_hybrid_route(monkeypatch, tmp_path, observed)
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "0.65")

    request = SearchRequest(query="hybrid retrieval", mode="hybrid", top_k=3)
    response = routes_search.search_papers(request)

    assert request.alpha is None
    assert observed["alpha"] == 0.65
    assert response.effective_alpha == 0.65


def test_explicit_search_alpha_overrides_environment(monkeypatch, tmp_path):
    observed = {}
    _configure_hybrid_route(monkeypatch, tmp_path, observed)
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "0.8")

    request = SearchRequest(
        query="hybrid retrieval", mode="hybrid", top_k=3, alpha=0.2
    )
    response = routes_search.search_papers(request)

    assert observed["alpha"] == 0.2
    assert response.effective_alpha == 0.2


def test_invalid_environment_alpha_returns_configuration_error(monkeypatch, tmp_path):
    observed = {}
    _configure_hybrid_route(monkeypatch, tmp_path, observed)
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "invalid")

    with pytest.raises(HTTPException) as exc_info:
        routes_search.search_papers(
            SearchRequest(query="hybrid retrieval", mode="hybrid", top_k=3)
        )

    assert exc_info.value.status_code == 500
    assert (
        exc_info.value.detail["error_code"]
        == "INVALID_HYBRID_ALPHA_CONFIGURATION"
    )
    assert observed == {}


def test_paper_deduplication_preserves_retrieval_rank(monkeypatch, tmp_path):
    ranked_results = [
        SearchResult(
            paper_id="P1",
            chunk_id="P1_best",
            title="Paper one",
            year=2024,
            venue=None,
            authors=[],
            score=0.1,
            abstract="Best-ranked chunk",
        ),
        SearchResult(
            paper_id="P1",
            chunk_id="P1_later",
            title="Paper one",
            year=2024,
            venue=None,
            authors=[],
            score=0.9,
            abstract="Later chunk with an incomparable score",
        ),
        SearchResult(
            paper_id="P2",
            chunk_id="P2_only",
            title="Paper two",
            year=2023,
            venue=None,
            authors=[],
            score=0.05,
            abstract="Second paper",
        ),
    ]

    monkeypatch.setattr(routes_search, "get_db_path", lambda: tmp_path / "metadata.sqlite")
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(
        routes_search, "search_lexical", lambda *args, **kwargs: ranked_results
    )
    monkeypatch.setattr(
        routes_search,
        "route_query",
        lambda query: SimpleNamespace(should_rag=False, reason="test"),
    )

    response = routes_search.search_papers(
        SearchRequest(query="ranked chunks", mode="lexical", top_k=3)
    )

    assert response.total_results == 2
    assert [result.chunk_id for result in response.results] == ["P1_best", "P2_only"]


def test_chinese_hybrid_uses_expanded_lexical_query(monkeypatch, tmp_path):
    observed = {}
    _configure_hybrid_route(monkeypatch, tmp_path, observed)
    query = "为什么 GAN 训练不稳定"
    expanded = f"{query} GAN mode collapse convergence"
    monkeypatch.setattr(
        routes_search,
        "prepare_lexical_query",
        lambda value: (expanded, "GAN mode collapse convergence"),
    )

    response = routes_search.search_papers(
        SearchRequest(query=query, mode="hybrid", top_k=3)
    )

    assert observed["query"] == query
    assert observed["lexical_query"] == expanded
    assert response.rewrite_keywords == "GAN mode collapse convergence"


def test_vector_mode_never_calls_rewrite(monkeypatch, tmp_path):
    (tmp_path / "index.faiss").write_text("fake", encoding="utf-8")
    (tmp_path / "id_map.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(routes_search, "get_db_path", lambda: tmp_path / "metadata.sqlite")
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(routes_search, "search_vector", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        routes_search,
        "prepare_lexical_query",
        lambda query: pytest.fail("Vector mode must not call rewrite"),
    )
    monkeypatch.setattr(
        routes_search,
        "route_query",
        lambda query: SimpleNamespace(should_rag=False, reason="test"),
    )

    response = routes_search.search_papers(
        SearchRequest(query="为什么 GAN 训练不稳定", mode="vector", top_k=3)
    )

    assert response.total_results == 0
    assert response.rewrite_keywords == ""


def test_zero_alpha_hybrid_skips_rewrite(monkeypatch, tmp_path):
    observed = {}
    _configure_hybrid_route(monkeypatch, tmp_path, observed)
    monkeypatch.setattr(
        routes_search,
        "prepare_lexical_query",
        lambda query: pytest.fail("Pure-vector Hybrid must not call rewrite"),
    )

    response = routes_search.search_papers(
        SearchRequest(
            query="为什么 GAN 训练不稳定",
            mode="hybrid",
            alpha=0.0,
            top_k=3,
        )
    )

    assert observed["lexical_query"] == "为什么 GAN 训练不稳定"
    assert response.rewrite_keywords == ""


def test_search_overview_receives_final_ranked_candidates(monkeypatch, tmp_path):
    ranked = [
        SearchResult(
            paper_id="P1",
            chunk_id="expanded_hit",
            title="Expanded hit",
            year=2024,
            venue=None,
            authors=[],
            score=1.0,
            abstract="Evidence",
        )
    ]
    ranked.extend(ranked[0].model_copy(update={
        "paper_id": f"P{number // 2 + 1}", "chunk_id": f"chunk{number}",
    }) for number in range(1, 6))
    observed = {}
    monkeypatch.setattr(routes_search, "get_db_path", lambda: tmp_path / "metadata.sqlite")
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(
        routes_search,
        "prepare_lexical_query",
        lambda query: (f"{query} retrieval evaluation", "retrieval evaluation"),
    )
    monkeypatch.setattr(routes_search, "search_lexical", lambda *args, **kwargs: ranked)
    monkeypatch.setattr(
        routes_search,
        "route_query",
        lambda query: SimpleNamespace(should_rag=True, reason="test"),
    )

    def fake_answer_question(**kwargs):
        observed.update(kwargs)
        return None

    monkeypatch.setattr(routes_search, "answer_question", fake_answer_question)

    response = routes_search.search_papers(
        SearchRequest(
            query="如何评估 RAG",
            mode="lexical",
            top_k=10,
            include_overview=True,
        )
    )

    assert observed["pre_retrieved"][0].chunk_id == "expanded_hit"
    assert observed["top_k"] == 5
    assert observed["pre_retrieved"] == ranked[:5]
    assert len(response.results) == 3
    assert response.results[0].chunk_id == "expanded_hit"
