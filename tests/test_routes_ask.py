"""Tests for Hybrid alpha resolution in Ask API routes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.api import routes_ask
from app.core.schemas import AskRequest, AskResponse


def _ask_response(question: str, effective_alpha: float | None) -> AskResponse:
    return AskResponse(
        question=question,
        answer="Grounded answer [1].",
        effective_alpha=effective_alpha,
        citations=[],
        citation_valid=True,
        citation_warnings=[],
    )


def test_ask_uses_environment_alpha_when_omitted(monkeypatch):
    observed = {}
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "0.65")

    def fake_answer_question(**kwargs):
        observed.update(kwargs)
        return _ask_response(kwargs["question"], kwargs["alpha"])

    monkeypatch.setattr(routes_ask, "answer_question", fake_answer_question)

    request = AskRequest(question="How does hybrid retrieval work?")
    response = routes_ask.ask_question(request)

    assert request.alpha is None
    assert request.top_k == observed["top_k"] == 5
    assert observed["alpha"] == 0.65
    assert response.effective_alpha == 0.65


def test_explicit_ask_alpha_overrides_environment(monkeypatch):
    observed = {}
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "0.8")

    def fake_answer_question(**kwargs):
        observed.update(kwargs)
        return _ask_response(kwargs["question"], kwargs["alpha"])

    monkeypatch.setattr(routes_ask, "answer_question", fake_answer_question)

    response = routes_ask.ask_question(
        AskRequest(question="How does hybrid retrieval work?", alpha=0.2)
    )

    assert observed["alpha"] == 0.2
    assert response.effective_alpha == 0.2


def test_non_hybrid_ask_does_not_read_invalid_environment(monkeypatch):
    observed = {}
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "invalid")

    def fake_answer_question(**kwargs):
        observed.update(kwargs)
        return _ask_response(kwargs["question"], kwargs["alpha"])

    monkeypatch.setattr(routes_ask, "answer_question", fake_answer_question)

    response = routes_ask.ask_question(
        AskRequest(
            question="Find exact terms",
            retrieval_mode="lexical",
        )
    )

    assert observed["alpha"] is None
    assert response.effective_alpha is None


def test_streaming_ask_uses_resolved_alpha(monkeypatch):
    observed = {}
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "0.65")

    from app.rag import router as rag_router

    monkeypatch.setattr(
        rag_router,
        "route_query",
        lambda question: SimpleNamespace(should_rag=True, reason="test"),
    )

    async def events():
        yield "event: done\ndata: \n\n"

    def fake_answer_question_stream(**kwargs):
        observed.update(kwargs)
        return events()

    monkeypatch.setattr(
        routes_ask, "answer_question_stream", fake_answer_question_stream
    )

    response = asyncio.run(
        routes_ask.ask_question_stream(
            AskRequest(question="How does hybrid retrieval work?")
        )
    )

    assert response.media_type == "text/event-stream"
    assert observed["alpha"] == 0.65
    assert observed["top_k"] == 5
