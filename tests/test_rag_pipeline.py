"""Behavior shared by ordinary and SSE RAG answers, using local evidence and MockLLM."""

import asyncio
import json
import sqlite3
import threading
from unittest.mock import Mock

import pytest

from app.core.schemas import AskResponse, SearchResult
from app.rag import answer as rag_answer
from app.rag.llm_provider import LLMResponse, MockLLMProvider
from scripts.build_fts import build_fts
from scripts.build_metadata_db import CREATE_TABLES


@pytest.fixture
def evidence_db(tmp_path):
    """Three papers; the first has two chunks that must remain separate evidence."""
    db_path = tmp_path / "metadata.sqlite"
    results = []
    with sqlite3.connect(db_path) as conn:
        conn.executescript(CREATE_TABLES)
        for number in range(1, 4):
            conn.execute(
                "INSERT INTO papers(paper_id, title, year, url) VALUES (?, ?, ?, ?)",
                (f"P{number}", f"Paper {number}", 2024, f"https://example.test/paper/{number}"),
            )
        for number, paper_id in enumerate(["P1", "P1", "P2", "P3"], start=1):
            chunk_id = f"C{number}"
            title = f"Paper {paper_id[-1]}"
            conn.execute(
                "INSERT INTO chunks(chunk_id, paper_id, chunk_text) VALUES (?, ?, ?)",
                (chunk_id, paper_id, f"Title: {title}\nAbstract: Complete evidence {number}."),
            )
            results.append(SearchResult(
                paper_id=paper_id,
                chunk_id=chunk_id,
                title=title,
                year=2024,
                venue=None,
                score=1 / number,
                snippet=f"DISPLAY_PREVIEW_{number}",
            ))
    build_fts(db_path)
    return db_path, results


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    monkeypatch.setattr(
        rag_answer, "create_provider",
        lambda: pytest.fail("This path must not create an LLM provider"),
    )


def install_mock_llm(monkeypatch, answer):
    provider = MockLLMProvider(answer)
    generate = Mock(wraps=provider.generate)
    if answer == "":
        # MockLLMProvider normally substitutes its default canned answer.
        generate.return_value = LLMResponse(text="", model="mock")
    monkeypatch.setattr(provider, "generate", generate)
    monkeypatch.setattr(rag_answer, "create_provider", lambda: provider)
    return generate


def decode_event(raw):
    event_line, data_line, end = raw.split("\n", 2)
    assert event_line.startswith("event: ")
    assert data_line.startswith("data: ")
    assert end == "\n"
    payload = data_line.removeprefix("data: ")
    return event_line.removeprefix("event: "), json.loads(payload) if payload else ""


async def collect_stream(**kwargs):
    return [decode_event(event) async for event in rag_answer.answer_question_stream(**kwargs)]


def assert_same_response(kwargs, phases):
    response = rag_answer.answer_question(**kwargs).model_dump(mode="json")
    events = asyncio.run(collect_stream(**kwargs))
    assert [name for name, _ in events] == ["status"] * len(phases) + ["result", "done"]
    assert [data["phase"] for name, data in events if name == "status"] == phases
    assert all(data["message"] for name, data in events if name == "status")
    assert events[-1] == ("done", "")
    streamed = events[-2][1]
    assert AskResponse.model_validate(streamed).model_dump(mode="json") == streamed
    assert response.pop("latency_ms") >= 0
    assert streamed.pop("latency_ms") >= 0
    assert streamed == response
    return response


@pytest.mark.parametrize(
    "answer,valid,warnings",
    [
        ("Grounded answer [1][2].", True, []),
        ("Unsupported citation [1][99].", False, [
            "回答中引用了不存在的证据编号: [99]。有效编号范围: 1-2。",
        ]),
        ("Answer without citations.", False, [
            "回答中没有使用任何引用标记 [N]。可能包含未经证实的论断。",
            "大部分证据未被引用（2/2 个未使用）。",
        ]),
        ("", False, [
            "回答中没有使用任何引用标记 [N]。可能包含未经证实的论断。",
            "大部分证据未被引用（2/2 个未使用）。",
        ]),
    ],
    ids=["grounded", "invalid-id", "no-citations", "empty-answer"],
)
def test_pre_retrieved_answers_keep_evidence_and_citation_warnings(
    monkeypatch, evidence_db, answer, valid, warnings,
):
    db_path, results = evidence_db
    generate = install_mock_llm(monkeypatch, answer)
    monkeypatch.setattr(
        rag_answer, "_retrieve_evidence",
        lambda *args, **kwargs: pytest.fail("Pre-retrieved evidence must not be retrieved again"),
    )
    # Invalid entries are ignored before top_k is applied; both transport
    # dictionaries and SearchResult objects are used by existing callers.
    raw = [None, {"paper_id": "broken"}, results[0], *[r.model_dump() for r in results[1:]]]
    response = assert_same_response(
        dict(question="How does grounding work?", pre_retrieved=raw, top_k=2,
             alpha=0.65, db_path=db_path),
        ["organizing", "generating", "verifying"],
    )
    assert response == {
        "question": "How does grounding work?",
        "answer": answer,
        "effective_alpha": 0.65,
        "citations": [
            {"citation_id": i, "paper_id": "P1", "chunk_id": f"C{i}",
             "title": "Paper 1", "url": "https://example.test/paper/1"}
            for i in (1, 2)
        ],
        "citation_valid": valid,
        "citation_warnings": warnings,
    }
    assert len(raw) == 6
    assert len(results) == 4
    assert generate.call_count == 2
    assert generate.call_args_list[0] == generate.call_args_list[1]
    prompt = generate.call_args.kwargs["user"]
    assert "Complete evidence 1." in prompt and "Complete evidence 2." in prompt
    assert "Complete evidence 3." not in prompt
    assert "DISPLAY_PREVIEW" not in prompt


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_standalone_answers_retrieve_in_the_requested_mode(monkeypatch, evidence_db, mode):
    db_path, results = evidence_db
    install_mock_llm(monkeypatch, "Retrieved answer [1][2].")
    # Exercise the actual SQLite retriever; dense retrievers use the same tiny
    # evidence set without downloading models or building a FAISS index.
    lexical = Mock(wraps=rag_answer.search_lexical)
    vector = Mock(return_value=results[:2])
    hybrid = Mock(return_value=results[:2])
    monkeypatch.setattr(rag_answer, "search_lexical", lexical)
    monkeypatch.setattr("app.retrieval.vector_store.search_vector", vector)
    monkeypatch.setattr(rag_answer, "search_hybrid", hybrid)
    index_dir = db_path.parent / "faiss"
    response = assert_same_response(
        dict(question="evidence", top_k=2, retrieval_mode=mode, alpha=0.65,
             db_path=db_path, index_dir=index_dir),
        ["retrieving", "organizing", "generating", "verifying"],
    )
    assert len(response["citations"]) == 2
    assert response["citation_valid"] is True
    assert response["effective_alpha"] == (0.65 if mode == "hybrid" else None)
    for name, retriever in [("lexical", lexical), ("vector", vector), ("hybrid", hybrid)]:
        assert retriever.call_count == (2 if name == mode else 0)
    selected = {"lexical": lexical, "vector": vector, "hybrid": hybrid}[mode]
    assert selected.call_args.kwargs["db_path"] == db_path
    assert selected.call_args.kwargs["top_k"] == 2
    if mode != "lexical":
        assert selected.call_args.kwargs["index_dir"] == index_dir
    if mode == "hybrid":
        assert selected.call_args.kwargs["alpha"] == 0.65


@pytest.mark.parametrize(
    "pre_retrieved", [None, [], [{
        "paper_id": "P2", "chunk_id": "C1", "title": "Wrong owner",
        "year": None, "venue": None, "score": 1.0, "snippet": "Untrusted evidence",
    }]], ids=["no-hits", "explicit-empty", "mismatched-evidence"],
)
def test_no_evidence_skips_generation_and_verification(monkeypatch, evidence_db, pre_retrieved):
    db_path, _ = evidence_db
    monkeypatch.setattr(
        rag_answer, "verify_citations",
        lambda *args: pytest.fail("No-evidence refusal must not be verified as an uncited answer"),
    )
    phases = ["organizing"]
    if pre_retrieved is None:
        phases.insert(0, "retrieving")
    else:
        monkeypatch.setattr(
            rag_answer, "_retrieve_evidence",
            lambda *args: pytest.fail("An explicit empty result list must skip retrieval"),
        )
    response = assert_same_response(
        dict(question="unfindabletoken", retrieval_mode="lexical",
             pre_retrieved=pre_retrieved, db_path=db_path), phases,
    )
    assert response == {
        "question": "unfindabletoken",
        "answer": "未找到相关证据，无法回答该问题。",
        "effective_alpha": None,
        "citations": [],
        "citation_valid": True,
        "citation_warnings": [],
    }


def test_stream_keeps_blocking_work_off_the_event_loop(monkeypatch, evidence_db):
    db_path, _ = evidence_db
    generate = install_mock_llm(monkeypatch, "Grounded answer [1].")
    event_loop_thread = threading.get_ident()
    observed = []

    def check_thread(name, operation):
        def run(*args, **kwargs):
            assert threading.get_ident() != event_loop_thread, f"{name} blocked the event loop"
            observed.append(name)
            return operation(*args, **kwargs)
        return run

    for name in ["search_lexical", "build_evidence", "verify_citations"]:
        monkeypatch.setattr(rag_answer, name, check_thread(name, getattr(rag_answer, name)))
    generate.side_effect = check_thread("generate", MockLLMProvider().generate)
    events = asyncio.run(collect_stream(
        question="evidence", retrieval_mode="lexical", db_path=db_path,
    ))
    assert observed == ["search_lexical", "build_evidence", "generate", "verify_citations"]
    assert events[-1] == ("done", "")


def test_provider_failure_does_not_emit_a_successful_answer(monkeypatch, evidence_db):
    db_path, results = evidence_db
    generate = install_mock_llm(monkeypatch, "Unused answer [1].")
    generate.side_effect = RuntimeError("provider failed")
    kwargs = dict(question="How does grounding work?", pre_retrieved=results, db_path=db_path)
    with pytest.raises(RuntimeError, match="provider failed"):
        rag_answer.answer_question(**kwargs)

    emitted = []

    async def consume_failure():
        async for raw in rag_answer.answer_question_stream(**kwargs):
            emitted.append(decode_event(raw))

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(consume_failure())
    assert [name for name, _ in emitted] == ["status", "status"]
    assert [data["phase"] for _, data in emitted] == ["organizing", "generating"]


def test_runtime_budget_is_applied_to_both_answer_paths(monkeypatch, evidence_db):
    db_path, results = evidence_db
    monkeypatch.setenv("CITEQUEST_RAG_CONTEXT_TOKENS", "1")
    response = assert_same_response(
        dict(question="How does grounding work?", pre_retrieved=results, db_path=db_path),
        ["organizing"],
    )
    assert response["citations"] == []
    assert response["answer"] == "未找到相关证据，无法回答该问题。"


def test_both_answer_paths_default_to_five_candidates(monkeypatch, evidence_db):
    db_path, results = evidence_db
    with sqlite3.connect(db_path) as conn:
        for number in (5, 6):
            conn.execute(
                "INSERT INTO chunks(chunk_id, paper_id, chunk_text) VALUES (?, 'P3', ?)",
                (f"C{number}", f"Complete evidence {number}."),
            )
            results.append(results[-1].model_copy(update={"chunk_id": f"C{number}"}))
    generate = install_mock_llm(monkeypatch, "Grounded answer [5].")
    response = assert_same_response(
        dict(question="How does grounding work?", pre_retrieved=results, db_path=db_path),
        ["organizing", "generating", "verifying"],
    )
    assert [c["chunk_id"] for c in response["citations"]] == [f"C{i}" for i in range(1, 6)]
    assert response["citation_valid"] is True
    assert "Complete evidence 6." not in generate.call_args.kwargs["user"]
