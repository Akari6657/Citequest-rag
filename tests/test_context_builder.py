"""Evidence must come from the local corpus, including its citation metadata."""

import sqlite3

import pytest

from app.core.schemas import SearchResult
from app.rag.context_builder import _estimate_tokens, build_evidence
from scripts.build_metadata_db import CREATE_TABLES


@pytest.fixture
def evidence_db(tmp_path):
    path = tmp_path / "evidence.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(CREATE_TABLES)
        conn.executemany(
            "INSERT INTO papers(paper_id, title, year, venue, url) VALUES (?, ?, ?, ?, ?)",
            [("P1", "Local title one", 2024, "Local venue", "https://example.test/paper/1"),
             ("P2", "Local title two", 2023, None, None),
             ("P3", "Empty paper", None, None, None)],
        )
        conn.executemany(
            "INSERT INTO chunks(chunk_id, paper_id, chunk_text, chunk_type) VALUES (?, ?, ?, ?)",
            [("C1", "P1", "Title: Local title one\nAbstract: Verified abstract one.", "title_abstract"),
             ("C1-body", "P1", "Body evidence.\nAbstract: This is part of the body.", "body"),
             ("C2", "P2", "Title: Local title two\nAbstract: Verified abstract two.", "metadata"),
             ("C3", "P3", " \n ", "title_abstract")],
        )
    return path


def result(chunk_id="C1", paper_id="P1"):
    return SearchResult(
        chunk_id=chunk_id, paper_id=paper_id, title="Untrusted request title",
        year=2099, venue="Untrusted venue", score=1.0,
        snippet="Untrusted snippet", abstract="Untrusted abstract",
    )


def test_empty_results_do_not_open_a_database(tmp_path):
    missing = tmp_path / "missing.sqlite"
    assert build_evidence([], db_path=missing) == ("", [])
    assert not missing.exists()


def test_text_and_citation_metadata_are_loaded_from_sqlite(evidence_db):
    text, citations = build_evidence([result()], db_path=evidence_db)
    assert "Verified abstract one." in text
    assert "Local title one" in text and "Local venue" in text and "2024" in text
    assert "Untrusted" not in text and "2099" not in text
    assert citations == [{
        "citation_id": 1, "paper_id": "P1", "chunk_id": "C1",
        "title": "Local title one", "url": "https://example.test/paper/1",
    }]


def test_unknown_mismatched_and_empty_chunks_are_skipped(evidence_db):
    raw = [result("missing"), result("C1", "wrong-paper"), result("C3", "P3"),
           result("C2", "P2")]
    text, citations = build_evidence(raw, db_path=evidence_db)
    assert len(citations) == 1
    assert citations[0]["citation_id"] == 1
    assert citations[0]["chunk_id"] == "C2"
    assert citations[0]["url"] is None  # Never synthesize an arXiv URL for another corpus.
    assert "Verified abstract two." in text
    assert "Untrusted" not in text and "Verified abstract one." not in text


def test_duplicate_chunks_are_removed_but_distinct_chunks_of_a_paper_remain(evidence_db):
    raw = [result("C1", "wrong-paper"), result(), result(), result("C1-body")]
    text, citations = build_evidence(raw, db_path=evidence_db)
    assert [c["chunk_id"] for c in citations] == ["C1", "C1-body"]
    assert [c["citation_id"] for c in citations] == [1, 2]
    assert text.count("Verified abstract one.") == 1
    assert "Body evidence.\nAbstract: This is part of the body." in text


def test_missing_database_is_not_created(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        build_evidence([result()], db_path=missing)
    assert not missing.exists()


def test_complete_chunks_keep_retrieval_order_when_they_fit(evidence_db):
    raw = [result("C2", "P2"), result(), result("C1-body")]
    text, citations = build_evidence(raw, db_path=evidence_db, max_tokens=8000)
    assert [c["chunk_id"] for c in citations] == ["C2", "C1", "C1-body"]
    assert [c["citation_id"] for c in citations] == [1, 2, 3]
    assert "已截断" not in text


@pytest.mark.parametrize("content", ["English evidence. " * 1000, "中文证据🙂。" * 1000])
def test_even_the_first_chunk_must_fit_the_estimated_budget(evidence_db, content):
    with sqlite3.connect(evidence_db) as conn:
        conn.execute("UPDATE chunks SET chunk_text=? WHERE chunk_id='C1'", (content,))
    # A later short chunk must not replace an oversized first candidate.
    assert build_evidence(
        [result(), result("C2", "P2")], db_path=evidence_db, max_tokens=96,
    ) == ("", [])


def test_stop_before_overflow_even_when_a_later_shorter_chunk_would_fit(evidence_db):
    first = result()
    last = result("C2", "P2")
    with sqlite3.connect(evidence_db) as conn:
        conn.execute("UPDATE chunks SET chunk_text=? WHERE chunk_id='C1-body'", ("Long body. " * 1000,))
    fitting_text, _ = build_evidence([first, last], db_path=evidence_db)
    first_text, _ = build_evidence([first], db_path=evidence_db)
    text, citations = build_evidence(
        [first, result("C1-body"), last], db_path=evidence_db,
        max_tokens=_estimate_tokens(fitting_text),
    )
    assert text == first_text
    assert [c["chunk_id"] for c in citations] == ["C1"]


def test_citations_only_describe_included_evidence(evidence_db):
    first_text, _ = build_evidence([result()], db_path=evidence_db, max_tokens=8000)
    text, citations = build_evidence(
        [result(), result("C2", "P2")], db_path=evidence_db,
        max_tokens=_estimate_tokens(first_text),
    )
    assert text == first_text
    assert [c["chunk_id"] for c in citations] == ["C1"]


def test_explicit_budget_overrides_runtime_configuration(monkeypatch, evidence_db):
    monkeypatch.setenv("CITEQUEST_RAG_CONTEXT_TOKENS", "invalid")
    _, citations = build_evidence([result()], db_path=evidence_db, max_tokens=8000)
    assert len(citations) == 1


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_explicit_budget_is_rejected(evidence_db, budget):
    with pytest.raises(ValueError, match="max_tokens"):
        build_evidence([result()], db_path=evidence_db, max_tokens=budget)
