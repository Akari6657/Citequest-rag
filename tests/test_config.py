"""Tests for runtime path configuration."""

from __future__ import annotations

import sqlite3

import pytest

from app.core.config import (
    DEFAULT_DB_PATH,
    DEFAULT_FAISS_DIR,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_REWRITE_TIMEOUT_SECONDS,
    DEFAULT_RAG_CONTEXT_TOKENS,
    get_db_path,
    get_faiss_dir,
    get_hybrid_alpha,
    get_rewrite_timeout_seconds,
    get_rag_context_tokens,
    resolve_hybrid_alpha,
)
from app.main import health


def test_default_paths(monkeypatch):
    monkeypatch.delenv("CITEQUEST_DB_PATH", raising=False)
    monkeypatch.delenv("CITEQUEST_FAISS_DIR", raising=False)

    assert get_db_path() == DEFAULT_DB_PATH
    assert get_faiss_dir() == DEFAULT_FAISS_DIR


def test_env_paths(monkeypatch, tmp_path):
    db_path = tmp_path / "demo" / "metadata.sqlite"
    faiss_dir = tmp_path / "demo" / "faiss"

    monkeypatch.setenv("CITEQUEST_DB_PATH", str(db_path))
    monkeypatch.setenv("CITEQUEST_FAISS_DIR", str(faiss_dir))

    assert get_db_path() == db_path
    assert get_faiss_dir() == faiss_dir


def test_default_hybrid_alpha(monkeypatch):
    monkeypatch.delenv("CITEQUEST_HYBRID_ALPHA", raising=False)

    assert get_hybrid_alpha() == DEFAULT_HYBRID_ALPHA == 0.5


def test_hybrid_alpha_from_environment(monkeypatch):
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "0.65")

    assert get_hybrid_alpha() == 0.65


def test_explicit_hybrid_alpha_overrides_environment(monkeypatch):
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", "invalid")

    assert resolve_hybrid_alpha(0.2) == 0.2


def test_default_rewrite_timeout(monkeypatch):
    monkeypatch.delenv("CITEQUEST_REWRITE_TIMEOUT_SECONDS", raising=False)

    assert get_rewrite_timeout_seconds() == DEFAULT_REWRITE_TIMEOUT_SECONDS == 2.0


def test_rewrite_timeout_from_environment(monkeypatch):
    monkeypatch.setenv("CITEQUEST_REWRITE_TIMEOUT_SECONDS", "0.75")

    assert get_rewrite_timeout_seconds() == 0.75


@pytest.mark.parametrize("value", ["", "invalid", "0", "-0.1", "nan", "inf"])
def test_invalid_rewrite_timeout_fails_clearly(monkeypatch, value):
    monkeypatch.setenv("CITEQUEST_REWRITE_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="CITEQUEST_REWRITE_TIMEOUT_SECONDS"):
        get_rewrite_timeout_seconds()


@pytest.mark.parametrize("value", ["", "invalid", "-0.1", "1.1", "nan", "inf"])
def test_invalid_environment_hybrid_alpha_fails_clearly(monkeypatch, value):
    monkeypatch.setenv("CITEQUEST_HYBRID_ALPHA", value)

    with pytest.raises(ValueError, match="CITEQUEST_HYBRID_ALPHA"):
        get_hybrid_alpha()


def test_health_uses_env_paths(monkeypatch, tmp_path):
    db_path = tmp_path / "demo" / "metadata.sqlite"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE chunk_fts(id TEXT)")
    conn.commit()
    conn.close()

    faiss_dir = tmp_path / "demo" / "faiss"
    faiss_dir.mkdir()
    (faiss_dir / "index.faiss").write_text("fake", encoding="utf-8")
    (faiss_dir / "id_map.json").write_text("[]", encoding="utf-8")

    monkeypatch.setenv("CITEQUEST_DB_PATH", str(db_path))
    monkeypatch.setenv("CITEQUEST_FAISS_DIR", str(faiss_dir))

    response = health()

    assert response["status"] == "healthy"
    assert response["paths"] == {
        "metadata_db": str(db_path),
        "faiss_dir": str(faiss_dir),
    }
    assert response["capabilities"]["lexical_search"] is True
    assert response["capabilities"]["vector_search"] is True
    assert response["capabilities"]["hybrid_search"] is True


def test_rag_context_budget_default_and_override(monkeypatch):
    monkeypatch.delenv("CITEQUEST_RAG_CONTEXT_TOKENS", raising=False)
    assert get_rag_context_tokens() == DEFAULT_RAG_CONTEXT_TOKENS == 8000
    monkeypatch.setenv("CITEQUEST_RAG_CONTEXT_TOKENS", "12000")
    assert get_rag_context_tokens() == 12000


@pytest.mark.parametrize("value", ["", "invalid", "0", "-1", "1.5", "nan", "inf"])
def test_invalid_rag_context_budget_fails_clearly(monkeypatch, value):
    monkeypatch.setenv("CITEQUEST_RAG_CONTEXT_TOKENS", value)
    with pytest.raises(ValueError, match="CITEQUEST_RAG_CONTEXT_TOKENS"):
        get_rag_context_tokens()
