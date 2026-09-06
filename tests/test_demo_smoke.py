"""Tests for the operational 10k demo acceptance workflow."""

from __future__ import annotations

import json
import sqlite3

from app.core.schemas import SearchResult
from app.eval.demo_smoke import (
    collect_database_signature,
    render_demo_smoke_markdown,
    run_demo_smoke,
    run_retrieval_smoke,
    validate_demo_artifacts,
    write_demo_smoke_outputs,
)
from app.retrieval.embeddings import DEFAULT_MODEL_NAME


def _build_demo_artifacts(tmp_path, embedding_model):
    import faiss

    db_path = tmp_path / "metadata.sqlite"
    index_dir = tmp_path / "faiss"
    raw_path = tmp_path / "papers.jsonl"
    index_dir.mkdir()
    papers = [
        ("P1", "Retrieval methods for grounded generation", "retrieval evidence grounding"),
        ("P2", "Vision transformer detection", "vision object detection"),
        ("P3", "Code generation with execution feedback", "code execution feedback"),
        ("P4", "Database query optimization", "database query planning"),
        ("P5", "Graph representation learning", "graph neural representation"),
        ("P6", "Robot control with reinforcement learning", "robot control policies"),
    ]
    raw_path.write_text(
        "".join(
            json.dumps({"paper_id": paper_id, "title": title, "abstract": abstract})
            + "\n"
            for paper_id, title, abstract in papers
        ),
        encoding="utf-8",
    )

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL DEFAULT '',
            full_text TEXT NOT NULL DEFAULT '',
            year INTEGER,
            venue TEXT,
            authors_json TEXT NOT NULL DEFAULT '[]',
            concepts_json TEXT NOT NULL DEFAULT '[]',
            doi TEXT,
            url TEXT,
            citation_count INTEGER NOT NULL DEFAULT 0,
            open_access INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_type TEXT NOT NULL DEFAULT 'metadata',
            token_count INTEGER NOT NULL DEFAULT 0,
            position INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
        );
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED,
            paper_id UNINDEXED,
            title,
            chunk_text
        );
        """
    )
    chunks = []
    for paper_id, title, abstract in papers:
        conn.execute(
            """INSERT INTO papers
               (paper_id, title, abstract, year, venue, authors_json, concepts_json)
               VALUES (?, ?, ?, 2025, 'arXiv', '[]', '[]')""",
            (paper_id, title, abstract),
        )
        chunk_id = f"{paper_id}_chunk0"
        chunk_text = f"Title: {title}\nAbstract: {abstract}"
        chunks.append((chunk_id, paper_id, chunk_text))
        conn.execute(
            """INSERT INTO chunks
               (chunk_id, paper_id, chunk_text, chunk_type, token_count, position)
               VALUES (?, ?, ?, 'metadata', 20, 0)""",
            (chunk_id, paper_id, chunk_text),
        )
    conn.execute(
        """INSERT INTO chunk_fts(chunk_id, paper_id, title, chunk_text)
           SELECT c.chunk_id, c.paper_id, p.title, c.chunk_text
           FROM chunks c JOIN papers p ON p.paper_id = c.paper_id"""
    )
    conn.commit()
    conn.close()

    vectors = embedding_model().encode([chunk[2] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "id_map.json").write_text(
        json.dumps(
            [
                {"faiss_id": position, "chunk_id": chunk_id, "paper_id": paper_id}
                for position, (chunk_id, paper_id, _) in enumerate(chunks)
            ]
        ),
        encoding="utf-8",
    )
    (index_dir / "build_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "complete",
                "embedding_model": DEFAULT_MODEL_NAME,
                "vector_dim": vectors.shape[1],
                "num_vectors": len(chunks),
                "index_type": "IndexFlatIP",
                "nlist": None,
                "db_signature": collect_database_signature(db_path),
            }
        ),
        encoding="utf-8",
    )
    return db_path, index_dir, raw_path, len(papers)


def _check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def test_validate_demo_artifacts_accepts_consistent_index(tmp_path, tiny_embedding_model):
    db_path, index_dir, raw_path, paper_count = _build_demo_artifacts(tmp_path, tiny_embedding_model)

    report = validate_demo_artifacts(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=raw_path,
        expected_papers=paper_count,
    )

    assert report["passed"] is True
    assert report["facts"]["paper_count"] == paper_count
    assert report["facts"]["chunk_count"] == paper_count
    assert report["facts"]["faiss_vector_count"] == paper_count
    assert _check(report, "id_map_matches_sqlite_order")["passed"] is True
    assert _check(report, "database_matches_faiss_build")["passed"] is True


def test_validate_demo_artifacts_rejects_id_map_order_mismatch(tmp_path, tiny_embedding_model):
    db_path, index_dir, raw_path, paper_count = _build_demo_artifacts(tmp_path, tiny_embedding_model)
    id_map_path = index_dir / "id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map[0], id_map[1] = id_map[1], id_map[0]
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")

    report = validate_demo_artifacts(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=raw_path,
        expected_papers=paper_count,
    )

    assert report["passed"] is False
    assert _check(report, "id_map_matches_sqlite_order")["passed"] is False


def test_validate_demo_artifacts_rejects_partial_checkpoint(tmp_path, tiny_embedding_model):
    db_path, index_dir, raw_path, paper_count = _build_demo_artifacts(tmp_path, tiny_embedding_model)
    (index_dir / ".build").mkdir()

    report = validate_demo_artifacts(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=raw_path,
        expected_papers=paper_count,
    )

    assert report["passed"] is False
    assert _check(report, "faiss_checkpoint_absent")["passed"] is False


def test_run_retrieval_smoke_reports_latency_without_quality_metrics(tmp_path):
    def factory(method):
        def search(query, top_k):
            suffix = {"bm25": "B", "dense": "D", "hybrid": "H"}[method]
            return [
                SearchResult(
                    paper_id=f"{suffix}1",
                    chunk_id=f"{suffix}1_chunk0",
                    title=f"{query} result",
                    year=2025,
                    venue="arXiv",
                    score=1.0,
                )
            ][:top_k]

        return search

    report = run_retrieval_smoke(
        queries=("retrieval", "graph"),
        db_path=tmp_path / "unused.sqlite",
        index_dir=tmp_path / "unused-faiss",
        top_k=3,
        runs=2,
        retriever_factory=factory,
    )

    assert report["passed"] is True
    assert report["quality_metrics_computed"] is False
    assert report["methods"]["bm25"]["measured_searches"] == 4
    assert report["methods"]["dense"]["latency"]["p95_ms"] >= 0
    assert len(report["cross_method_overlap"]) == 2


def test_complete_demo_smoke_uses_real_retrievers_and_mock_rag(
    tmp_path, monkeypatch, tiny_embedding_model
):
    import app.retrieval.vector_store as vector_store

    db_path, index_dir, raw_path, paper_count = _build_demo_artifacts(tmp_path, tiny_embedding_model)
    monkeypatch.setattr(vector_store, "EmbeddingModel", tiny_embedding_model)
    monkeypatch.setattr(vector_store, "_index_cache", None)
    monkeypatch.setattr(vector_store, "_model_cache", None)

    report = run_demo_smoke(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=raw_path,
        queries=("retrieval grounding", "graph learning"),
        expected_papers=paper_count,
        top_k=3,
        runs=1,
    )

    assert report["status"] == "passed"
    assert report["official_benchmark"] is False
    assert report["retrieval"]["methods"]["dense"]["passed"] is True
    assert report["api"]["endpoints"]["search_hybrid"]["passed"] is True
    assert report["api"]["endpoints"]["ask_mock_rag"]["citation_valid"] is True
    assert report["api"]["external_llm_called"] is False

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_demo_smoke_outputs(
        report, json_path=json_path, markdown_path=markdown_path
    )
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert persisted["quality_metrics_computed"] is False
    assert "not Retrieval Benchmark v1" in markdown
    assert "HitRate" not in markdown


def test_render_failed_report_keeps_demo_disclaimer():
    report = {
        "status": "failed",
        "created_at": "2026-08-20T00:00:00+00:00",
        "provenance": {"git_commit": "abc123", "git_dirty": True},
        "artifacts": {"facts": {}, "checks": []},
        "retrieval": {"skipped": True, "reason": "artifact validation failed"},
        "api": {"skipped": True, "reason": "artifact validation failed"},
    }

    markdown = render_demo_smoke_markdown(report)

    assert "Operational smoke only" in markdown
    assert "not Retrieval Benchmark v1" in markdown
