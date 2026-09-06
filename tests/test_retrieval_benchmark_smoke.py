"""Small end-to-end benchmark smoke test using real FTS5/FAISS retrievers."""

from __future__ import annotations

import json
import sqlite3

from app.eval.fusion_experiment import run_fusion_comparison
from app.eval.retrieval_eval import EvalQuery, run_benchmark, write_benchmark_outputs


def _build_smoke_artifacts(tmp_path, embedding_model):
    import faiss

    db_path = tmp_path / "metadata.sqlite"
    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    papers = [
        ("P1", "Retrieval methods for grounded generation", "retrieval evidence grounding"),
        ("P2", "Vision transformer detection", "vision object detection"),
        ("P3", "Code generation with execution feedback", "code execution feedback"),
        ("P4", "Database query optimization", "database query planning"),
        ("P5", "Graph representation learning", "graph neural representation"),
        ("P6", "Robot control with reinforcement learning", "robot control policies"),
    ]

    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER,
            venue TEXT, authors_json TEXT DEFAULT '[]'
        )"""
    )
    conn.execute(
        """CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, paper_id TEXT, chunk_text TEXT,
            chunk_type TEXT, token_count INTEGER, position INTEGER
        )"""
    )
    chunks = []
    for paper_id, title, abstract in papers:
        conn.execute(
            "INSERT INTO papers VALUES (?, ?, ?, 2025, 'arXiv', '[]')",
            (paper_id, title, abstract),
        )
        chunk_text = f"Title: {title}\nAbstract: {abstract}"
        chunk_id = f"{paper_id}_chunk0"
        chunks.append((chunk_id, paper_id, chunk_text))
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, 'title_abstract', 20, 0)",
            (chunk_id, paper_id, chunk_text),
        )
    conn.execute(
        """CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED, paper_id UNINDEXED, title, chunk_text
        )"""
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
                {"faiss_id": index, "chunk_id": chunk_id, "paper_id": paper_id}
                for index, (chunk_id, paper_id, _) in enumerate(chunks)
            ]
        ),
        encoding="utf-8",
    )
    return db_path, index_dir


def test_end_to_end_retrieval_benchmark_smoke(tmp_path, monkeypatch, tiny_embedding_model):
    import app.retrieval.vector_store as vector_store

    db_path, index_dir = _build_smoke_artifacts(tmp_path, tiny_embedding_model)
    monkeypatch.setattr(vector_store, "EmbeddingModel", tiny_embedding_model)
    monkeypatch.setattr(vector_store, "_index_cache", None)
    monkeypatch.setattr(vector_store, "_model_cache", None)

    queries = [
        EvalQuery("q0001", "retrieval grounding", "keyword", "dev", ("P1",)),
        EvalQuery(
            "q0002", "How can vision models detect objects?", "natural_question", "dev", ("P2",)
        ),
        EvalQuery(
            "q0003", "program synthesis using runtime results", "semantic_paraphrase", "dev", ("P3",)
        ),
        EvalQuery("q0004", "database optimization", "keyword", "test", ("P4",)),
        EvalQuery(
            "q0005", "How are graph representations learned?", "natural_question", "test", ("P5",)
        ),
        EvalQuery(
            "q0006", "policy learning for autonomous control", "semantic_paraphrase", "test", ("P6",)
        ),
    ]
    manifest = {
        "corpus": "synthetic_smoke",
        "paper_count": 6,
        "chunk_count": 6,
        "embedding_model": "deterministic-test-double",
        "embedding_dim": 6,
        "faiss_index_type": "IndexFlatIP",
        "faiss_nlist": None,
        "faiss_nprobe": None,
        "environment": {"platform": "pytest"},
    }

    result = run_benchmark(
        queries,
        db_path=db_path,
        index_dir=index_dir,
        manifest=manifest,
        retrieval_depth=10,
    )
    manifest_path = tmp_path / "manifest.json"
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    write_benchmark_outputs(
        result,
        manifest_path=manifest_path,
        json_report_path=json_path,
        markdown_report_path=markdown_path,
    )

    assert result["status"] == "smoke_or_development"
    assert result["selected_alpha"] in {0.2, 0.35, 0.5, 0.65, 0.8}
    assert result["test_results"]["dense"]["metrics"]["hit_rate@10"] == 1.0
    assert manifest_path.exists() and json_path.exists() and markdown_path.exists()
    assert "Smoke / Development Run" in markdown_path.read_text(encoding="utf-8")


def test_end_to_end_fusion_comparison_six_query_smoke(tmp_path, monkeypatch, tiny_embedding_model):
    import app.retrieval.vector_store as vector_store
    from app.retrieval.lexical import search_lexical
    from app.retrieval.vector_store import search_vector

    db_path, index_dir = _build_smoke_artifacts(tmp_path, tiny_embedding_model)
    monkeypatch.setattr(vector_store, "EmbeddingModel", tiny_embedding_model)
    monkeypatch.setattr(vector_store, "_index_cache", None)
    monkeypatch.setattr(vector_store, "_model_cache", None)

    queries = [
        EvalQuery("q0001", "retrieval grounding", "keyword", "dev", ("P1",)),
        EvalQuery(
            "q0002",
            "How can vision models detect objects?",
            "natural_question",
            "dev",
            ("P2",),
        ),
        EvalQuery(
            "q0003",
            "program synthesis using runtime results",
            "semantic_paraphrase",
            "dev",
            ("P3",),
        ),
        EvalQuery("q0004", "database optimization", "keyword", "dev", ("P4",)),
        EvalQuery(
            "q0005",
            "How are graph representations learned?",
            "natural_question",
            "dev",
            ("P5",),
        ),
        EvalQuery(
            "q0006",
            "policy learning for autonomous control",
            "semantic_paraphrase",
            "dev",
            ("P6",),
        ),
    ]

    report = run_fusion_comparison(
        queries,
        lambda query, top_k: search_lexical(
            query,
            top_k=top_k,
            db_path=db_path,
        ),
        lambda query, top_k: search_vector(
            query,
            top_k=top_k,
            db_path=db_path,
            index_dir=index_dir,
        ),
    )

    assert report["status"] == "smoke_or_development"
    assert report["query_count"] == 6
    assert report["protocol"]["candidate_depth"] == 20
    assert report["protocol"]["final_top_k"] == 10
    assert len(report["candidate_audit"]) == 6
    assert report["methods"]["minmax"]["metrics"]["query_count"] == 6
    assert report["methods"]["rrf"]["metrics"]["query_count"] == 6
    pairwise = report["pairwise_first_relevant_rank"]
    assert pairwise["minmax_wins"] + pairwise["rrf_wins"] + pairwise["ties"] == 6
