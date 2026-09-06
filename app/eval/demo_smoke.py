"""Operational acceptance checks for a local CiteQuest demo index.

This module deliberately does not calculate retrieval-quality metrics. A demo
corpus without frozen relevance judgments can prove that the system is
consistent and runnable, but it cannot support HitRate, MRR, or nDCG claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import patch

from app.core.config import DEFAULT_RAG_TOP_K
from app.core.schemas import SearchResult
from app.eval.runtime_info import collect_accelerator_info
from app.retrieval.embeddings import DEFAULT_MODEL_NAME
from app.retrieval.hybrid import search_hybrid
from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DEMO_QUERIES = (
    "machine learning",
    "climate change",
    "gene expression",
    "quantum computing",
    "natural language processing",
)
METHODS = ("bm25", "dense", "hybrid")

SearchFn = Callable[[str, int], list[SearchResult]]
RetrieverFactory = Callable[[str], SearchFn]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _hash_and_count_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    line_count = 0
    last_byte = b""
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            line_count += block.count(b"\n")
            last_byte = block[-1:]
    if path.stat().st_size and last_byte != b"\n":
        line_count += 1
    return digest.hexdigest(), line_count


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": round(sum(values) / len(values), 3) if values else 0.0,
        "p50_ms": round(_percentile(values, 50), 3),
        "p95_ms": round(_percentile(values, 95), 3),
    }


def collect_database_signature(db_path: str | Path) -> dict[str, Any]:
    """Return the corpus identity fields recorded by ``build_faiss.py``."""
    db_path = Path(db_path).resolve()
    stat = db_path.stat()
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            """SELECT COUNT(*), COALESCE(MIN(rowid), 0), COALESCE(MAX(rowid), 0)
               FROM chunks"""
        ).fetchone()
        first = conn.execute(
            "SELECT chunk_id FROM chunks ORDER BY rowid LIMIT 1"
        ).fetchone()
        last = conn.execute(
            "SELECT chunk_id FROM chunks ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return {
        "path": str(db_path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "chunk_count": int(row[0]),
        "min_rowid": int(row[1]),
        "max_rowid": int(row[2]),
        "first_chunk_id": first[0] if first else None,
        "last_chunk_id": last[0] if last else None,
    }


def validate_demo_artifacts(
    *,
    db_path: str | Path,
    index_dir: str | Path,
    raw_path: str | Path | None = None,
    expected_papers: int | None = None,
    expected_model: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Validate SQLite, FTS5, FAISS, ID-map, and build provenance together."""
    db_path = Path(db_path).resolve()
    index_dir = Path(index_dir).resolve()
    raw_path = Path(raw_path).resolve() if raw_path is not None else None
    index_path = index_dir / "index.faiss"
    id_map_path = index_dir / "id_map.json"
    build_meta_path = index_dir / "build_meta.json"
    checkpoint_path = index_dir / ".build"

    checks: list[dict[str, Any]] = []
    facts: dict[str, Any] = {
        "database_file": _display_path(db_path),
        "index_dir": _display_path(index_dir),
        "raw_file": _display_path(raw_path) if raw_path else None,
    }

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    required_paths = {
        "metadata_database_exists": db_path,
        "faiss_index_exists": index_path,
        "faiss_id_map_exists": id_map_path,
        "faiss_build_metadata_exists": build_meta_path,
    }
    if raw_path is not None:
        required_paths["raw_corpus_exists"] = raw_path
    for name, path in required_paths.items():
        record(name, path.is_file(), _display_path(path))
    record(
        "faiss_checkpoint_absent",
        not checkpoint_path.exists(),
        "No partial build checkpoint remains"
        if not checkpoint_path.exists()
        else f"Incomplete or active build checkpoint: {_display_path(checkpoint_path)}",
    )

    if checkpoint_path.exists():
        facts["build_in_progress_or_incomplete"] = True
        return {"passed": False, "checks": checks, "facts": facts}

    if not db_path.is_file():
        return {"passed": False, "checks": checks, "facts": facts}

    paper_count = 0
    chunk_count = 0
    fts_row_count = 0
    database_signature: dict[str, Any] | None = None
    conn = None
    try:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        missing_tables = sorted({"papers", "chunks", "chunk_fts"} - tables)
        record(
            "required_sqlite_tables",
            not missing_tables,
            "papers, chunks, and chunk_fts are present"
            if not missing_tables
            else f"Missing tables: {', '.join(missing_tables)}",
        )

        quick_check = [row[0] for row in conn.execute("PRAGMA quick_check")]
        record(
            "sqlite_quick_check",
            quick_check == ["ok"],
            "; ".join(quick_check),
        )

        if "papers" in tables:
            paper_count = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
            facts["paper_count"] = paper_count
            record("papers_nonempty", paper_count > 0, f"{paper_count:,} papers")
            if expected_papers is not None:
                record(
                    "expected_paper_count",
                    paper_count == expected_papers,
                    f"observed={paper_count:,}, expected={expected_papers:,}",
                )

        if "chunks" in tables:
            chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            facts["chunk_count"] = chunk_count
            record("chunks_nonempty", chunk_count > 0, f"{chunk_count:,} chunks")
            if "papers" in tables:
                orphan_count = int(
                    conn.execute(
                        """SELECT COUNT(*) FROM chunks c
                           LEFT JOIN papers p ON p.paper_id = c.paper_id
                           WHERE p.paper_id IS NULL"""
                    ).fetchone()[0]
                )
                facts["orphan_chunk_count"] = orphan_count
                record(
                    "no_orphan_chunks",
                    orphan_count == 0,
                    f"{orphan_count:,} orphan chunks",
                )

        foreign_key_errors = list(conn.execute("PRAGMA foreign_key_check"))
        record(
            "sqlite_foreign_key_check",
            not foreign_key_errors,
            f"{len(foreign_key_errors):,} violations",
        )

        if "chunk_fts" in tables and "chunks" in tables:
            fts_row_count = int(
                conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
            )
            facts["fts_row_count"] = fts_row_count
            # chunk_id is UNINDEXED in FTS5. Two linear scans plus set
            # comparison avoid a correlated lookup that can become quadratic.
            chunk_ids = {
                row[0] for row in conn.execute("SELECT chunk_id FROM chunks")
            }
            fts_ids = [
                row[0] for row in conn.execute("SELECT chunk_id FROM chunk_fts")
            ]
            unique_fts_ids = set(fts_ids)
            missing_from_fts = len(chunk_ids - unique_fts_ids)
            unknown_fts_rows = len(unique_fts_ids - chunk_ids)
            duplicate_fts_rows = len(fts_ids) - len(unique_fts_ids)
            facts.update(
                {
                    "chunks_missing_from_fts": missing_from_fts,
                    "unknown_fts_rows": unknown_fts_rows,
                    "duplicate_fts_rows": duplicate_fts_rows,
                }
            )
            record(
                "fts_matches_chunks",
                fts_row_count == chunk_count
                and missing_from_fts == 0
                and unknown_fts_rows == 0
                and duplicate_fts_rows == 0,
                (
                    f"rows={fts_row_count:,}, missing={missing_from_fts:,}, "
                    f"unknown={unknown_fts_rows:,}, duplicate_rows={duplicate_fts_rows:,}"
                ),
            )
    except sqlite3.Error as exc:
        record("sqlite_readable", False, str(exc))
    finally:
        if conn is not None:
            conn.close()

    try:
        database_signature = collect_database_signature(db_path)
        facts["database_signature"] = database_signature
    except (OSError, sqlite3.Error) as exc:
        record("database_signature_readable", False, str(exc))

    if raw_path is not None and raw_path.is_file():
        raw_sha256, raw_line_count = _hash_and_count_lines(raw_path)
        facts.update(
            {
                "raw_file_sha256": raw_sha256,
                "raw_line_count": raw_line_count,
                "raw_size_bytes": raw_path.stat().st_size,
            }
        )
        record(
            "raw_corpus_covers_papers",
            raw_line_count >= paper_count > 0,
            f"raw_records={raw_line_count:,}, indexed_papers={paper_count:,}",
        )

    index = None
    if index_path.is_file():
        try:
            import faiss

            index = faiss.read_index(str(index_path))
            nlist = int(index.nlist) if hasattr(index, "nlist") else None
            nprobe = min(nlist // 4, 64) if nlist is not None else None
            facts.update(
                {
                    "faiss_index_type": index.__class__.__name__,
                    "faiss_vector_count": int(index.ntotal),
                    "embedding_dim": int(index.d),
                    "faiss_nlist": nlist,
                    "runtime_nprobe": nprobe,
                    "faiss_index_size_bytes": index_path.stat().st_size,
                }
            )
            record("faiss_index_trained", bool(index.is_trained), str(index.is_trained))
            record(
                "faiss_vectors_match_chunks",
                int(index.ntotal) == chunk_count,
                f"vectors={int(index.ntotal):,}, chunks={chunk_count:,}",
            )
        except Exception as exc:
            record("faiss_index_readable", False, str(exc))

    id_map: Any = None
    if id_map_path.is_file():
        try:
            with open(id_map_path, "r", encoding="utf-8") as handle:
                id_map = json.load(handle)
            is_list = isinstance(id_map, list)
            record("faiss_id_map_is_list", is_list, type(id_map).__name__)
            if is_list:
                facts["id_map_count"] = len(id_map)
                record(
                    "id_map_count_matches_chunks",
                    len(id_map) == chunk_count,
                    f"id_map={len(id_map):,}, chunks={chunk_count:,}",
                )
        except (OSError, json.JSONDecodeError) as exc:
            record("faiss_id_map_readable", False, str(exc))

    if isinstance(id_map, list) and len(id_map) == chunk_count and chunk_count > 0:
        mismatch: str | None = None
        conn = None
        try:
            conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
            rows = conn.execute(
                "SELECT chunk_id, paper_id FROM chunks ORDER BY rowid"
            )
            for faiss_id, row in enumerate(rows):
                entry = id_map[faiss_id]
                if (
                    not isinstance(entry, dict)
                    or entry.get("faiss_id") != faiss_id
                    or entry.get("chunk_id") != row[0]
                    or entry.get("paper_id") != row[1]
                ):
                    mismatch = f"first mismatch at FAISS position {faiss_id}"
                    break
        except sqlite3.Error as exc:
            mismatch = str(exc)
        finally:
            if conn is not None:
                conn.close()
        record(
            "id_map_matches_sqlite_order",
            mismatch is None,
            mismatch or f"All {chunk_count:,} entries match SQLite row order",
        )

    build_meta: Any = None
    if build_meta_path.is_file():
        try:
            with open(build_meta_path, "r", encoding="utf-8") as handle:
                build_meta = json.load(handle)
            record(
                "faiss_build_complete",
                isinstance(build_meta, dict) and build_meta.get("status") == "complete",
                f"status={build_meta.get('status') if isinstance(build_meta, dict) else None}",
            )
        except (OSError, json.JSONDecodeError) as exc:
            record("faiss_build_metadata_readable", False, str(exc))

    if isinstance(build_meta, dict):
        facts["faiss_build"] = build_meta
        record(
            "embedding_model_matches",
            build_meta.get("embedding_model") == expected_model,
            f"observed={build_meta.get('embedding_model')}, expected={expected_model}",
        )
        record(
            "build_vector_count_matches",
            build_meta.get("num_vectors") == chunk_count,
            f"build={build_meta.get('num_vectors')}, chunks={chunk_count}",
        )
        index_dim = int(index.d) if index is not None else None
        record(
            "build_dimension_matches",
            index_dim is not None and build_meta.get("vector_dim") == index_dim,
            f"build={build_meta.get('vector_dim')}, index={index_dim}",
        )
        build_signature = build_meta.get("db_signature")
        signature_matches = (
            isinstance(build_signature, dict)
            and database_signature is not None
            and all(
                build_signature.get(field) == database_signature.get(field)
                for field in (
                    "path",
                    "size_bytes",
                    "mtime_ns",
                    "chunk_count",
                    "min_rowid",
                    "max_rowid",
                    "first_chunk_id",
                    "last_chunk_id",
                )
            )
        )
        record(
            "database_matches_faiss_build",
            signature_matches,
            "Current SQLite signature matches build_meta.json"
            if signature_matches
            else "Current SQLite signature differs from build_meta.json",
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "facts": facts,
    }


def _default_retriever_factory(
    db_path: Path,
    index_dir: Path,
    alpha: float,
) -> RetrieverFactory:
    def factory(method: str) -> SearchFn:
        if method == "bm25":
            return lambda query, top_k: search_lexical(
                query, top_k=top_k, db_path=db_path
            )
        if method == "dense":
            return lambda query, top_k: search_vector(
                query, top_k=top_k, db_path=db_path, index_dir=index_dir
            )
        if method == "hybrid":
            return lambda query, top_k: search_hybrid(
                query,
                top_k=top_k,
                alpha=alpha,
                db_path=db_path,
                index_dir=index_dir,
            )
        raise ValueError(f"Unsupported retrieval method: {method}")

    return factory


def _paper_preview(results: Sequence[SearchResult], limit: int) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        if result.paper_id in seen:
            continue
        seen.add(result.paper_id)
        previews.append(
            {
                "paper_id": result.paper_id,
                "chunk_id": result.chunk_id,
                "title": result.title,
                "score": result.score,
            }
        )
        if len(previews) == limit:
            break
    return previews


def run_retrieval_smoke(
    *,
    queries: Sequence[str],
    db_path: str | Path,
    index_dir: str | Path,
    top_k: int = 10,
    runs: int = 3,
    alpha: float = 0.5,
    retriever_factory: RetrieverFactory | None = None,
) -> dict[str, Any]:
    """Warm and time BM25, Dense, and Hybrid without claiming relevance."""
    normalized_queries = [query.strip() for query in queries if query.strip()]
    if not normalized_queries:
        raise ValueError("At least one non-empty smoke query is required")
    if len(set(normalized_queries)) != len(normalized_queries):
        raise ValueError("Smoke queries must be unique")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    db_path = Path(db_path).resolve()
    index_dir = Path(index_dir).resolve()
    factory = retriever_factory or _default_retriever_factory(
        db_path, index_dir, alpha
    )
    method_reports: dict[str, Any] = {}
    rankings: dict[str, dict[str, list[str]]] = {}

    for method in METHODS:
        search_fn = factory(method)
        latencies: list[float] = []
        method_errors: list[str] = []
        per_query: list[dict[str, Any]] = []
        rankings[method] = {}
        try:
            started = time.perf_counter()
            search_fn(normalized_queries[0], top_k)
            warmup_ms = (time.perf_counter() - started) * 1000

            for query in normalized_queries:
                query_latencies: list[float] = []
                first_results: list[SearchResult] | None = None
                first_ranking: list[str] | None = None
                for run_number in range(runs):
                    started = time.perf_counter()
                    results = list(search_fn(query, top_k))
                    latency_ms = (time.perf_counter() - started) * 1000
                    query_latencies.append(latency_ms)
                    latencies.append(latency_ms)

                    if len(results) > top_k:
                        method_errors.append(
                            f"{query!r} run {run_number + 1} returned {len(results)} > top_k={top_k}"
                        )
                    if any(
                        not result.paper_id
                        or not result.chunk_id
                        or not math.isfinite(result.score)
                        for result in results
                    ):
                        method_errors.append(
                            f"{query!r} run {run_number + 1} returned an invalid result"
                        )

                    ranking = [
                        item["paper_id"] for item in _paper_preview(results, top_k)
                    ]
                    if first_results is None:
                        first_results = results
                        first_ranking = ranking
                    elif ranking != first_ranking:
                        method_errors.append(
                            f"{query!r} produced a non-deterministic paper ranking"
                        )

                first_results = first_results or []
                first_ranking = first_ranking or []
                if not first_results:
                    method_errors.append(f"{query!r} returned no results")
                rankings[method][query] = first_ranking
                per_query.append(
                    {
                        "query": query,
                        "chunk_result_count": len(first_results),
                        "unique_paper_count": len(first_ranking),
                        "latency": _latency_summary(query_latencies),
                        "top_results": _paper_preview(first_results, min(top_k, 3)),
                    }
                )
        except Exception as exc:
            warmup_ms = 0.0
            method_errors.append(f"{type(exc).__name__}: {exc}")

        method_reports[method] = {
            "passed": not method_errors,
            "warmup_ms": round(warmup_ms, 3),
            "query_count": len(normalized_queries),
            "runs_per_query": runs,
            "measured_searches": len(latencies),
            "latency": _latency_summary(latencies),
            "errors": method_errors,
            "per_query": per_query,
        }

    overlap: list[dict[str, Any]] = []
    pairs = (("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid"))
    for query in normalized_queries:
        row: dict[str, Any] = {"query": query}
        for left, right in pairs:
            left_ids = set(rankings[left].get(query, []))
            right_ids = set(rankings[right].get(query, []))
            union = left_ids | right_ids
            row[f"{left}_{right}_jaccard"] = round(
                len(left_ids & right_ids) / len(union), 4
            ) if union else 0.0
        overlap.append(row)

    return {
        "passed": all(report["passed"] for report in method_reports.values()),
        "quality_metrics_computed": False,
        "top_k": top_k,
        "alpha": alpha,
        "methods": method_reports,
        "cross_method_overlap": overlap,
    }


def _restore_environment(name: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


def run_api_smoke(
    *,
    query: str,
    db_path: str | Path,
    index_dir: str | Path,
    top_k: int = 5,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Exercise FastAPI handlers and response models with a mock LLM."""
    db_path = Path(db_path).resolve()
    index_dir = Path(index_dir).resolve()
    old_db = os.environ.get("CITEQUEST_DB_PATH")
    old_faiss = os.environ.get("CITEQUEST_FAISS_DIR")
    old_api_key = os.environ.get("LLM_API_KEY")
    os.environ["CITEQUEST_DB_PATH"] = str(db_path)
    os.environ["CITEQUEST_FAISS_DIR"] = str(index_dir)
    os.environ["LLM_API_KEY"] = ""

    endpoints: dict[str, Any] = {}
    errors: list[str] = []
    try:
        from app.api.routes_ask import ask_question
        from app.api.routes_search import search_papers
        from app.core.schemas import AskRequest, SearchRequest
        from app.main import health
        from app.rag.llm_provider import MockLLMProvider

        health_body = health()
        health_passed = (
            health_body.get("status") == "healthy"
            and health_body.get("capabilities", {}).get("hybrid_search") is True
        )
        endpoints["health"] = {
            "passed": health_passed,
            "service_status": health_body.get("status"),
            "capabilities": health_body.get("capabilities"),
        }
        if not health_passed:
            errors.append("health handler did not report a healthy hybrid-search service")

        hybrid_results: list[dict[str, Any]] = []
        for mode in ("lexical", "vector", "hybrid"):
            response = search_papers(
                SearchRequest(
                    query=query,
                    top_k=top_k,
                    mode=mode,
                    alpha=alpha,
                    include_overview=False,
                )
            )
            body = response.model_dump(mode="json")
            result_count = len(body["results"])
            passed = body["mode"] == mode and result_count > 0
            endpoints[f"search_{mode}"] = {
                "passed": passed,
                "result_count": result_count,
                "latency_ms": body["latency_ms"],
            }
            if not passed:
                errors.append(f"search handler mode={mode} returned no results")
            if mode == "hybrid":
                hybrid_results = body["results"]

        if hybrid_results:
            rag_top_k = min(top_k, len(hybrid_results), DEFAULT_RAG_TOP_K)
            mock = MockLLMProvider(
                "The retrieved evidence supports this demo answer [1]."
            )
            with patch("app.rag.answer.create_provider", return_value=mock):
                response = ask_question(
                    AskRequest(
                        question="What do the retrieved papers study?",
                        top_k=rag_top_k,
                        retrieval_mode="hybrid",
                        alpha=alpha,
                        pre_retrieved=hybrid_results[:rag_top_k],
                    )
                )
            body = response.model_dump(mode="json")
            citation_count = len(body["citations"])
            passed = (
                bool(body["answer"])
                and citation_count > 0
                and body["citation_valid"] is True
            )
            endpoints["ask_mock_rag"] = {
                "passed": passed,
                "citation_count": citation_count,
                "citation_valid": body["citation_valid"],
                "latency_ms": body["latency_ms"],
                "external_llm_called": False,
            }
            if not passed:
                errors.append("ask handler mock-RAG citation smoke failed")
        else:
            endpoints["ask_mock_rag"] = {
                "passed": False,
                "skipped": True,
                "external_llm_called": False,
            }
            errors.append("ask handler could not run because hybrid search returned no evidence")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        _restore_environment("CITEQUEST_DB_PATH", old_db)
        _restore_environment("CITEQUEST_FAISS_DIR", old_faiss)
        _restore_environment("LLM_API_KEY", old_api_key)

    return {
        "passed": not errors and all(
            endpoint.get("passed") for endpoint in endpoints.values()
        ),
        "transport": "direct_fastapi_route_handlers",
        "llm_mode": "forced_mock",
        "external_llm_called": False,
        "endpoints": endpoints,
        "errors": errors,
    }


def run_demo_smoke(
    *,
    db_path: str | Path,
    index_dir: str | Path,
    raw_path: str | Path | None,
    queries: Sequence[str] = DEFAULT_DEMO_QUERIES,
    expected_papers: int | None = 10_000,
    top_k: int = 10,
    runs: int = 3,
    alpha: float = 0.5,
    include_api: bool = True,
    retriever_factory: RetrieverFactory | None = None,
) -> dict[str, Any]:
    """Run the complete demo acceptance workflow and return a JSON-safe report."""
    db_path = Path(db_path).resolve()
    index_dir = Path(index_dir).resolve()
    raw_path = Path(raw_path).resolve() if raw_path is not None else None
    created_at = _utc_now()
    artifacts = validate_demo_artifacts(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=raw_path,
        expected_papers=expected_papers,
    )

    if artifacts["passed"]:
        retrieval = run_retrieval_smoke(
            queries=queries,
            db_path=db_path,
            index_dir=index_dir,
            top_k=top_k,
            runs=runs,
            alpha=alpha,
            retriever_factory=retriever_factory,
        )
        api = (
            run_api_smoke(
                query=queries[0],
                db_path=db_path,
                index_dir=index_dir,
                top_k=min(top_k, 5),
                alpha=alpha,
            )
            if include_api
            else {"passed": True, "skipped": True, "reason": "disabled by caller"}
        )
    else:
        retrieval = {
            "passed": False,
            "skipped": True,
            "reason": "artifact validation failed",
            "quality_metrics_computed": False,
        }
        api = {
            "passed": False,
            "skipped": True,
            "reason": "artifact validation failed",
        }

    git_status = _git_value("status", "--porcelain")
    passed = artifacts["passed"] and retrieval["passed"] and api["passed"]
    return {
        "schema_version": 1,
        "report_type": "demo_smoke",
        "status": "passed" if passed else "failed",
        "official_benchmark": False,
        "quality_metrics_computed": False,
        "created_at": created_at,
        "disclaimer": (
            "Operational demo smoke only. The corpus has no frozen relevance judgments, "
            "so this report must not be presented as Retrieval Benchmark v1."
        ),
        "configuration": {
            "database_file": _display_path(db_path),
            "index_dir": _display_path(index_dir),
            "raw_file": _display_path(raw_path) if raw_path else None,
            "expected_papers": expected_papers,
            "queries": list(queries),
            "top_k": top_k,
            "runs_per_query": runs,
            "hybrid_alpha": alpha,
            "api_smoke_enabled": include_api,
        },
        "provenance": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "accelerator": collect_accelerator_info(),
        },
        "artifacts": artifacts,
        "retrieval": retrieval,
        "api": api,
    }


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_demo_smoke_markdown(report: dict[str, Any]) -> str:
    """Render a compact, recruiter-readable operational smoke report."""
    accelerator = report["provenance"].get("accelerator", {})
    device = accelerator.get("device_name") or "CPU"
    lines = [
        "# CiteQuest 10k Demo Smoke Report",
        "",
        "> Operational smoke only; this is not Retrieval Benchmark v1 and contains no relevance metrics.",
        "",
        f"**Status:** `{report['status']}`  ",
        f"**Created:** `{report['created_at']}`  ",
        f"**Git commit:** `{report['provenance'].get('git_commit') or 'unknown'}`  ",
        f"**Git dirty:** `{report['provenance']['git_dirty']}`",
        f"**Accelerator:** `{device}`",
        "",
        "## Artifact Integrity",
        "",
    ]
    facts = report["artifacts"].get("facts", {})
    lines.extend(
        [
            "| Fact | Value |",
            "|---|---:|",
            f"| Papers | {facts.get('paper_count', 'n/a')} |",
            f"| Chunks | {facts.get('chunk_count', 'n/a')} |",
            f"| FTS rows | {facts.get('fts_row_count', 'n/a')} |",
            f"| FAISS vectors | {facts.get('faiss_vector_count', 'n/a')} |",
            f"| ID-map entries | {facts.get('id_map_count', 'n/a')} |",
            f"| Embedding dimensions | {facts.get('embedding_dim', 'n/a')} |",
            f"| FAISS index | {_markdown_escape(facts.get('faiss_index_type', 'n/a'))} |",
            "",
            "| Check | Result | Detail |",
            "|---|---|---|",
        ]
    )
    for check in report["artifacts"].get("checks", []):
        lines.append(
            f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | "
            f"{_markdown_escape(check['detail'])} |"
        )

    retrieval = report["retrieval"]
    lines.extend(["", "## Warm Retrieval Latency", ""])
    if retrieval.get("skipped"):
        lines.append(f"Skipped: {_markdown_escape(retrieval.get('reason', 'unknown'))}")
    else:
        lines.extend(
            [
                "| Method | Searches | Mean (ms) | p50 (ms) | p95 (ms) | Result |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for method in METHODS:
            method_report = retrieval["methods"][method]
            latency = method_report["latency"]
            lines.append(
                f"| {method} | {method_report['measured_searches']} | "
                f"{latency['mean_ms']:.3f} | {latency['p50_ms']:.3f} | "
                f"{latency['p95_ms']:.3f} | "
                f"{'PASS' if method_report['passed'] else 'FAIL'} |"
            )
        lines.extend(
            [
                "",
                "Latency excludes one recorded warm-up query per method. Values are operational timings, not quality metrics.",
                "",
                "## Representative Results",
                "",
                "| Query | Method | Unique papers | Top paper |",
                "|---|---|---:|---|",
            ]
        )
        for method in METHODS:
            for query_report in retrieval["methods"][method]["per_query"]:
                top_results = query_report["top_results"]
                top_title = top_results[0]["title"] if top_results else "none"
                lines.append(
                    f"| {_markdown_escape(query_report['query'])} | {method} | "
                    f"{query_report['unique_paper_count']} | {_markdown_escape(top_title)} |"
                )

    api = report["api"]
    lines.extend(["", "## API And RAG Smoke", ""])
    if api.get("skipped"):
        lines.append(f"Skipped: {_markdown_escape(api.get('reason', 'disabled'))}")
    else:
        lines.extend(["| Check | Result |", "|---|---|"])
        for name, endpoint in api.get("endpoints", {}).items():
            lines.append(
                f"| `{name}` | {'PASS' if endpoint.get('passed') else 'FAIL'} |"
            )
        lines.extend(
            [
                "",
                "These checks call the FastAPI route handlers through their Pydantic contracts; they do not open a network socket.",
                "The RAG check forces `MockLLMProvider`; no external LLM request is made.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Passing proves that the local artifacts agree and the BM25, Dense, Hybrid, API, and citation paths execute.",
            "- It does not prove retrieval relevance because this demo corpus has no frozen judgments.",
            "- Quality comparisons belong only in the separate 50k arXiv CS Retrieval Benchmark v1 report.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_demo_smoke_outputs(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Atomically write machine-readable and human-readable smoke reports."""
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    _atomic_write_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(markdown_path, render_demo_smoke_markdown(report))
