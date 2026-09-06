"""Measure FAISS IVF approximation error without changing embeddings.

The experiment encodes each frozen dev query once, then searches the same
vector with the production IVF ``nprobe`` and with every IVF list enabled.
The all-list condition is exhaustive over the vectors stored in IndexIVFFlat,
so differences isolate ANN search approximation from representation quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.eval.fusion_experiment import (
    evaluate_dev_artifact_gate,
    load_selected_dev_queries,
)
from app.eval.retrieval_eval import (
    DEFAULT_DB as DEFAULT_BENCHMARK_DB,
    DEFAULT_EVAL as DEFAULT_EVAL_PATH,
    DEFAULT_INDEX_DIR as DEFAULT_BENCHMARK_INDEX_DIR,
    OFFICIAL_DEV_COUNT,
    QUERY_TYPES,
    EvalQuery,
    build_benchmark_manifest,
    hit_rate_at_k,
    ndcg_at_k,
    reciprocal_rank_at_k,
)
from app.retrieval.embeddings import EmbeddingModel


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "arxiv_cs_benchmark_v1_50000.jsonl"
DEFAULT_REFERENCE_REPORT = PROJECT_ROOT / "reports" / "retrieval_diagnostics_v1_3.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "ann_probe_v1_3_1.json"
DEFAULT_MD_REPORT = PROJECT_ROOT / "reports" / "ann_probe_v1_3_1.md"

DEFAULT_CURRENT_NPROBE = 64
DEFAULT_CANDIDATE_DEPTH = 20
DEFAULT_DEEP_PROBE_DEPTH = 1_000
FORMAL_STATUS = "ann_probe_v1_3_1"


@dataclass(frozen=True)
class AnnRanking:
    """One FAISS search result before any metadata hydration."""

    chunk_ids: tuple[str, ...]
    paper_ids: tuple[str, ...]
    scores: tuple[float, ...]
    latency_ms: float


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path, *, label: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label} at {path}: {exc}") from exc


def load_faiss_artifacts(
    index_dir: str | Path,
) -> tuple[Any, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Load and validate the independent index copy used by the experiment."""
    index_dir = Path(index_dir)
    index_path = index_dir / "index.faiss"
    id_map_path = index_dir / "id_map.json"
    build_meta_path = index_dir / "build_meta.json"
    for path in (index_path, id_map_path, build_meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"required FAISS artifact not found: {path}")

    import faiss

    index = faiss.read_index(str(index_path))
    id_map_raw = _read_json(id_map_path, label="FAISS ID map")
    build_meta = _read_json(build_meta_path, label="FAISS build metadata")
    if not isinstance(id_map_raw, list) or len(id_map_raw) != int(index.ntotal):
        raise RuntimeError("FAISS ID map count does not match the index")
    for faiss_id, entry in enumerate(id_map_raw):
        if (
            not isinstance(entry, dict)
            or entry.get("faiss_id") != faiss_id
            or not isinstance(entry.get("chunk_id"), str)
            or not isinstance(entry.get("paper_id"), str)
        ):
            raise RuntimeError(f"invalid FAISS ID map entry at position {faiss_id}")
    if not isinstance(build_meta, dict):
        raise RuntimeError("FAISS build metadata must be a JSON object")
    if build_meta.get("index_type") != "IndexIVFFlat":
        raise RuntimeError("ANN probe requires an IndexIVFFlat artifact")
    if not hasattr(index, "nlist") or not hasattr(index, "nprobe"):
        raise RuntimeError("loaded FAISS index does not expose IVF nprobe/nlist")
    if build_meta.get("nlist") != int(index.nlist):
        raise RuntimeError("FAISS nlist does not match build metadata")
    if build_meta.get("num_vectors") != int(index.ntotal):
        raise RuntimeError("FAISS vector count does not match build metadata")
    if build_meta.get("vector_dim") != int(index.d):
        raise RuntimeError("FAISS dimension does not match build metadata")
    return index, tuple(id_map_raw), build_meta


def load_diagnostics_reference(
    path: str | Path = DEFAULT_REFERENCE_REPORT,
) -> dict[str, Any]:
    """Load the formal v1.3 report that anchors production Dense candidates."""
    report = _read_json(Path(path), label="v1.3 diagnostics reference")
    if not isinstance(report, dict):
        raise ValueError("diagnostics reference must be a JSON object")
    if report.get("status") != "dev_diagnostics_v1_3":
        raise ValueError("diagnostics reference must be a formal v1.3 result")
    if report.get("query_count") != OFFICIAL_DEV_COUNT:
        raise ValueError(f"diagnostics reference must contain {OFFICIAL_DEV_COUNT} queries")
    if report.get("artifact_gate", {}).get("passed") is not True:
        raise ValueError("diagnostics reference artifact gate did not pass")
    if report.get("reference_reproduction", {}).get("matches") is not True:
        raise ValueError("diagnostics reference did not reproduce v1.2")
    protocol = report.get("protocol", {})
    if protocol.get("candidate_depth") != DEFAULT_CANDIDATE_DEPTH:
        raise ValueError("diagnostics reference candidate depth is not 20")
    per_query = report.get("per_query")
    if not isinstance(per_query, list) or len(per_query) != OFFICIAL_DEV_COUNT:
        raise ValueError("diagnostics reference has incomplete per-query rows")
    return report


def evaluate_ann_artifact_gate(
    manifest: dict[str, Any],
    reference: dict[str, Any],
    *,
    current_nprobe: int,
    exhaustive_nprobe: int,
) -> dict[str, Any]:
    """Require unchanged Benchmark v1 artifacts and the formal v1.3 anchor."""
    reference_manifest = reference.get("manifest")
    if not isinstance(reference_manifest, dict):
        raise ValueError("diagnostics reference has no valid manifest")
    base = evaluate_dev_artifact_gate(
        manifest,
        {
            "raw_file_sha256": reference_manifest.get("raw_file_sha256"),
            "eval_file_sha256": reference_manifest.get("eval_file_sha256"),
        },
    )
    checks = dict(base["checks"])
    checks.update(
        {
            "reference_status_is_formal_v1_3": (
                reference.get("status") == "dev_diagnostics_v1_3"
            ),
            "reference_artifact_gate_passed": (
                reference.get("artifact_gate", {}).get("passed") is True
            ),
            "reference_v1_2_reproduction_matched": (
                reference.get("reference_reproduction", {}).get("matches") is True
            ),
            "index_is_ivf_flat": manifest.get("faiss_index_type") == "IndexIVFFlat",
            "current_nprobe_is_frozen_64": current_nprobe == DEFAULT_CURRENT_NPROBE,
            "current_nprobe_matches_manifest": (
                current_nprobe == manifest.get("faiss_nprobe")
            ),
            "exhaustive_nprobe_equals_nlist": (
                exhaustive_nprobe == manifest.get("faiss_nlist")
            ),
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {"passed": not failures, "checks": checks, "failures": failures}


def encode_queries_once(
    queries: Sequence[EvalQuery],
    *,
    model_name: str,
    expected_dimension: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode each query once in production-compatible single-query calls."""
    if not queries:
        raise ValueError("queries must not be empty")
    model = EmbeddingModel(model_name=model_name)
    if int(model.dim) != expected_dimension:
        raise RuntimeError(
            f"embedding dimension {model.dim} does not match index {expected_dimension}"
        )

    warm_started = time.perf_counter()
    warm = model.encode([queries[0].query], show_progress=False)
    warm_latency = (time.perf_counter() - warm_started) * 1_000
    if warm.shape != (1, expected_dimension):
        raise RuntimeError(f"warm-up embedding has unexpected shape {warm.shape}")

    vectors: list[np.ndarray] = []
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        vector = np.asarray(
            model.encode([query.query], show_progress=False),
            dtype=np.float32,
        )
        latencies.append((time.perf_counter() - started) * 1_000)
        if vector.shape != (1, expected_dimension):
            raise RuntimeError(
                f"query {query.query_id} embedding has unexpected shape {vector.shape}"
            )
        vectors.append(vector[0])
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    return matrix, {
        "model": model_name,
        "dimension": expected_dimension,
        "query_count": len(queries),
        "warmup_latency_ms": round(warm_latency, 3),
        "warmup_excluded": True,
        "mean_ms": round(float(np.mean(latencies)), 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "same_vector_reused_across_conditions": True,
    }


def _search_once(
    index: Any,
    id_map: Sequence[dict[str, Any]],
    query_vector: np.ndarray,
    *,
    nprobe: int,
    top_k: int,
) -> AnnRanking:
    if nprobe <= 0 or nprobe > int(index.nlist):
        raise ValueError(f"nprobe must be between 1 and {index.nlist}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    if vector.shape[1] != int(index.d):
        raise ValueError("query vector dimension does not match the index")
    index.nprobe = int(nprobe)
    started = time.perf_counter()
    scores, faiss_ids = index.search(vector, top_k)
    latency_ms = (time.perf_counter() - started) * 1_000

    chunk_ids: list[str] = []
    paper_ids: list[str] = []
    valid_scores: list[float] = []
    for score, faiss_id in zip(scores[0], faiss_ids[0]):
        fid = int(faiss_id)
        if fid < 0 or fid >= len(id_map):
            continue
        entry = id_map[fid]
        chunk_ids.append(entry["chunk_id"])
        paper_ids.append(entry["paper_id"])
        valid_scores.append(float(score))
    return AnnRanking(
        chunk_ids=tuple(chunk_ids),
        paper_ids=tuple(paper_ids),
        scores=tuple(valid_scores),
        latency_ms=latency_ms,
    )


def _first_relevant_rank(
    paper_ids: Sequence[str], relevant_paper_ids: set[str]
) -> int | None:
    return next(
        (
            rank
            for rank, paper_id in enumerate(paper_ids, start=1)
            if paper_id in relevant_paper_ids
        ),
        None,
    )


def _ranking_payload(
    ranking: AnnRanking,
    relevant: set[str],
    *,
    deep_rank: int | None,
) -> dict[str, Any]:
    return {
        "chunk_ids": list(ranking.chunk_ids),
        "paper_ids": list(ranking.paper_ids),
        "scores": list(ranking.scores),
        "first_relevant_rank_at_candidate_depth": _first_relevant_rank(
            ranking.paper_ids, relevant
        ),
        "first_relevant_rank_at_deep_probe": deep_rank,
    }


def _condition_metrics(
    rows: Sequence[dict[str, Any]], condition: str
) -> dict[str, Any]:
    if not rows:
        return {
            "query_count": 0,
            "hit_rate@5": 0.0,
            "hit_rate@10": 0.0,
            "hit_rate@20": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
        }
    return {
        "query_count": len(rows),
        "hit_rate@5": float(
            np.mean(
                [
                    hit_rate_at_k(
                        row[condition]["paper_ids"],
                        set(row["relevant_paper_ids"]),
                        5,
                    )
                    for row in rows
                ]
            )
        ),
        "hit_rate@10": float(
            np.mean(
                [
                    hit_rate_at_k(
                        row[condition]["paper_ids"],
                        set(row["relevant_paper_ids"]),
                        10,
                    )
                    for row in rows
                ]
            )
        ),
        "hit_rate@20": float(
            np.mean(
                [
                    hit_rate_at_k(
                        row[condition]["paper_ids"],
                        set(row["relevant_paper_ids"]),
                        20,
                    )
                    for row in rows
                ]
            )
        ),
        "mrr@10": float(
            np.mean(
                [
                    reciprocal_rank_at_k(
                        row[condition]["paper_ids"],
                        set(row["relevant_paper_ids"]),
                        10,
                    )
                    for row in rows
                ]
            )
        ),
        "ndcg@10": float(
            np.mean(
                [
                    ndcg_at_k(
                        row[condition]["paper_ids"],
                        set(row["relevant_paper_ids"]),
                        10,
                    )
                    for row in rows
                ]
            )
        ),
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": round(float(np.mean(values)), 3) if values else 0.0,
        "p50_ms": round(float(np.percentile(values, 50)), 3) if values else 0.0,
        "p95_ms": round(float(np.percentile(values, 95)), 3) if values else 0.0,
    }


def compare_current_candidates_with_reference(
    rows: Sequence[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, Any]:
    reference_rows = reference.get("per_query")
    if not isinstance(reference_rows, list):
        raise ValueError("diagnostics reference has no per-query rows")
    by_id = {row.get("query_id"): row for row in reference_rows}
    failures: list[str] = []
    for row in rows:
        query_id = row["query_id"]
        expected = by_id.get(query_id)
        if expected is None:
            failures.append(f"{query_id}: missing reference row")
            continue
        expected_ids = expected.get("candidate_ids", {}).get("dense_chunk_ids")
        if row["current"]["chunk_ids"] != expected_ids:
            failures.append(f"{query_id}: current Dense candidate IDs differ")
        if row["current"]["first_relevant_rank_at_candidate_depth"] != expected.get(
            "dense_first_relevant_rank"
        ):
            failures.append(f"{query_id}: current Dense relevant rank differs")
    return {
        "status": "matched" if not failures else "mismatch",
        "matches": not failures,
        "query_count": len(rows),
        "failures": failures,
    }


def run_ann_probe_comparison(
    queries: Sequence[EvalQuery],
    query_vectors: np.ndarray,
    index: Any,
    id_map: Sequence[dict[str, Any]],
    *,
    reference: dict[str, Any],
    current_nprobe: int = DEFAULT_CURRENT_NPROBE,
    exhaustive_nprobe: int | None = None,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    deep_probe_depth: int = DEFAULT_DEEP_PROBE_DEPTH,
) -> dict[str, Any]:
    """Compare two nprobe settings with identical query vectors."""
    if not queries:
        raise ValueError("queries must not be empty")
    unexpected = sorted({query.split for query in queries if query.split != "dev"})
    if unexpected:
        raise ValueError(
            "ANN probe expected only 'dev' queries; found " + ", ".join(unexpected)
        )
    vectors = np.asarray(query_vectors, dtype=np.float32)
    if vectors.shape != (len(queries), int(index.d)):
        raise ValueError(
            f"query_vectors shape {vectors.shape} does not match "
            f"({len(queries)}, {index.d})"
        )
    exhaustive_nprobe = (
        int(index.nlist) if exhaustive_nprobe is None else exhaustive_nprobe
    )
    if not 0 < current_nprobe <= int(index.nlist):
        raise ValueError("current_nprobe is outside the IVF range")
    if exhaustive_nprobe != int(index.nlist):
        raise ValueError("exhaustive_nprobe must equal index.nlist")
    if candidate_depth <= 0:
        raise ValueError("candidate_depth must be positive")
    if deep_probe_depth < candidate_depth:
        raise ValueError("deep_probe_depth must be at least candidate_depth")

    warmups = {
        "current": _search_once(
            index,
            id_map,
            vectors[0],
            nprobe=current_nprobe,
            top_k=candidate_depth,
        ).latency_ms,
        "exhaustive": _search_once(
            index,
            id_map,
            vectors[0],
            nprobe=exhaustive_nprobe,
            top_k=candidate_depth,
        ).latency_ms,
    }

    top_rankings: dict[str, list[AnnRanking | None]] = {
        "current": [None] * len(queries),
        "exhaustive": [None] * len(queries),
    }
    latencies: dict[str, list[float]] = {"current": [], "exhaustive": []}
    conditions = {
        "current": current_nprobe,
        "exhaustive": exhaustive_nprobe,
    }
    for query_index, vector in enumerate(vectors):
        order = (
            ("current", "exhaustive")
            if query_index % 2 == 0
            else ("exhaustive", "current")
        )
        for condition in order:
            ranking = _search_once(
                index,
                id_map,
                vector,
                nprobe=conditions[condition],
                top_k=candidate_depth,
            )
            top_rankings[condition][query_index] = ranking
            latencies[condition].append(ranking.latency_ms)

    rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    top_rank_wins = Counter()
    deep_rank_wins = Counter()
    for query_index, query in enumerate(queries):
        current = top_rankings["current"][query_index]
        exhaustive = top_rankings["exhaustive"][query_index]
        if current is None or exhaustive is None:
            raise RuntimeError("ANN condition did not produce a ranking")
        relevant = set(query.relevant_paper_ids)
        current_rank = _first_relevant_rank(current.paper_ids, relevant)
        exhaustive_rank = _first_relevant_rank(exhaustive.paper_ids, relevant)
        deep_ranks: dict[str, int | None] = {
            "current": current_rank,
            "exhaustive": exhaustive_rank,
        }
        if current_rank is None:
            ranking = _search_once(
                index,
                id_map,
                vectors[query_index],
                nprobe=current_nprobe,
                top_k=deep_probe_depth,
            )
            deep_ranks["current"] = _first_relevant_rank(
                ranking.paper_ids, relevant
            )
        if exhaustive_rank is None:
            ranking = _search_once(
                index,
                id_map,
                vectors[query_index],
                nprobe=exhaustive_nprobe,
                top_k=deep_probe_depth,
            )
            deep_ranks["exhaustive"] = _first_relevant_rank(
                ranking.paper_ids, relevant
            )

        current_value = current_rank if current_rank is not None else math.inf
        exhaustive_value = exhaustive_rank if exhaustive_rank is not None else math.inf
        top_winner = "tie"
        if current_value < exhaustive_value:
            top_winner = "current"
        elif exhaustive_value < current_value:
            top_winner = "exhaustive"
        top_rank_wins[top_winner] += 1

        current_deep_value = (
            deep_ranks["current"]
            if deep_ranks["current"] is not None
            else math.inf
        )
        exhaustive_deep_value = (
            deep_ranks["exhaustive"]
            if deep_ranks["exhaustive"] is not None
            else math.inf
        )
        deep_winner = "tie"
        if current_deep_value < exhaustive_deep_value:
            deep_winner = "current"
        elif exhaustive_deep_value < current_deep_value:
            deep_winner = "exhaustive"
        deep_rank_wins[deep_winner] += 1

        current_set = set(current.chunk_ids)
        exhaustive_set = set(exhaustive.chunk_ids)
        overlap = current_set & exhaustive_set
        union = current_set | exhaustive_set
        overlap_row = {
            "query_id": query.query_id,
            "overlap_count": len(overlap),
            "union_count": len(union),
            "jaccard": len(overlap) / len(union) if union else 1.0,
            "current_recall_of_exhaustive": (
                len(overlap) / len(exhaustive_set) if exhaustive_set else 1.0
            ),
        }
        overlap_rows.append(overlap_row)
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "query_type": query.query_type,
                "source_category": query.source_category,
                "relevant_paper_ids": list(query.relevant_paper_ids),
                "current": _ranking_payload(
                    current,
                    relevant,
                    deep_rank=deep_ranks["current"],
                ),
                "exhaustive": _ranking_payload(
                    exhaustive,
                    relevant,
                    deep_rank=deep_ranks["exhaustive"],
                ),
                "top_rank_winner": top_winner,
                "deep_rank_winner": deep_winner,
                "candidate_overlap": overlap_row,
            }
        )

    methods = {
        condition: {
            "nprobe": nprobe,
            "metrics": _condition_metrics(rows, condition),
            "by_query_type": {
                query_type: _condition_metrics(
                    [row for row in rows if row["query_type"] == query_type],
                    condition,
                )
                for query_type in QUERY_TYPES
                if any(row["query_type"] == query_type for row in rows)
            },
            "search_latency_top20": _latency_summary(latencies[condition]),
        }
        for condition, nprobe in conditions.items()
    }
    current_metrics = methods["current"]["metrics"]
    exhaustive_metrics = methods["exhaustive"]["metrics"]
    metric_deltas = {
        name: float(exhaustive_metrics[name] - current_metrics[name])
        for name in ("hit_rate@5", "hit_rate@10", "hit_rate@20", "mrr@10", "ndcg@10")
    }
    target_recovered = sum(
        row["current"]["first_relevant_rank_at_candidate_depth"] is None
        and row["exhaustive"]["first_relevant_rank_at_candidate_depth"] is not None
        for row in rows
    )
    target_lost = sum(
        row["current"]["first_relevant_rank_at_candidate_depth"] is not None
        and row["exhaustive"]["first_relevant_rank_at_candidate_depth"] is None
        for row in rows
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_or_development",
        "query_count": len(queries),
        "protocol": {
            "query_split": "dev",
            "candidate_depth": candidate_depth,
            "deep_probe_depth": deep_probe_depth,
            "current_nprobe": current_nprobe,
            "exhaustive_nprobe": exhaustive_nprobe,
            "nlist": int(index.nlist),
            "comparison_control": "same encoded query vector for both conditions",
            "search_order": "alternated by query to reduce order bias",
            "latency_scope": "FAISS search only at candidate depth; encoding excluded",
            "deep_probe_usage": "target misses only; excluded from metrics and latency",
        },
        "warmup": {
            "current_latency_ms": round(warmups["current"], 3),
            "exhaustive_latency_ms": round(warmups["exhaustive"], 3),
            "excluded_from_measured_latency": True,
        },
        "methods": methods,
        "metric_deltas_exhaustive_minus_current": metric_deltas,
        "candidate_overlap": {
            "mean_jaccard": float(np.mean([row["jaccard"] for row in overlap_rows])),
            "mean_current_recall_of_exhaustive": float(
                np.mean([row["current_recall_of_exhaustive"] for row in overlap_rows])
            ),
            "per_query": overlap_rows,
        },
        "target_rank_comparison": {
            "top20": {
                "current_wins": top_rank_wins["current"],
                "exhaustive_wins": top_rank_wins["exhaustive"],
                "ties": top_rank_wins["tie"],
                "targets_recovered_by_exhaustive": target_recovered,
                "targets_lost_by_exhaustive": target_lost,
            },
            "deep_probe": {
                "current_wins": deep_rank_wins["current"],
                "exhaustive_wins": deep_rank_wins["exhaustive"],
                "ties": deep_rank_wins["tie"],
            },
        },
        "reference_reproduction": compare_current_candidates_with_reference(
            rows, reference
        ),
        "per_query": rows,
    }


def _formal_protocol_matches(report: dict[str, Any]) -> bool:
    protocol = report.get("protocol", {})
    return (
        protocol.get("query_split") == "dev"
        and protocol.get("candidate_depth") == DEFAULT_CANDIDATE_DEPTH
        and protocol.get("deep_probe_depth") == DEFAULT_DEEP_PROBE_DEPTH
        and protocol.get("current_nprobe") == DEFAULT_CURRENT_NPROBE
        and protocol.get("exhaustive_nprobe") == protocol.get("nlist")
    )


def finalize_ann_probe_report(
    report: dict[str, Any],
    *,
    manifest: dict[str, Any],
    reference: dict[str, Any],
    reference_path: str | Path,
    artifact_gate: dict[str, Any],
    encoding: dict[str, Any],
    source_dev_count: int,
    formal_run: bool,
) -> dict[str, Any]:
    """Attach provenance, formal status, and a predeclared next-step decision."""
    report["manifest"] = manifest
    report["reference_report"] = {
        "path": _display_path(reference_path),
        "status": reference.get("status"),
        "git_commit": reference.get("manifest", {}).get("git_commit"),
    }
    report["artifact_gate"] = artifact_gate
    report["encoding"] = encoding
    report["query_selection"] = {
        "split": "dev",
        "source_dev_count": source_dev_count,
        "selected_query_count": report["query_count"],
        "selection": (
            "all_frozen_dev_queries"
            if formal_run
            else "first_n_dev_queries_in_frozen_file_order"
        ),
    }
    if not formal_run:
        report["status"] = "smoke_or_development"
    elif (
        report["query_count"] == OFFICIAL_DEV_COUNT
        and source_dev_count == OFFICIAL_DEV_COUNT
        and _formal_protocol_matches(report)
        and artifact_gate.get("passed") is True
        and report["reference_reproduction"]["matches"] is True
    ):
        report["status"] = FORMAL_STATUS
    else:
        report["status"] = "invalid_ann_probe"

    top20 = report["target_rank_comparison"]["top20"]
    applicable = report["status"] == FORMAL_STATUS
    if not applicable:
        recommendation = "no_decision_from_smoke_or_invalid_run"
    elif top20["targets_recovered_by_exhaustive"] > 0:
        recommendation = "evaluate_nprobe_quality_latency_tradeoff"
    else:
        recommendation = "retain_nprobe_64_and_test_representation"
    report["decision"] = {
        "applicable": applicable,
        "rule": (
            "Run an nprobe quality/latency sweep only if exhaustive IVF recovers "
            "at least one known target into Dense Top-20; otherwise retain 64 "
            "and investigate query/document representation."
        ),
        "recommendation": recommendation,
        "production_changed": False,
    }
    report["limitations"] = [
        (
            "The experiment isolates search approximation only; it does not "
            "test another embedding model."
        ),
        "Known-item labels cannot identify other papers that may be relevant to the query.",
        "Top-1000 ranks are diagnostic and are excluded from candidate metrics and latency.",
        "The 50-query dev split is reused for diagnosis and is not a fresh holdout.",
        "Exhaustive IVF latency is a diagnostic upper bound, not a proposed production setting.",
    ]
    return report


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _rank(value: int | None) -> str:
    return str(value) if value is not None else "not found"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_ann_probe_markdown(report: dict[str, Any]) -> str:
    """Render an answer-first technical report."""
    current = report["methods"]["current"]
    top20 = report["target_rank_comparison"]["top20"]
    lines = [
        "# CiteQuest ANN Approximation Probe v1.3.1",
        "",
        f"Status: **`{report['status']}`**  ",
        f"Queries: **{report['query_count']}** frozen dev records",
        "",
        "## Conclusion",
        "",
        (
            f"Exhaustive IVF recovered {top20['targets_recovered_by_exhaustive']} "
            f"known targets into Dense Top-20 and lost "
            f"{top20['targets_lost_by_exhaustive']} relative to production "
            f"nprobe={current['nprobe']}."
        ),
        "",
        f"Recommendation: **`{report['decision']['recommendation']}`**.",
        "",
        "No production configuration was changed.",
        "",
        "## Dense Quality",
        "",
        "| Condition | nprobe | HitRate@5 | HitRate@10 | HitRate@20 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("current", "Production"), ("exhaustive", "All IVF lists")):
        method = report["methods"][name]
        metrics = method["metrics"]
        lines.append(
            f"| {label} | {method['nprobe']} | {_fmt(metrics['hit_rate@5'])} "
            f"| {_fmt(metrics['hit_rate@10'])} | {_fmt(metrics['hit_rate@20'])} "
            f"| {_fmt(metrics['mrr@10'])} | {_fmt(metrics['ndcg@10'])} |"
        )

    overlap = report["candidate_overlap"]
    lines.extend(
        [
            "",
            "## Approximation Agreement",
            "",
            f"- Mean Top-20 Jaccard: `{overlap['mean_jaccard']:.4f}`",
            (
                "- Mean production recall of exhaustive Top-20: "
                f"`{overlap['mean_current_recall_of_exhaustive']:.4f}`"
            ),
            f"- Exhaustive Top-20 wins: `{top20['exhaustive_wins']}`",
            f"- Production Top-20 wins: `{top20['current_wins']}`",
            f"- Target-rank ties: `{top20['ties']}`",
            "",
            "## Target Misses and Rank Changes",
            "",
            (
                "| ID | Type | Production Top-20 | Exhaustive Top-20 | "
                "Production deep | Exhaustive deep | Query |"
            ),
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["per_query"]:
        current_rank = row["current"]["first_relevant_rank_at_candidate_depth"]
        exhaustive_rank = row["exhaustive"]["first_relevant_rank_at_candidate_depth"]
        if current_rank == exhaustive_rank and current_rank is not None:
            continue
        lines.append(
            f"| {row['query_id']} | {_escape(row['query_type'])} "
            f"| {_rank(current_rank)} | {_rank(exhaustive_rank)} "
            f"| {_rank(row['current']['first_relevant_rank_at_deep_probe'])} "
            f"| {_rank(row['exhaustive']['first_relevant_rank_at_deep_probe'])} "
            f"| {_escape(row['query'])} |"
        )

    lines.extend(
        [
            "",
            "## FAISS Search Latency",
            "",
            "Encoding and metadata hydration are excluded. Warm-up is excluded.",
            "",
            "| Condition | Mean ms | p50 ms | p95 ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, label in (("current", "Production"), ("exhaustive", "All IVF lists")):
        latency = report["methods"][name]["search_latency_top20"]
        lines.append(
            f"| {label} | {latency['mean_ms']:.3f} | {latency['p50_ms']:.3f} "
            f"| {latency['p95_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Evidence Gates",
            "",
            f"- Artifact gate: `{'pass' if report['artifact_gate']['passed'] else 'fail'}`",
            (
                "- Production Dense candidate reproduction: "
                f"`{'pass' if report['reference_reproduction']['matches'] else 'fail'}`"
            ),
            (
                "- Same query vector reused: "
                f"`{report['encoding']['same_vector_reused_across_conditions']}`"
            ),
            f"- Git commit: `{report['manifest'].get('git_commit')}`",
            f"- Git worktree clean: `{not report['manifest'].get('git_dirty')}`",
            "",
            "## Decision Rule",
            "",
            report["decision"]["rule"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def write_ann_probe_outputs(
    report: dict[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_path: str | Path = DEFAULT_MD_REPORT,
) -> None:
    _atomic_write_text(
        Path(json_path),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(Path(markdown_path), render_ann_probe_markdown(report))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare production IVF nprobe with exhaustive all-list search"
    )
    parser.add_argument("--eval", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--db", default=str(DEFAULT_BENCHMARK_DB))
    parser.add_argument("--index-dir", default=str(DEFAULT_BENCHMARK_INDEX_DIR))
    parser.add_argument("--raw", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--reference-report", default=str(DEFAULT_REFERENCE_REPORT))
    parser.add_argument(
        "--limit",
        type=int,
        help="Run the first N dev queries as smoke; omitted means formal 50-query run",
    )
    parser.add_argument(
        "--current-nprobe", type=int, default=DEFAULT_CURRENT_NPROBE
    )
    parser.add_argument(
        "--candidate-depth", type=int, default=DEFAULT_CANDIDATE_DEPTH
    )
    parser.add_argument(
        "--deep-probe-depth", type=int, default=DEFAULT_DEEP_PROBE_DEPTH
    )
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_REPORT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD_REPORT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    formal_run = args.limit is None
    reference = load_diagnostics_reference(args.reference_report)
    db_path = Path(args.db)
    index_dir = Path(args.index_dir)
    queries, source_dev_count = load_selected_dev_queries(
        args.eval, db_path, limit=args.limit
    )
    manifest = build_benchmark_manifest(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=args.raw,
        eval_path=args.eval,
        corpus="arxiv_cs",
    )
    index, id_map, build_meta = load_faiss_artifacts(index_dir)
    exhaustive_nprobe = int(index.nlist)
    artifact_gate = evaluate_ann_artifact_gate(
        manifest,
        reference,
        current_nprobe=args.current_nprobe,
        exhaustive_nprobe=exhaustive_nprobe,
    )
    frozen_parameters = (
        args.current_nprobe == DEFAULT_CURRENT_NPROBE
        and args.candidate_depth == DEFAULT_CANDIDATE_DEPTH
        and args.deep_probe_depth == DEFAULT_DEEP_PROBE_DEPTH
    )
    if formal_run and not frozen_parameters:
        raise ValueError("formal ANN probe requires the frozen v1.3.1 parameters")
    if formal_run and not artifact_gate["passed"]:
        raise RuntimeError(
            "formal ANN probe artifact preflight failed: "
            + ", ".join(artifact_gate["failures"])
        )

    vectors, encoding = encode_queries_once(
        queries,
        model_name=build_meta["embedding_model"],
        expected_dimension=int(index.d),
    )
    report = run_ann_probe_comparison(
        queries,
        vectors,
        index,
        id_map,
        reference=reference,
        current_nprobe=args.current_nprobe,
        exhaustive_nprobe=exhaustive_nprobe,
        candidate_depth=args.candidate_depth,
        deep_probe_depth=args.deep_probe_depth,
    )
    report = finalize_ann_probe_report(
        report,
        manifest=manifest,
        reference=reference,
        reference_path=args.reference_report,
        artifact_gate=artifact_gate,
        encoding=encoding,
        source_dev_count=source_dev_count,
        formal_run=formal_run,
    )
    write_ann_probe_outputs(
        report,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )
    logger.info("ANN probe status: %s", report["status"])
    logger.info("JSON report: %s", args.output_json)
    logger.info("Markdown report: %s", args.output_md)
    if formal_run and report["status"] != FORMAL_STATUS:
        raise RuntimeError("formal ANN probe failed its protocol or reproduction gate")


if __name__ == "__main__":
    main()
