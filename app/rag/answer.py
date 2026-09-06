"""
Full RAG pipeline: retrieve → build evidence → call LLM → verify citations.

Usage:
    from app.rag.answer import answer_question
    response = answer_question("神经网络如何优化？", top_k=5)

Streaming (SSE):
    from app.rag.answer import answer_question_stream
    async for event in answer_question_stream("神经网络如何优化？"):
        ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.core.config import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RAG_TOP_K,
    get_db_path,
    get_faiss_dir,
    validate_hybrid_alpha,
)
from app.core.schemas import AskResponse, CitationInfo, SearchResult
from app.rag.citation import CitationResult, verify_citations
from app.rag.context_builder import build_evidence
from app.rag.llm_provider import LLMResponse, create_provider
from app.rag.prompt import build_prompts
from app.rag.rewriter import prepare_lexical_query
from app.retrieval.hybrid import search_hybrid
from app.retrieval.lexical import search_lexical

logger = logging.getLogger(__name__)


def _effective_hybrid_alpha(
    retrieval_mode: str,
    alpha: float | None,
) -> float | None:
    """Validate Hybrid alpha without consulting production environment config."""
    if retrieval_mode != "hybrid":
        return None
    value = DEFAULT_HYBRID_ALPHA if alpha is None else alpha
    return validate_hybrid_alpha(value)


def _retrieve_evidence(
    question: str,
    top_k: int,
    retrieval_mode: str,
    effective_alpha: float | None,
    db_path: Path,
    index_dir: Path,
) -> list[SearchResult]:
    """Retrieve evidence with optional rewrite confined to the BM25 branch."""
    lexical_query = question
    uses_lexical_signal = retrieval_mode == "lexical" or (
        retrieval_mode == "hybrid"
        and effective_alpha is not None
        and effective_alpha > 0
    )
    if uses_lexical_signal:
        lexical_query, _ = prepare_lexical_query(question)

    if retrieval_mode == "hybrid":
        assert effective_alpha is not None
        return search_hybrid(
            question,
            top_k=top_k,
            alpha=effective_alpha,
            db_path=db_path,
            index_dir=index_dir,
            lexical_query=lexical_query,
        )
    if retrieval_mode == "vector":
        from app.retrieval.vector_store import search_vector

        return search_vector(
            question,
            top_k=top_k,
            db_path=db_path,
            index_dir=index_dir,
        )
    return search_lexical(lexical_query, top_k=top_k, db_path=db_path)


def _dicts_to_search_results(raw: list[dict | SearchResult]) -> list[SearchResult]:
    """Normalize pre-retrieved transport dicts or in-process result objects."""
    results = []
    for item in raw:
        if isinstance(item, SearchResult):
            results.append(item)
            continue
        try:
            results.append(SearchResult(**item))
        except (TypeError, ValueError):
            pass  # skip malformed entries
    return results


def _resolve_evidence(
    question: str,
    pre_retrieved: list[dict | SearchResult] | None,
    top_k: int,
    retrieval_mode: str,
    effective_alpha: float | None,
    db_path: Path,
    index_dir: Path,
) -> list[SearchResult]:
    """Reuse supplied chunks, including an empty list, or retrieve them once."""
    if pre_retrieved is not None:
        results = _dicts_to_search_results(pre_retrieved)[:top_k]
        source = "pre-retrieved"
    else:
        results = _retrieve_evidence(
            question, top_k, retrieval_mode, effective_alpha, db_path, index_dir,
        )
        source = "retrieved"
    logger.info("Using %d %s chunks for question: %s", len(results), source, question[:60])
    return results


def _generate_answer(system: str, user: str) -> LLMResponse:
    """Call the configured provider and log its generation timing."""
    llm = create_provider()
    response = llm.generate(system=system, user=user)
    logger.info("LLM generated %d chars in %.0f ms", len(response.text), response.latency_ms)
    return response


def _build_response(
    question: str,
    answer_text: str | None,
    citation_map: list[dict[str, Any]],
    effective_alpha: float | None,
    started_at: float,
) -> AskResponse:
    """Verify and format either response; None means evidence was unavailable.

    A no-evidence refusal skips citation verification. An empty LLM answer
    still goes through verification and receives the missing-citation warning.
    """
    cit_result = (
        verify_citations(answer_text, citation_map)
        if answer_text is not None else CitationResult()
    )
    citations = [
        CitationInfo(
            citation_id=c["citation_id"],
            paper_id=c["paper_id"],
            chunk_id=c["chunk_id"],
            title=c["title"],
            url=c.get("url"),
        )
        for c in citation_map
    ]
    elapsed = (time.perf_counter() - started_at) * 1000
    logger.info(
        "RAG answer complete: citations=%d/%d valid=%s latency=%.0f ms",
        len(cit_result.cited_ids), len(citation_map), cit_result.valid, elapsed,
    )
    return AskResponse(
        question=question,
        answer=answer_text if answer_text is not None else "未找到相关证据，无法回答该问题。",
        effective_alpha=effective_alpha,
        citations=citations,
        citation_valid=cit_result.valid,
        citation_warnings=cit_result.warnings,
        latency_ms=round(elapsed, 2),
    )


def answer_question(
    question: str,
    pre_retrieved: list[dict | SearchResult] | None = None,
    top_k: int = DEFAULT_RAG_TOP_K,
    retrieval_mode: str = "hybrid",
    alpha: float | None = DEFAULT_HYBRID_ALPHA,
    db_path: str | Path | None = None,
    index_dir: str | Path | None = None,
) -> AskResponse:
    """Answer a question with citation-grounded RAG.

    Pipeline:
    1. Use pre_retrieved results (if provided), otherwise do internal retrieval.
    2. Format evidence block with [N] citation IDs.
    3. Build system + user prompts.
    4. Call LLM to generate an answer.
    5. Verify that citation markers are valid.

    Args:
        question: Natural-language question.
        pre_retrieved: Optional pre-retrieved search results from /search.
                       When provided, internal retrieval is skipped entirely.
        top_k: Maximum evidence chunks, also applied to pre_retrieved results.
        retrieval_mode: 'lexical', 'vector', or 'hybrid' (fallback only).
        alpha: Hybrid weight (fallback only).

    Returns:
        AskResponse with answer text, citations, validity, and latency.
    """
    t0 = time.perf_counter()
    db_path = Path(db_path) if db_path is not None else get_db_path()
    index_dir = Path(index_dir) if index_dir is not None else get_faiss_dir()
    effective_alpha = _effective_hybrid_alpha(retrieval_mode, alpha)

    # — 1. Evidence: reuse pre-retrieved or do internal retrieval ———————
    results = _resolve_evidence(
        question, pre_retrieved, top_k, retrieval_mode, effective_alpha, db_path, index_dir,
    )

    # — 2. Build evidence context ———————————————————————————————————————
    evidence_text, citation_map = build_evidence(results, db_path=db_path)

    if not evidence_text:
        return _build_response(question, None, [], effective_alpha, t0)

    # — 3. Build prompts ————————————————————————————————————————————————
    system, user = build_prompts(evidence_text, question)

    # — 4. Call LLM —————————————————————————————————————————————————————
    llm_response = _generate_answer(system, user)

    # — 5. Verify citations —————————————————————————————————————————————
    return _build_response(question, llm_response.text, citation_map, effective_alpha, t0)


# ============================================================================
# Streaming (SSE) variant — real-time phase updates for the frontend
# ============================================================================


def _sse(event_type: str, data: dict | str = "") -> str:
    """Format a single SSE event.

    Args:
        event_type: SSE event name (e.g. 'status', 'result', 'done').
        data: Either a dict (serialised as JSON) or a plain string.
    """
    if isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = data
    return f"event: {event_type}\ndata: {payload}\n\n"


# Map phase keys to user-visible Chinese messages
_PHASE_MESSAGES = {
    "retrieving": "正在检索相关论文...",
    "organizing": "正在阅读并整理信息...",
    "generating": "正在生成回答...",
    "verifying": "正在审核引用...",
}


async def answer_question_stream(
    question: str,
    pre_retrieved: list[dict | SearchResult] | None = None,
    top_k: int = DEFAULT_RAG_TOP_K,
    retrieval_mode: str = "hybrid",
    alpha: float | None = DEFAULT_HYBRID_ALPHA,
    db_path: str | Path | None = None,
    index_dir: str | Path | None = None,
):
    """Async generator that yields SSE events as the RAG pipeline progresses.

    Yields:
        SSE-formatted strings (event + data pairs) for each phase transition
        and the final result. Consume with::

            async for event in answer_question_stream(q):
                send_to_client(event)

    Uses ``asyncio.to_thread`` for blocking I/O (LLM calls, DB queries) so
    the event loop stays responsive.
    """
    t0 = time.perf_counter()
    db_path = Path(db_path) if db_path is not None else get_db_path()
    index_dir = Path(index_dir) if index_dir is not None else get_faiss_dir()
    effective_alpha = _effective_hybrid_alpha(retrieval_mode, alpha)

    # ---- Phase 1: Retrieving / Organizing ----------------------------------
    if pre_retrieved is None:
        yield _sse("status", {"phase": "retrieving", "message": _PHASE_MESSAGES["retrieving"]})

    results = await asyncio.to_thread(
        _resolve_evidence,
        question, pre_retrieved, top_k, retrieval_mode, effective_alpha, db_path, index_dir,
    )

    # ---- Phase 2: Organizing -----------------------------------------------
    yield _sse("status", {"phase": "organizing", "message": _PHASE_MESSAGES["organizing"]})

    evidence_text, citation_map = await asyncio.to_thread(build_evidence, results, db_path=db_path)

    if not evidence_text:
        response = _build_response(question, None, [], effective_alpha, t0)
        yield _sse("result", response.model_dump(mode="json"))
        yield _sse("done", "")
        return

    system_prompt, user_prompt = build_prompts(evidence_text, question)

    # ---- Phase 3: Generating -----------------------------------------------
    yield _sse("status", {"phase": "generating", "message": _PHASE_MESSAGES["generating"]})

    llm_response = await asyncio.to_thread(_generate_answer, system_prompt, user_prompt)

    # ---- Phase 4: Verifying ------------------------------------------------
    yield _sse("status", {"phase": "verifying", "message": _PHASE_MESSAGES["verifying"]})

    response = await asyncio.to_thread(
        _build_response, question, llm_response.text, citation_map, effective_alpha, t0,
    )

    yield _sse("result", response.model_dump(mode="json"))
    yield _sse("done", "")
