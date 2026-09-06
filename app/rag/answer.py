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

from app.core.config import (
    DEFAULT_HYBRID_ALPHA,
    get_db_path,
    get_faiss_dir,
    validate_hybrid_alpha,
)
from app.core.schemas import AskResponse, CitationInfo, SearchResult
from app.rag.citation import verify_citations
from app.rag.context_builder import build_evidence
from app.rag.llm_provider import create_provider
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


def answer_question(
    question: str,
    pre_retrieved: list | None = None,
    top_k: int = 8,
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
        top_k: Number of evidence chunks (only used if pre_retrieved is None).
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
    if pre_retrieved is not None:
        results = _dicts_to_search_results(pre_retrieved)
        results = results[:top_k]
        logger.info("Using %d pre-retrieved chunks for question: %s", len(results), question[:60])
    else:
        # Fallback: standalone /ask call without pre-retrieved results
        results = _retrieve_evidence(
            question,
            top_k,
            retrieval_mode,
            effective_alpha,
            db_path,
            index_dir,
        )
        logger.info("Retrieved %d evidence chunks for question: %s", len(results), question[:60])

    # — 2. Build evidence context ———————————————————————————————————————
    evidence_text, citation_map = build_evidence(results, db_path=db_path)

    if not evidence_text:
        elapsed = (time.perf_counter() - t0) * 1000
        return AskResponse(
            question=question,
            answer="未找到相关证据，无法回答该问题。",
            effective_alpha=effective_alpha,
            citations=[],
            citation_valid=True,
            citation_warnings=[],
            latency_ms=round(elapsed, 2),
        )

    # — 3. Build prompts ————————————————————————————————————————————————
    system, user = build_prompts(evidence_text, question)

    # — 4. Call LLM —————————————————————————————————————————————————————
    llm = create_provider()
    llm_response = llm.generate(system=system, user=user)

    logger.info("LLM generated %d chars in %.0f ms", len(llm_response.text), llm_response.latency_ms)

    # — 5. Verify citations —————————————————————————————————————————————
    cit_result = verify_citations(llm_response.text, citation_map)

    # Build CitationInfo list
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

    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        "answer_question complete: citations=%d/%d valid=%s latency=%.0f ms",
        len(cit_result.cited_ids),
        len(citation_map),
        cit_result.valid,
        elapsed,
    )

    return AskResponse(
        question=question,
        answer=llm_response.text,
        effective_alpha=effective_alpha,
        citations=citations,
        citation_valid=cit_result.valid,
        citation_warnings=cit_result.warnings,
        latency_ms=round(elapsed, 2),
    )


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


async def answer_question_stream(
    question: str,
    pre_retrieved: list | None = None,
    top_k: int = 8,
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
    if pre_retrieved is not None:
        results = _dicts_to_search_results(pre_retrieved)
        results = results[:top_k]
        logger.info("Using %d pre-retrieved chunks for question: %s", len(results), question[:60])
        # Skip "retrieving" phase — results are already available
    else:
        # Fallback: do internal retrieval
        yield _sse("status", {"phase": "retrieving", "message": _PHASE_MESSAGES["retrieving"]})

        def _retrieve():
            return _retrieve_evidence(
                question,
                top_k,
                retrieval_mode,
                effective_alpha,
                db_path,
                index_dir,
            )

        results = await asyncio.to_thread(_retrieve)
        logger.info("Retrieved %d evidence chunks for question: %s", len(results), question[:60])

    # ---- Phase 2: Organizing -----------------------------------------------
    yield _sse("status", {"phase": "organizing", "message": _PHASE_MESSAGES["organizing"]})

    def _build():
        evidence_text, citation_map = build_evidence(results, db_path=db_path)
        if not evidence_text:
            return None, None, None, None
        system, user = build_prompts(evidence_text, question)
        return evidence_text, citation_map, system, user

    evidence_text, citation_map, system_prompt, user_prompt = await asyncio.to_thread(_build)

    if not evidence_text:
        elapsed = (time.perf_counter() - t0) * 1000
        yield _sse("result", {
            "question": question,
            "answer": "未找到相关证据，无法回答该问题。",
            "effective_alpha": effective_alpha,
            "citations": [],
            "citation_valid": True,
            "citation_warnings": [],
            "latency_ms": round(elapsed, 2),
        })
        yield _sse("done", "")
        return

    # ---- Phase 3: Generating -----------------------------------------------
    yield _sse("status", {"phase": "generating", "message": _PHASE_MESSAGES["generating"]})

    def _generate():
        llm = create_provider()
        return llm.generate(system=system_prompt, user=user_prompt)

    llm_response = await asyncio.to_thread(_generate)
    logger.info("LLM generated %d chars in %.0f ms", len(llm_response.text), llm_response.latency_ms)

    # ---- Phase 4: Verifying ------------------------------------------------
    yield _sse("status", {"phase": "verifying", "message": _PHASE_MESSAGES["verifying"]})

    def _verify():
        return verify_citations(llm_response.text, citation_map)

    cit_result = await asyncio.to_thread(_verify)

    # ---- Build result ------------------------------------------------------
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

    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        "answer_question_stream complete: citations=%d/%d valid=%s latency=%.0f ms",
        len(cit_result.cited_ids), len(citation_map), cit_result.valid, elapsed,
    )

    yield _sse("result", {
        "question": question,
        "answer": llm_response.text,
        "effective_alpha": effective_alpha,
        "citations": [
            {
                "citation_id": c.citation_id,
                "paper_id": c.paper_id,
                "chunk_id": c.chunk_id,
                "title": c.title,
                "url": c.url,
            }
            for c in citations
        ],
        "citation_valid": cit_result.valid,
        "citation_warnings": cit_result.warnings,
        "latency_ms": round(elapsed, 2),
    })
    yield _sse("done", "")
