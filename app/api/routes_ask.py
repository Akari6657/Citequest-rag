"""
RAG ask API route for CiteQuest-RAG.

POST /ask        — ask a question, get a citation-grounded answer.
POST /ask/stream — same, but via SSE with real-time phase updates.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.retrieval_config import resolve_request_hybrid_alpha
from app.core.schemas import AskRequest, AskResponse
from app.rag.answer import answer_question, answer_question_stream

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Ask a question and receive a citation-grounded answer.

    Evidence is retrieved via hybrid search, formatted with citation IDs,
    and sent to an LLM that answers strictly based on the provided evidence.
    """
    effective_alpha = resolve_request_hybrid_alpha(
        request.retrieval_mode, request.alpha
    )
    logger.info(
        "ask question=%r top_k=%d mode=%s effective_alpha=%s",
        request.question,
        request.top_k,
        request.retrieval_mode,
        f"{effective_alpha:.2f}" if effective_alpha is not None else "n/a",
    )

    return answer_question(
        question=request.question,
        pre_retrieved=request.pre_retrieved,
        top_k=request.top_k,
        retrieval_mode=request.retrieval_mode,
        alpha=effective_alpha,
    )


@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """Ask a question with SSE streaming — real-time phase updates.

    Returns ``text/event-stream`` with these event types:

    - ``status`` — pipeline phase change (retrieving/organizing/generating/verifying)
    - ``result`` — final AskResponse as JSON
    - ``done``   — stream end marker

    The router runs first; if it decides against RAG, an empty result
    is returned immediately without calling the LLM.
    """
    from app.rag.router import route_query

    router_result = route_query(request.question)
    if not router_result.should_rag:
        logger.info("ask/stream skipped by router: %s", router_result.reason)
        return StreamingResponse(
            _router_skip_stream(router_result.reason),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    effective_alpha = resolve_request_hybrid_alpha(
        request.retrieval_mode, request.alpha
    )
    logger.info(
        "ask/stream question=%r top_k=%d mode=%s effective_alpha=%s",
        request.question,
        request.top_k,
        request.retrieval_mode,
        f"{effective_alpha:.2f}" if effective_alpha is not None else "n/a",
    )

    return StreamingResponse(
        answer_question_stream(
            question=request.question,
            pre_retrieved=request.pre_retrieved,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            alpha=effective_alpha,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


async def _router_skip_stream(reason: str):
    """Yield a minimal SSE stream when the router says no RAG is needed."""
    import json
    yield f"event: status\ndata: {json.dumps({'phase': 'skipped', 'message': f'路由判断：{reason}，跳过 AI Overview'})}\n\n"
    yield f"event: done\ndata: \n\n"
