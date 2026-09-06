"""
Query rewriter: extract English academic keywords to improve lexical recall.

When a query is in Chinese (or mixed), an optional real LLM extracts English
keywords. The keywords are combined with the original query before lexical
retrieval, while Dense retrieval continues to use the original query.

Usage:
    from app.rag.rewriter import prepare_lexical_query
    lexical_query, keywords = prepare_lexical_query("为什么GAN训练不稳定？")
    # keywords → "GAN training mode collapse instability convergence"
"""

from __future__ import annotations

import logging
import re
import time

from app.core.config import get_rewrite_timeout_seconds
from app.rag.llm_provider import create_provider, is_llm_configured

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿"


def detect_language(text: str) -> str:
    """Return 'zh' if text contains CJK characters, otherwise 'en'."""
    for c in text:
        if _is_cjk(c):
            return "zh"
    return "en"


# ---------------------------------------------------------------------------
# Keyword extraction prompt
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """从以下查询中提取 3-5 个英文学术关键词，用于搜索论文。只返回关键词，用空格分隔，不要其他内容。

查询："{query}"

关键词："""

_KEYWORD_PREFIX_RE = re.compile(r"^(?:keywords?|关键词)\s*[:：-]\s*", re.IGNORECASE)
_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._+/-][A-Za-z0-9]+)*")
_MAX_KEYWORD_CHARS = 200
_MAX_KEYWORD_TOKENS = 12
_REWRITE_MAX_TOKENS = 128


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validated_keywords(text: str) -> str:
    """Return a compact English keyword string, or an empty string if invalid."""
    candidate = _KEYWORD_PREFIX_RE.sub("", text.strip().strip("`\"'"))
    if (
        not candidate
        or len(candidate) > _MAX_KEYWORD_CHARS
        or any(_is_cjk(char) for char in candidate)
        or not re.search(r"[A-Za-z]", candidate)
    ):
        return ""

    tokens = _ENGLISH_TOKEN_RE.findall(candidate)
    if not tokens or len(tokens) > _MAX_KEYWORD_TOKENS:
        return ""
    return " ".join(tokens)


def rewrite_query(query: str) -> str:
    """Extract English academic keywords from a query.

    If the query is already in English, returns the original query as-is.
    If Chinese/mixed, calls LLM to extract English keywords.

    Args:
        query: Raw user query (any language).

    Returns:
        Space-separated English keywords suitable for lexical search.
    """
    lang = detect_language(query)

    # English queries don't need rewriting for lexical search
    if lang == "en":
        logger.debug("rewrite: English query, using as-is")
        return query

    if not is_llm_configured():
        logger.info("rewrite status=skipped_no_api_key latency_ms=0.0")
        return query

    try:
        timeout = get_rewrite_timeout_seconds()
    except ValueError as exc:
        logger.error("rewrite status=invalid_timeout latency_ms=0.0 error=%s", exc)
        return query

    # Chinese/mixed: ask a real LLM to extract English keywords. Mock output is
    # deliberately excluded because it is a test answer, not a query rewrite.
    prompt = _REWRITE_PROMPT.format(query=query)
    started = time.perf_counter()

    try:
        llm = create_provider(timeout=timeout, short_task=True)
        response = llm.generate(
            user=prompt,
            temperature=0.0,
            max_tokens=_REWRITE_MAX_TOKENS,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        status = "timeout" if "timeout" in type(exc).__name__.lower() else "provider_error"
        logger.warning(
            "rewrite status=%s latency_ms=%.1f error_type=%s; using original query",
            status,
            elapsed_ms,
            type(exc).__name__,
        )
        return query

    elapsed_ms = (time.perf_counter() - started) * 1000
    keywords = _validated_keywords(response.text)
    if not keywords:
        logger.warning(
            "rewrite status=invalid_output latency_ms=%.1f; using original query",
            elapsed_ms,
        )
        return query

    logger.info(
        "rewrite status=success latency_ms=%.1f query=%r keywords=%r",
        elapsed_ms,
        query[:60],
        keywords[:80],
    )
    return keywords


def prepare_lexical_query(query: str) -> tuple[str, str]:
    """Return the effective BM25 query and successful rewrite keywords.

    Failure and non-applicable paths return ``(query, "")`` exactly. A valid
    rewrite is appended to the original text so FTS5 can rank all OR terms in a
    single BM25 result list.
    """
    keywords = rewrite_query(query)
    if keywords == query:
        return query, ""
    return f"{query} {keywords}", keywords
