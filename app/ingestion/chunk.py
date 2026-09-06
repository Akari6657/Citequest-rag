"""
Chunk generator: turn a normalized paper into searchable text segments.

v0.1: one chunk per paper (Title + Abstract only).
v0.6: multi-chunk — title_abstract + body paragraphs with greedy merging.

Strategy:
  - Chunk 0: "Title: {title}\\nAbstract: {abstract}" (~150 tokens, fixed).
  - Chunk 1+: body text, greedy paragraph merge to ~800 tokens.
  - Overlap: 1 paragraph between consecutive chunks.
  - Sliding window fallback for paragraphs > 800 tokens (120t overlap, sentence-aligned).

Usage:
    from app.ingestion.chunk import chunk_paper
    chunks = chunk_paper(normalized_paper)  # list[dict]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_TARGET_TOKENS = 800
"""Target size for body chunks; several can fit into the separately configured
RAG evidence budget. Index chunking is independent of runtime context size."""

CHUNK_OVERLAP_WINDOW = 120
"""Token overlap for sliding window on very long paragraphs (~15% of target)."""

SENTENCE_TOLERANCE = 40
"""± tokens to look for a sentence boundary when placing a window cut."""

MIN_CHUNK_TOKENS = 50
"""Chunks shorter than this are discarded (semantically incomplete)."""

SENTENCE_BOUNDARIES = ".!?\n"
"""Characters that mark a sentence or paragraph boundary."""

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token count. For English academic text, ~1.3 tokens per word."""
    words = len(text.split())
    return max(1, int(words * 1.3))


# ---------------------------------------------------------------------------
# Chunk text builders
# ---------------------------------------------------------------------------


def _build_title_abstract(paper: dict[str, Any]) -> str:
    """Chunk 0: Title + Abstract — the core metadata chunk."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    if title and abstract:
        return f"Title: {title}\nAbstract: {abstract}"
    elif title:
        return f"Title: {title}"
    else:
        return f"Abstract: {abstract}"


def _make_chunk(paper_id: str, chunk_text: str, chunk_type: str, position: int) -> dict[str, Any]:
    """Build a single chunk dict."""
    return {
        "chunk_id": f"{paper_id}_chunk{position}",
        "paper_id": paper_id,
        "chunk_text": chunk_text,
        "chunk_type": chunk_type,
        "token_count": _estimate_tokens(chunk_text),
        "position": position,
    }


# ---------------------------------------------------------------------------
# Sliding window (for paragraphs > 800 tokens)
# ---------------------------------------------------------------------------


def _find_sentence_boundary(text: str, target: int) -> int:
    """Find the nearest sentence boundary within ±SENTENCE_TOLERANCE of target.

    Returns the index AFTER the boundary character (i.e., the start of the
    next sentence). Falls back to the exact target if no boundary is found.
    """
    # Search outward from target: prefer a good cut nearby over a far one
    for offset in range(SENTENCE_TOLERANCE + 1):
        for direction in [0, -1, 1]:  # exact, before, after
            pos = target + offset * direction
            if 0 <= pos < len(text):
                if text[pos] in SENTENCE_BOUNDARIES:
                    return pos + 1
    return target  # absolute fallback


def _sliding_window(text: str, target: int, overlap: int) -> list[str]:
    """Split a very long text into overlapping windows.

    Each window is ~target tokens, starting at sentence-aligned boundaries
    with ~overlap tokens of shared context between consecutive windows.
    """
    windows = []
    start = 0

    while start < len(text):
        # Where would we cut without sentence alignment?
        rough_end = min(start + target * 3, len(text))  # ~target tokens ≈ target*3 chars

        # Find a sentence boundary near rough_end
        if rough_end < len(text):
            cut = _find_sentence_boundary(text, rough_end)
        else:
            cut = len(text)

        window_text = text[start:cut].strip()
        if window_text and _estimate_tokens(window_text) >= MIN_CHUNK_TOKENS:
            windows.append(window_text)

        # Next window starts overlap tokens before the cut
        overlap_chars = overlap * 3  # tokens → chars approximation
        start = max(cut - overlap_chars, start + target)  # ensure forward progress

        if start >= len(text):
            break

    return windows


# ---------------------------------------------------------------------------
# Greedy paragraph merging (body)
# ---------------------------------------------------------------------------


def _merge_body_paragraphs(paragraphs: list[str]) -> list[str]:
    """Merge body paragraphs into chunks using greedy accumulation.

    Each paragraph starts on its own. We accumulate consecutive paragraphs
    until the total exceeds CHUNK_TARGET_TOKENS, then emit the chunk and
    start a new one overlapping the last paragraph.
    """
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    prev_paras: list[str] = []  # last emitted chunk's paragraphs (for overlap)

    for i, para in enumerate(paragraphs):
        para_tokens = _estimate_tokens(para)

        # ---- Very long paragraph → sliding window ----
        if para_tokens > CHUNK_TARGET_TOKENS:
            # Emit whatever we've accumulated first
            if buffer:
                chunks.append("\n\n".join(buffer))
                prev_paras = list(buffer)
                buffer = []

            # Sliding window on this paragraph
            windows = _sliding_window(para, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_WINDOW)
            chunks.extend(windows)

            # Overlap: carry the last window's tail into the next chunk
            if windows:
                last_win = windows[-1]
                # Approximate: carry ~1 sentence from the last window
                buffer = [last_win] if _estimate_tokens(last_win) < CHUNK_TARGET_TOKENS // 2 else []
            continue

        # ---- Normal paragraph: try to fit in buffer ----
        merged_text = "\n\n".join(buffer + [para])
        merged_tokens = _estimate_tokens(merged_text)

        if merged_tokens <= CHUNK_TARGET_TOKENS:
            # Fits — add to buffer
            buffer.append(para)
        else:
            # Doesn't fit — emit current buffer
            if buffer:
                chunks.append("\n\n".join(buffer))
                prev_paras = list(buffer)

            # New buffer: overlap the last paragraph from previous chunk
            if prev_paras:
                buffer = [prev_paras[-1], para]
            else:
                buffer = [para]

    # ---- Emit remaining buffer ----
    if buffer:
        final_text = "\n\n".join(buffer)
        if _estimate_tokens(final_text) >= MIN_CHUNK_TOKENS:
            chunks.append(final_text)

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_paper(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Create multiple chunks from a normalized paper.

    Returns:
        list of chunk dicts. Chunk 0 is always title_abstract.
        Body chunks follow (if full_text is available).
    """
    paper_id = paper["paper_id"]
    all_chunks: list[dict[str, Any]] = []

    # Chunk 0: Title + Abstract
    ta_text = _build_title_abstract(paper)
    all_chunks.append(_make_chunk(paper_id, ta_text, "title_abstract", 0))

    # Body chunks: from full_text
    full_text = paper.get("full_text", "")
    if not full_text:
        return all_chunks

    paragraphs = full_text.split("\n\n")
    # Skip paragraph 0 (title) and paragraph 1 (abstract) — already in chunk 0
    body_paras = paragraphs[2:] if len(paragraphs) > 2 else []

    if not body_paras:
        return all_chunks

    body_chunks = _merge_body_paragraphs(body_paras)

    for i, chunk_text in enumerate(body_chunks):
        all_chunks.append(_make_chunk(paper_id, chunk_text, "body", i + 1))

    logger.debug("Chunking %s → %d chunks (title_abstract + %d body)",
                 paper_id, len(all_chunks), len(body_chunks))

    return all_chunks
