"""
Context builder: format retrieved chunks into an evidence block for the LLM.

Each chunk gets a citation ID like [1], [2], [3]. The evidence block respects
an estimated token budget; provider tokenization and prompt/output overhead
are separate from this budget.

Usage:
    from app.rag.context_builder import build_evidence
    evidence, id_map = build_evidence(search_results, max_tokens=8000)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import get_rag_context_tokens, validate_rag_context_tokens
from app.core.schemas import SearchResult

logger = logging.getLogger(__name__)

# Portable estimate for predominantly English papers: four ASCII characters
# per token; conservatively allow two tokens per non-ASCII character. These
# integer quarter-token units also let us budget separators and truncation
# markers exactly under the same estimate, without downloading a tokenizer.
_TRUNCATION_MARKER = " …（内容已截断）"


def _token_units(text: str) -> int:
    return sum(1 if char.isascii() else 8 for char in text)


def _estimate_tokens(text: str) -> int:
    """Estimate evidence tokens; this is not the provider's tokenizer count."""
    return (_token_units(text) + 3) // 4


def _truncate_text(text: str, max_units: int) -> str:
    """Keep a Unicode-safe prefix within the remaining estimate."""
    used = 0
    end = 0
    for char in text:
        units = 1 if char.isascii() else 8
        if used + units > max_units:
            break
        used += units
        end += 1
    prefix = text[:end].rstrip()
    # Avoid ending halfway through an English word when a boundary exists.
    if (
        end < len(text) and prefix
        and prefix[-1].isascii() and prefix[-1].isalnum()
        and text[end].isascii() and text[end].isalnum()
    ):
        boundary = prefix.rfind(" ")
        if boundary > 0:
            prefix = prefix[:boundary].rstrip()
    return prefix


def build_evidence(
    results: list[SearchResult],
    db_path: str | Path = "data/indexes/metadata.sqlite",
    max_tokens: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a formatted evidence block from search results.

    Reads text and paper metadata from SQLite, checking that each chunk belongs
    to the supplied paper. Missing, mismatched, and empty chunks are skipped;
    request snippets and titles are never used as substitute evidence.

    Args:
        results: Ranked search results from any retriever.
        db_path: Path to metadata SQLite DB (for fetching chunk_text).
        max_tokens: Estimated evidence-token budget. None reads
                    CITEQUEST_RAG_CONTEXT_TOKENS (default 8000).

    Complete chunks are included in retrieval order. The last fitting chunk
    may be shortened, with an explicit marker; even the first chunk cannot
    exceed the estimate. Prompts and generated output need separate headroom.

    Returns:
        (evidence_text, citation_map) where:
        - evidence_text is a formatted string like "[1] Title: ...\\n内容: ..."
        - citation_map is a list of {citation_id, paper_id, chunk_id, title, url}
          used by the verifier to validate citations.
    """
    budget = get_rag_context_tokens() if max_tokens is None else validate_rag_context_tokens(max_tokens)
    if not results:
        return "", []

    # Open read-only: an incorrect path must not create an empty database.
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        chunk_ids = [r.chunk_id for r in results]
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"""SELECT c.chunk_id, c.paper_id, c.chunk_text, c.chunk_type,
                       p.title, p.year, p.venue, p.url
                FROM chunks c JOIN papers p ON p.paper_id = c.paper_id
                WHERE c.chunk_id IN ({placeholders})""",
            chunk_ids,
        ).fetchall()
        chunks = {row["chunk_id"]: row for row in rows}
    finally:
        conn.close()

    budget_units = budget * 4
    evidence_parts: list[str] = []
    citation_map: list[dict[str, Any]] = []
    used_units = 0
    truncated_chunks = 0
    next_id = 1
    seen_chunks: set[str] = set()

    for r in results:
        chunk = chunks.get(r.chunk_id)
        if chunk is None or chunk["paper_id"] != r.paper_id:
            logger.warning("Skipping missing or mismatched evidence chunk: %s", r.chunk_id)
            continue
        if r.chunk_id in seen_chunks:
            continue
        raw_text = chunk["chunk_text"].strip()
        if not raw_text:
            logger.warning("Skipping empty evidence chunk: %s", r.chunk_id)
            continue

        # Build one evidence entry
        parts = [f"[{next_id}]"]
        parts.append(f"标题: {chunk['title']}")

        if chunk["year"]:
            parts.append(f"年份: {chunk['year']}")
        if chunk["venue"]:
            parts.append(f"来源: {chunk['venue']}")

        # Extract just the abstract from chunk_text.
        # chunk_text format: "Title: ...\nAbstract: ..."
        # Title is already shown above, so only include the abstract part.
        evidence_text = raw_text
        if chunk["chunk_type"] in {"title_abstract", "metadata", "abstract"}:
            if "\nAbstract: " in raw_text:
                evidence_text = raw_text.split("\nAbstract: ", 1)[1].strip()
            elif raw_text.startswith("Title: "):
                evidence_text = raw_text[len("Title: "):].strip()
        if not evidence_text:
            continue
        header = "\n".join(parts) + "\n内容: "
        separator = "\n\n" if evidence_parts else ""
        entry = header + evidence_text
        remaining = budget_units - used_units
        shortened = _token_units(separator + entry) > remaining
        if shortened:
            overhead = _token_units(separator + header + _TRUNCATION_MARKER)
            prefix = _truncate_text(evidence_text, remaining - overhead)
            if not prefix:
                # Another candidate with a shorter title may still fit.
                continue
            entry = header + prefix + _TRUNCATION_MARKER
            truncated_chunks += 1

        evidence_parts.append(entry)
        used_units += _token_units(separator + entry)
        seen_chunks.add(r.chunk_id)

        citation_map.append({
            "citation_id": next_id,
            "paper_id": chunk["paper_id"],
            "chunk_id": r.chunk_id,
            "title": chunk["title"],
            "url": chunk["url"] or None,
        })

        next_id += 1
        if shortened:
            break

    evidence_text = "\n\n".join(evidence_parts)

    logger.info(
        "build_evidence: candidates=%d used=%d estimated_tokens=%d budget=%d truncated=%d",
        len(results), len(evidence_parts),
        _estimate_tokens(evidence_text),
        budget, truncated_chunks,
    )

    return evidence_text, citation_map
