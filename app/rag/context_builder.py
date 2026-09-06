"""
Context builder: format retrieved chunks into an evidence block for the LLM.

Each chunk gets a citation ID like [1], [2], [3].  The evidence block
respects a token budget so we don't exceed the model's context window.

Usage:
    from app.rag.context_builder import build_evidence
    evidence, id_map = build_evidence(search_results, max_tokens=2000)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.schemas import SearchResult

logger = logging.getLogger(__name__)

# Rough estimate: English ≈ 1.3 tokens/char, Chinese ≈ 0.5 chars/token.
# We use a conservative 1.5 chars/token as a universal estimate.
CHARS_PER_TOKEN = 1.5
DEFAULT_MAX_TOKENS = 2000


def _estimate_tokens(text: str) -> int:
    """Rough token count for mixed Chinese/English text."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def build_evidence(
    results: list[SearchResult],
    db_path: str | Path = "data/indexes/metadata.sqlite",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a formatted evidence block from search results.

    Fetches full chunk_text from the database — SearchResult.snippet is a
    display hint, not the complete evidence.

    Args:
        results: Ranked search results from any retriever.
        db_path: Path to metadata SQLite DB (for fetching chunk_text).
        max_tokens: Maximum tokens for the evidence block (default 2000).

    Returns:
        (evidence_text, citation_map) where:
        - evidence_text is a formatted string like "[1] Title: ...\\n内容: ..."
        - citation_map is a list of {citation_id, paper_id, chunk_id, title, url}
          used by the verifier to validate citations.
    """
    if not results:
        return "", []

    # Fetch full chunk_text from DB
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    chunk_texts: dict[str, str] = {}
    try:
        chunk_ids = [r.chunk_id for r in results]
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT chunk_id, chunk_text FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        for cid, ctext in rows:
            chunk_texts[cid] = ctext
    finally:
        conn.close()

    budget_chars = max_tokens * CHARS_PER_TOKEN
    evidence_parts: list[str] = []
    citation_map: list[dict[str, Any]] = []
    used_chars = 0
    next_id = 1

    for r in results:
        # Build one evidence entry
        parts = [f"[{next_id}]"]
        parts.append(f"标题: {r.title}")

        if r.year:
            parts.append(f"年份: {r.year}")
        if r.venue:
            parts.append(f"来源: {r.venue}")

        # Extract just the abstract from chunk_text.
        # chunk_text format: "Title: ...\nAbstract: ..."
        # Title is already shown above, so only include the abstract part.
        raw_text = chunk_texts.get(r.chunk_id, "") or r.snippet or r.title or ""
        if "\nAbstract: " in raw_text:
            evidence_text = raw_text.split("\nAbstract: ", 1)[1]
        elif raw_text.startswith("Title: "):
            evidence_text = raw_text[len("Title: "):]
        else:
            evidence_text = raw_text
        parts.append(f"摘要: {evidence_text}")

        entry = "\n".join(parts)
        entry_chars = len(entry)

        # Check budget — include at least the first result
        if used_chars + entry_chars > budget_chars and next_id > 1:
            logger.debug(
                "Evidence budget reached: %d/%d chars after %d chunks",
                used_chars,
                budget_chars,
                next_id - 1,
            )
            break

        evidence_parts.append(entry)
        used_chars += entry_chars + 2  # +2 for "\n\n" separator

        citation_map.append({
            "citation_id": next_id,
            "paper_id": r.paper_id,
            "chunk_id": r.chunk_id,
            "title": r.title,
            "url": f"https://arxiv.org/abs/{r.paper_id}" if r.paper_id else None,
        })

        next_id += 1

    evidence_text = "\n\n".join(evidence_parts)

    logger.debug(
        "build_evidence: %d chunks → %d chars (~%d tokens)",
        len(evidence_parts),
        len(evidence_text),
        _estimate_tokens(evidence_text),
    )

    return evidence_text, citation_map
