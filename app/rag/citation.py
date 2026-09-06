"""
Citation extraction and verification.

After the LLM generates an answer, we:
1. Extract all citation markers [1], [2], [3] from the text.
2. Verify each one maps to a valid evidence chunk.
3. Warn about missing or invalid citations.

Usage:
    from app.rag.citation import verify_citations
    result = verify_citations(answer_text, citation_map)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Matches [1], [2], [123], etc.
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class CitationResult:
    """Result of citation verification."""

    valid: bool = True
    """True if all citation markers reference existing evidence chunks."""

    cited_ids: list[int] = field(default_factory=list)
    """All citation IDs found in the answer (deduplicated)."""

    invalid_ids: list[int] = field(default_factory=list)
    """Citation IDs that don't correspond to any evidence chunk."""

    warnings: list[str] = field(default_factory=list)
    """Human-readable warnings about citation issues."""


def extract_citations(text: str) -> list[int]:
    """Extract all citation marker numbers from an answer text.

    Example: "This is shown in [1] and [3]." → [1, 3]
    """
    matches = _CITATION_RE.findall(text)
    # Deduplicate while preserving order
    seen = set()
    ids = []
    for m in matches:
        n = int(m)
        if n not in seen:
            seen.add(n)
            ids.append(n)
    return ids


def verify_citations(answer_text: str, citation_map: list[dict[str, Any]]) -> CitationResult:
    """Verify that citation markers in the answer are valid.

    Args:
        answer_text: The LLM-generated answer text.
        citation_map: List of {citation_id, paper_id, chunk_id, title, url}
                      from context_builder.build_evidence().

    Returns:
        CitationResult with validity, cited IDs, invalid IDs, and warnings.
    """
    cited = extract_citations(answer_text)
    valid_ids = {c["citation_id"] for c in citation_map}
    max_id = max(valid_ids) if valid_ids else 0

    result = CitationResult(cited_ids=cited)

    # Check for invalid citations
    invalid = [n for n in cited if n not in valid_ids]
    if invalid:
        result.invalid_ids = invalid
        result.valid = False
        result.warnings.append(
            f"回答中引用了不存在的证据编号: {invalid}。"
            f"有效编号范围: 1-{max_id}。"
        )

    # Warn if no citations at all
    if not cited:
        result.valid = False
        result.warnings.append("回答中没有使用任何引用标记 [N]。可能包含未经证实的论断。")

    # Warn if many evidence chunks were not cited (low coverage)
    cited_set = set(cited)
    uncited = valid_ids - cited_set
    if uncited and len(uncited) == len(valid_ids) and cited:
        # Only warn if some evidence was used but others completely ignored
        pass  # This is fine — the LLM chose the most relevant chunks
    elif len(uncited) > len(valid_ids) * 0.7:
        result.warnings.append(
            f"大部分证据未被引用（{len(uncited)}/{len(valid_ids)} 个未使用）。"
        )

    return result
