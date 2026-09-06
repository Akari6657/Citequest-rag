"""
Pydantic data models for CiteQuest-RAG.

These schemas are the type contract shared across all modules:
- ingestion produces Paper and Chunk objects
- retrieval consumes them and produces SearchResult objects
- API layer receives SearchRequest and returns SearchResponse
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.core.config import DEFAULT_RAG_TOP_K


# ---------------------------------------------------------------------------
# Data models (ingestion output)
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    """A single academic paper with its metadata."""

    paper_id: str = Field(..., description="Stable unique ID (e.g. OpenAlex W-identifier)")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(default="", description="Reconstructed abstract text (may be empty)")
    year: int | None = Field(default=None, description="Publication year")
    venue: str | None = Field(default=None, description="Journal or conference name")
    authors: list[str] = Field(default_factory=list, description="Author display names in order")
    concepts: list[str] = Field(default_factory=list, description="Top-level concept labels")
    doi: str | None = Field(default=None, description="DOI identifier")
    url: str | None = Field(default=None, description="Landing page URL")
    citation_count: int = Field(default=0, ge=0, description="Number of citations")
    open_access: bool = Field(default=False, description="Whether the paper is open-access")


class Chunk(BaseModel):
    """A searchable text segment belonging to a paper."""

    chunk_id: str = Field(..., description="Unique chunk ID (e.g. '<paper_id>_default')")
    paper_id: str = Field(..., description="FK to papers.paper_id")
    chunk_text: str = Field(..., description="Full text of this chunk for indexing and display")
    chunk_type: Literal["metadata", "abstract", "full_text"] = Field(
        default="metadata",
        description="Source of this chunk text",
    )
    token_count: int = Field(default=0, ge=0, description="Approximate token count")


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Incoming search request (POST /search)."""

    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    mode: Literal["lexical", "vector", "hybrid"] = Field(
        default="lexical",
        description="Keyword, semantic, or weighted Hybrid search",
    )
    year_from: int | None = Field(
        default=None,
        description="Optional inclusive earliest publication year",
    )
    year_to: int | None = Field(
        default=None,
        description="Optional inclusive latest publication year",
    )
    alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional Hybrid weight: 0=pure semantic, 1=pure keyword; "
            "when omitted, the server runtime default is used"
        ),
    )
    include_overview: bool = Field(
        default=False,
        description="Whether to generate an AI Overview alongside search results",
    )

    @model_validator(mode="after")
    def validate_year_range(self) -> SearchRequest:
        """Reject an inverted inclusive publication-year range."""
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to")
        return self


class SearchResult(BaseModel):
    """A single search result."""

    paper_id: str
    chunk_id: str
    title: str
    year: int | None
    venue: str | None
    authors: list[str] = Field(default_factory=list)
    score: float = Field(
        ...,
        description=(
            "Retriever-specific relevance score (higher is better within the "
            "returned ranking; not calibrated across modes or queries)"
        ),
    )
    snippet: str = Field(default="", description="Relevant text excerpt from the chunk")
    abstract: str = Field(default="", description="Paper abstract (first ~300 chars)")


class SearchResponse(BaseModel):
    """Complete search response (POST /search)."""

    query: str
    mode: str
    effective_alpha: float | None = Field(
        default=None,
        description="Resolved Hybrid weight, or null when Hybrid retrieval was not used",
    )
    total_results: int
    results: list[SearchResult]
    ai_overview: AskResponse | None = Field(default=None, description="AI-generated overview (only when requested)")
    latency_ms: float = Field(default=0.0, description="Server-side processing time in milliseconds")
    should_rag: bool = Field(default=False, description="Router decision: whether this query should trigger RAG")
    rag_reason: str = Field(default="", description="Human-readable routing reason")
    rewrite_keywords: str = Field(default="", description="English keywords extracted from a Chinese query, if any")


# ---------------------------------------------------------------------------
# RAG / Ask models (v0.3+)
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Incoming RAG question (POST /ask)."""

    question: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(default=DEFAULT_RAG_TOP_K, ge=1, le=20, description="Maximum candidate evidence chunks before budget packing")
    retrieval_mode: Literal["lexical", "vector", "hybrid"] = Field(
        default="hybrid",
        description="Search mode for evidence retrieval",
    )
    alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional Hybrid weight; when omitted, the server runtime default is used"
        ),
    )
    pre_retrieved: list[dict] | None = Field(
        default=None,
        description=(
            "Pre-retrieved results from /search. Skips internal retrieval, but "
            "chunk existence and paper ownership are checked against SQLite; "
            "evidence text and citation metadata always come from the database."
        ),
    )


class CitationInfo(BaseModel):
    """Metadata for one citation marker."""

    citation_id: int
    paper_id: str
    chunk_id: str
    title: str
    url: str | None = None


class AskResponse(BaseModel):
    """Complete RAG answer response (POST /ask)."""

    question: str
    answer: str
    effective_alpha: float | None = Field(
        default=None,
        description="Resolved Hybrid weight, or null when Hybrid retrieval was not used",
    )
    citations: list[CitationInfo] = Field(default_factory=list)
    citation_valid: bool = True
    citation_warnings: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, description="End-to-end processing time in milliseconds")
