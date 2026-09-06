"""Runtime configuration for CiteQuest-RAG.

The default app uses ``data/indexes``.  Environment variables let us point the
same API at a smaller demo index without changing code or rebuilding the whole
local benchmark database.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

DEFAULT_DB_PATH = Path("data/indexes/metadata.sqlite")
DEFAULT_FAISS_DIR = Path("data/indexes/faiss")
DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_REWRITE_TIMEOUT_SECONDS = 2.0
DEFAULT_RAG_CONTEXT_TOKENS = 8000
APP_VERSION = "0.5.0"

DB_PATH_ENV = "CITEQUEST_DB_PATH"
FAISS_DIR_ENV = "CITEQUEST_FAISS_DIR"
HYBRID_ALPHA_ENV = "CITEQUEST_HYBRID_ALPHA"
REWRITE_TIMEOUT_ENV = "CITEQUEST_REWRITE_TIMEOUT_SECONDS"
RAG_CONTEXT_TOKENS_ENV = "CITEQUEST_RAG_CONTEXT_TOKENS"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser()


def get_db_path() -> Path:
    """Return the metadata SQLite path for the current process."""
    return _path_from_env(DB_PATH_ENV, DEFAULT_DB_PATH)


def get_faiss_dir() -> Path:
    """Return the FAISS artifact directory for the current process."""
    return _path_from_env(FAISS_DIR_ENV, DEFAULT_FAISS_DIR)


def validate_hybrid_alpha(value: object, *, source: str = "alpha") -> float:
    """Return a finite Hybrid weight in the inclusive range [0, 1]."""
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} must be a finite number between 0 and 1; got {value!r}"
        ) from exc

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"{source} must be a finite number between 0 and 1; got {value!r}"
        )
    return alpha


def get_hybrid_alpha() -> float:
    """Return the configured production default for Hybrid retrieval."""
    value = os.getenv(HYBRID_ALPHA_ENV)
    if value is None:
        return DEFAULT_HYBRID_ALPHA
    return validate_hybrid_alpha(value, source=HYBRID_ALPHA_ENV)


def get_rewrite_timeout_seconds() -> float:
    """Return the time budget for optional LLM query rewriting."""
    value = os.getenv(REWRITE_TIMEOUT_ENV)
    if value is None:
        return DEFAULT_REWRITE_TIMEOUT_SECONDS

    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{REWRITE_TIMEOUT_ENV} must be a finite number greater than 0; "
            f"got {value!r}"
        ) from exc

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            f"{REWRITE_TIMEOUT_ENV} must be a finite number greater than 0; "
            f"got {value!r}"
        )
    return timeout


def resolve_hybrid_alpha(request_alpha: float | None) -> float:
    """Resolve request override first, then runtime configuration."""
    if request_alpha is None:
        return get_hybrid_alpha()
    return validate_hybrid_alpha(request_alpha, source="request alpha")


def validate_rag_context_tokens(value: object, *, source: str = "max_tokens") -> int:
    """Validate the positive integer budget for estimated evidence tokens."""
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError
        tokens = int(value)
        if tokens <= 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"{source} must be a positive integer; got {value!r}") from exc
    return tokens


def get_rag_context_tokens() -> int:
    """Return the runtime budget for evidence, excluding prompts and output."""
    return validate_rag_context_tokens(
        os.getenv(RAG_CONTEXT_TOKENS_ENV, DEFAULT_RAG_CONTEXT_TOKENS),
        source=RAG_CONTEXT_TOKENS_ENV,
    )
