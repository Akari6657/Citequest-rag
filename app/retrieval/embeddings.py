"""
Embedding pipeline: encode chunk texts into dense vectors.

Uses BAAI/bge-m3 — 1024-dimensional embeddings with native multi-language
support and 8192 token input limit. Much better for academic long-text retrieval.

Vectors are L2-normalized so FAISS IndexFlatIP (inner product) gives
cosine similarity.

Usage:
    from app.retrieval.embeddings import EmbeddingModel

    model = EmbeddingModel()
    vectors = model.encode(["text 1", "text 2", ...])  # shape (N, 1024)
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Small protocol shared by SentenceTransformer and test doubles."""

    def get_embedding_dimension(self) -> int:
        ...

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        ...

# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
"""Default embedding model. 1024 dims, ~2.2 GB on first download. 8192 token max input."""

# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Thin wrapper around sentence-transformers with L2 normalization.

    The model is loaded once and cached.  Call encode() for batched inference.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model: EmbeddingBackend | None = None):
        self._model_name = model_name
        self._model: EmbeddingBackend | None = model

    @property
    def model(self) -> EmbeddingBackend:
        if self._model is None:
            logger.info("Loading embedding model: %s ...", self._model_name)
            # Lazy import — sentence-transformers is heavy
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info("Model loaded. dim=%d", self.dim)
        return self._model

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self.model.get_embedding_dimension()

    def encode(self, texts: list[str], *, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts into a (N, dim) float32 array.

        Args:
            texts: List of chunk texts to encode.
            batch_size: Batch size for the encoder.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            numpy array of shape (len(texts), dim), L2-normalized so that
            inner product == cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        logger.info("Encoding %d texts ...", len(texts))
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 norm → cosine via Inner Product
            convert_to_numpy=True,
        )

        logger.info("Encoded %d vectors, shape=%s", len(texts), vectors.shape)
        return np.asarray(vectors, dtype=np.float32)
