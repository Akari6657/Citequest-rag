"""Shared test doubles for the real FTS5/FAISS smoke tests."""

import pytest


@pytest.fixture
def tiny_embedding_model():
    """Return a deterministic model class for both indexing and retrieval."""
    import numpy as np

    concepts = ("retrieval", "vision", "code", "database", "graph", "robot")

    class TinyEmbeddingModel:
        dim = len(concepts)

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, *, show_progress=False, **kwargs):
            vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
            for row, text in enumerate(texts):
                lowered = text.lower()
                for column, concept in enumerate(concepts):
                    if concept in lowered:
                        vectors[row, column] = 1.0
                if not vectors[row].any():
                    vectors[row, -1] = 1.0
                vectors[row] /= np.linalg.norm(vectors[row])
            return vectors

    return TinyEmbeddingModel
