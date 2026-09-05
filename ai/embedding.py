"""High-level embedding helper.

Topic 3 uses embeddings as the *semantic* layer of the two-stage dedup
pipeline (cheap URL/hash check first, embedding cosine second).
"""

from __future__ import annotations

import numpy as np

from ai.providers.base import EmbeddingProvider
from ai.providers.factory import get_embedder


def embed(text: str, *, embedder: EmbeddingProvider | None = None) -> np.ndarray:
    """Return a unit-normalized embedding vector for `text`.

    Raises
    ------
    ValueError
        If `text` is empty after stripping.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    embedder = embedder or get_embedder()
    return embedder.embed(text)
