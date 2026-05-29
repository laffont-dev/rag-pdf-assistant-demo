"""Tests for the embedder module."""

import numpy as np
from rag.embedder import Embedder


def test_embedding_dimension():
    """Embedding of a single phrase should have dimension 384."""
    embedder = Embedder()
    vec = embedder.embed(["Bonjour le monde"])
    assert vec.shape == (1, 384), (
        f"Expected shape (1, 384), got {vec.shape}"
    )


def test_similar_phrases_high_cosine():
    """Two very similar phrases should have cosine similarity > 0.7."""
    embedder = Embedder()
    vecs = embedder.embed([
        "Le chat dort sur le canapé",
        "Le chat est en train de dormir sur le canapé",
    ])
    v1 = vecs[0]
    v2 = vecs[1]
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert cosine > 0.7, f"Cosine similarity {cosine} <= 0.7"


def test_no_nan_or_inf():
    """Embedding vectors should not contain NaN or Inf values."""
    embedder = Embedder()
    vecs = embedder.embed([
        "Une phrase en français",
        "Another sentence in English",
        "Python est un langage de programmation",
    ])
    assert not np.any(np.isnan(vecs)), "Vectors contain NaN"
    assert not np.any(np.isinf(vecs)), "Vectors contain Inf"