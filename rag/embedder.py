"""Embedding module using sentence-transformers with lazy model loading."""

import numpy as np


class Embedder:
    """
    Text embedder using sentence-transformers.

    The model is loaded lazily on the first call to embed().
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load the sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into a numpy array of vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy.ndarray of shape (len(texts), embedding_dim).
        """
        if self._model is None:
            self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings
