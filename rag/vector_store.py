"""Vector store using FAISS for similarity search."""

import faiss
import numpy as np


class VectorStore:
    """
    FAISS-based vector store for document chunk retrieval.

    Builds an index from chunk embeddings, then searches by query.
    """

    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict], embedder) -> None:
        """
        Build the FAISS index from a list of chunks.

        Args:
            chunks: List of chunk dicts with at least a "text" key.
            embedder: An Embedder instance with an embed() method.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype(np.float32))

    def search(
        self,
        query: str,
        embedder,
        k: int = 3
    ) -> list[dict]:
        """
        Search for the top-k most relevant chunks.

        Args:
            query: The search query string.
            embedder: An Embedder instance with an embed() method.
            k: Number of results to return.

        Returns:
            List of result dicts with keys: text, page, source, score.
            Returns empty list if the best score exceeds the relevance threshold (1.5).
        """
        query_vec = embedder.embed([query])
        distances, indices = self.index.search(
            query_vec.astype(np.float32), k
        )

        best_score = distances[0][0]
        if best_score > 1.5:
            return []

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
                "score": float(distances[0][i])
            })

        return results
