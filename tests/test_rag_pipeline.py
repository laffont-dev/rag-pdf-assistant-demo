"""Integration tests for the full RAG pipeline (embeddings + vector store)."""

from rag.embedder import Embedder
from rag.vector_store import VectorStore


def test_pipeline_retrieves_most_relevant_chunk():
    """
    Create 3 chunks, build the index, and search.
    The top-1 result should be the most relevant chunk.
    """
    embedder = Embedder()
    chunks = [
        {"text": "Le siège social est à Paris, France.", "page": 1, "source": "doc.pdf", "chunk_id": 0},
        {"text": "La politique de remboursement est de 30 jours.", "page": 2, "source": "doc.pdf", "chunk_id": 1},
        {"text": "Le contact technique est Jean Dupont.", "page": 3, "source": "doc.pdf", "chunk_id": 2},
    ]

    vs = VectorStore()
    vs.build(chunks, embedder)

    results = vs.search("Où se trouve le siège social ?", embedder, k=1)

    assert len(results) > 0, "Should find at least one result"
    assert "Paris" in results[0]["text"], (
        f"Expected result about Paris, got: {results[0]['text']}"
    )
    assert results[0]["page"] == 1


def test_relevance_threshold_filters_out_dissimilar_chunks():
    """
    A query very dissimilar to all chunks should return an empty list
    because the best score exceeds the 1.5 threshold.
    """
    embedder = Embedder()
    chunks = [
        {"text": "Le chat dort sur le canapé.", "page": 1, "source": "doc.pdf", "chunk_id": 0},
        {"text": "Il fait beau aujourd'hui à Paris.", "page": 2, "source": "doc.pdf", "chunk_id": 1},
        {"text": "La recette du gâteau au chocolat.", "page": 3, "source": "doc.pdf", "chunk_id": 2},
    ]

    vs = VectorStore()
    vs.build(chunks, embedder)

    # A completely unrelated query
    results = vs.search("quantum physics string theory black hole", embedder, k=3)

    assert len(results) == 0, (
        f"Expected empty results for dissimilar query, got {len(results)} results"
    )