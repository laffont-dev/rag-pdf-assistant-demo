"""Tests for the chunker module."""

from rag.chunker import chunk_documents


def test_chunking_produces_valid_chunks():
    """A 1000-word text should produce non-empty chunks, each < 600 words."""
    words = ["word"] * 1000
    text = " ".join(words)

    docs = [{"page": 1, "text": text, "source": "test.pdf"}]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)

    assert len(chunks) > 0, "Should produce at least one chunk"
    for chunk in chunks:
        chunk_word_count = len(chunk["text"].split())
        assert chunk_word_count <= 600, (
            f"Chunk has {chunk_word_count} words, expected <= 600"
        )
        # Check metadata
        assert "text" in chunk
        assert "page" in chunk
        assert "source" in chunk
        assert "chunk_id" in chunk

    # Check that chunk_ids are sequential
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks))), "chunk_ids should be sequential"


def test_overlap_between_consecutive_chunks():
    """Consecutive chunks should share overlapping words."""
    # Create text with distinguishable words
    text = " ".join([f"unique_word_{i}" for i in range(200)])

    docs = [{"page": 1, "text": text, "source": "test.pdf"}]
    chunks = chunk_documents(docs, chunk_size=80, overlap=20)

    if len(chunks) >= 2:
        # The end of the first chunk should overlap with the start of the second
        first_words = chunks[0]["text"].split()
        second_words = chunks[1]["text"].split()

        overlap_words = set(first_words) & set(second_words)
        assert len(overlap_words) > 0, (
            "There should be overlapping words between consecutive chunks"
        )


def test_empty_document():
    """An empty document should produce no chunks."""
    docs = [{"page": 1, "text": "", "source": "test.pdf"}]
    chunks = chunk_documents(docs)
    assert len(chunks) == 0


def test_short_document():
    """A document shorter than chunk_size should produce one chunk."""
    docs = [{"page": 1, "text": "hello world", "source": "test.pdf"}]
    chunks = chunk_documents(docs, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"