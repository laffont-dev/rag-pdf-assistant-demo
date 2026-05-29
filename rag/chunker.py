"""Text chunker for splitting documents into overlapping chunks."""


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split document pages into overlapping chunks based on word count.

    Args:
        docs: List of page dicts from pdf_loader.load_pdf.
        chunk_size: Target number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of chunk dicts with keys: text, page, source, chunk_id.
    """
    chunks = []
    chunk_id = 0

    for page_doc in docs:
        words = page_doc["text"].split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "page": page_doc["page"],
                "source": page_doc["source"],
                "chunk_id": chunk_id
            })
            chunk_id += 1

            # Move start forward, accounting for overlap
            # If we're at the end of the page, stop
            if end >= len(words):
                break

            start = end - overlap

    return chunks
