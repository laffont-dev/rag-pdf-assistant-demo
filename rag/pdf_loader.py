"""PDF loader module using PyMuPDF (fitz)."""

import fitz
import os


def load_pdf(path: str) -> list[dict]:
    """
    Load a PDF file and return its pages as a list of dicts.

    Args:
        path: Path to the PDF file.

    Returns:
        List of dicts with keys: page (int), text (str), source (str).
        Pages with fewer than 10 characters are ignored.
    """
    docs = []
    source = os.path.basename(path)

    with fitz.open(path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text().strip()
            if len(text) >= 10:
                docs.append({
                    "page": page_num + 1,
                    "text": text,
                    "source": source
                })

    return docs
