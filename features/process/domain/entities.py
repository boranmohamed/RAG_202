"""
Domain entities for the PDF processing feature.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PdfDocument:
    """Represents a source PDF to process."""

    path: str
    pages: Optional[int] = None


@dataclass
class ProcessedText:
    """Represents extracted and preprocessed text."""

    raw_text: str
    preprocessed_text: str
    total_chars: int
    arabic_chars: int
    english_words: int
    pages: int
    output_path: Optional[str] = None



