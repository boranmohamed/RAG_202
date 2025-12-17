"""
Data Transfer Object for OCR Full Pipeline Request.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractOcrFullRequestDTO:
    """
    Request DTO for complete OCR pipeline (Extract + Preprocess + Chunk).
    
    This is used for the unified /process-ocr endpoint that does everything in one call.
    
    Following Yearbook 2025 rules: character-based limits (NOT token-based).
    """
    
    pdf_path: str
    document_name: str = "Statistical Year Book 2025"
    year: int = 2024
    max_chars: int = 1500  # Maximum characters per chunk (~1500 per language block)
    min_chars: int = 50  # Minimum characters per chunk
    overlap_chars: int = 100  # Overlap between chunks in characters
    ocr_dpi: int = 300  # OCR-specific parameter
    ocr_threshold: int = 50  # Minimum chars to consider page has embedded text
    max_pages: int | None = None  # Limit processing to first N pages

