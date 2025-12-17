"""
DTO for PDF extraction and normalization request.
"""

from dataclasses import dataclass


@dataclass
class ExtractPdfRequestDTO:
    """
    Input for PDF extraction (Phase 1) and normalization (Phase 2) pipeline.
    """

    pdf_path: str

