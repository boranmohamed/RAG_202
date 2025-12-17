"""
DTO for PDF extraction and normalization response.
"""

from dataclasses import dataclass
from typing import List

from .normalized_block_dto import NormalizedBlockDTO


@dataclass
class ExtractPdfResponseDTO:
    """
    Output of PDF extraction and normalization pipeline: normalized blocks ready for chunking.
    """

    pdf_path: str
    pages: int
    blocks: List[NormalizedBlockDTO]

