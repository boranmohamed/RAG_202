"""
DTO for chunking request.
"""

from dataclasses import dataclass
from typing import List, Optional

from .normalized_block_dto import NormalizedBlockDTO


@dataclass
class ChunkingRequestDTO:
    """
    Input for chunking normalized blocks.
    
    Following Yearbook 2025 rules: character-based limits (NOT token-based).
    """
    blocks: List[NormalizedBlockDTO]  # From extraction pipeline
    document_name: str = "Statistical Year Book 2025"
    year: int = 2024
    max_chars: int = 1500  # Maximum characters per chunk (~1500 per language block)
    min_chars: int = 50  # Minimum characters per chunk
    overlap_chars: int = 100  # Overlap between chunks in characters
    pdf_path: Optional[str] = None  # Optional PDF path for table serialization with pdfplumber

