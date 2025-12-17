"""
DTO for chunking request.
"""

from dataclasses import dataclass
from typing import List

from .normalized_block_dto import NormalizedBlockDTO


@dataclass
class ChunkingRequestDTO:
    """Input for chunking normalized blocks."""
    blocks: List[NormalizedBlockDTO]  # From extraction pipeline
    document_name: str = "Statistical Year Book 2025"
    year: int = 2024
    max_tokens: int = 350
    min_tokens: int = 120
    overlap_tokens: int = 40

