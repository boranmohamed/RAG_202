"""
DTO for chunking response.
"""

from dataclasses import dataclass
from typing import List

from .chunk_dto import ChunkDTO


@dataclass
class ChunkingResponseDTO:
    """Output of chunking pipeline: validated chunks ready for embedding."""
    chunks: List[ChunkDTO]
    total_chunks: int
    narrative_chunks: int
    table_chunks: int

