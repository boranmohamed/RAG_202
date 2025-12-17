"""
DTO for a single chunk ready for embedding.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .chunk_metadata_dto import ChunkMetadataDTO


@dataclass
class ChunkDTO:
    """Single chunk ready for Phase 4 (embedding)."""
    chunk_id: str
    content: Dict[str, Optional[str]]  # {"ar": "...", "en": "..."}
    metadata: ChunkMetadataDTO

