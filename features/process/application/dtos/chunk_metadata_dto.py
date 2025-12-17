"""
DTO for chunk metadata (Rule 11).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkMetadataDTO:
    """Metadata for a chunk (Rule 11)."""
    document: str
    year: int
    chapter: Optional[str]
    section: Optional[str]
    page_start: int
    page_end: int
    chunk_type: str  # "narrative" | "table" | "glossary" | etc.
    language: List[str]
    embedding_allowed: bool  # NEW: Explicit embedding control

