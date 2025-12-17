"""
DTO for chunk metadata (Rule 11).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


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
    # Table-specific metadata
    units: Optional[List[str]] = None  # Measurement units (e.g., ["000 MT", "000 BBL"])
    data_year: Optional[List[int]] = None  # Years covered by data (e.g., [2024, 2023])
    geography: Optional[List[str]] = None  # Geographic areas (e.g., ["Muscat Governorate"])
    bilingual_alignment: Optional[Dict[str, str]] = None  # Arabic -> English term mapping

