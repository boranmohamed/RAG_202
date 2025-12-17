"""
DTO for a single normalized block from extraction pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class NormalizedBlockDTO:
    """
    Single normalized block with bilingual content.
    
    Arabic and English are kept together in the same semantic unit.
    """

    pageNumber: int
    type: str
    content: Dict[str, Optional[str]]  # {"ar": "...", "en": "..."}
    chapter: Optional[str] = None  # Chapter name from Phase 1
    section: Optional[str] = None  # Section name from Phase 1

