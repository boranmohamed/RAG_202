"""
Interface for chunking normalized blocks into semantic chunks.

This port defines the contract for chunking operations in Phase 3.
Infrastructure adapters (e.g., ChunkWiseBilingualChunker) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IChunker(ABC):
    """Port for chunking normalized blocks into semantic chunks."""
    
    @abstractmethod
    def chunk_blocks(
        self,
        blocks: List[Dict[str, Any]],
        document_name: str = "Statistical Year Book 2025",
        year: int = 2024,
        max_chars: int = 1500,
        min_chars: int = 50,
        overlap_chars: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Chunk normalized blocks from Phase 2 into semantic chunks.
        
        Following Yearbook 2025 rules:
        - Character-based limits (NOT token-based)
        - ~1500 characters per language block
        - Semantic boundaries (headings, sections, chapters)
        - NEVER cut tables, bilingual paragraphs, Arabic sentences mid-way
        
        Args:
            blocks: List of normalized blocks with bilingual content
            document_name: Document name for metadata
            year: Year for metadata
            max_chars: Maximum characters per chunk (~1500 per language block)
            min_chars: Minimum characters per chunk
            overlap_chars: Overlap between chunks in characters
        
        Returns:
            List of validated chunks ready for Phase 4 (embeddings)
        """
        raise NotImplementedError

