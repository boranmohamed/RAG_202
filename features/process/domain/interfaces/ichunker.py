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
        max_tokens: int = 350,
        min_tokens: int = 120,
        overlap_tokens: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Chunk normalized blocks from Phase 2 into semantic chunks.
        
        Args:
            blocks: List of normalized blocks with bilingual content
            document_name: Document name for metadata
            year: Year for metadata
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk
            overlap_tokens: Overlap between chunks
        
        Returns:
            List of validated chunks ready for Phase 4 (embeddings)
        """
        raise NotImplementedError

