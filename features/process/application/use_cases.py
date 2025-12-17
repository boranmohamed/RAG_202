"""
Application use cases for the PDF processing feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from features.process.infrastructure.pdf_structured_extractor_pymupdf import (
    extract_structured_pdf,
)
from features.process.infrastructure.phase2_preprocessor_bilingual import (
    normalize_phase2_from_pages,
)
from features.process.infrastructure.phase1_ocr_extractor_pdfplumber import (
    extract_structured_pdf_ocr,
)
from features.process.infrastructure.phase2_ocr_preprocessor import (
    normalize_ocr_blocks,
)
from features.process.domain.interfaces import IChunker
from .dtos import (
    NormalizedBlockDTO,
    ExtractPdfRequestDTO,
    ExtractPdfResponseDTO,
    ExtractOcrFullRequestDTO,
    ChunkDTO,
    ChunkMetadataDTO,
    ChunkingRequestDTO,
    ChunkingResponseDTO,
)


@dataclass
class ExtractPdfUseCase:
    """
    PDF extraction and normalization pipeline.
    
    Phase 1: Structured extraction (page/blocks/bbox, tables preserved)
    Phase 2: Arabic/English normalization (no chunking or embedding)
    
    This does NOT chunk or embed; it only prepares normalized blocks.
    """

    def execute(self, request: ExtractPdfRequestDTO) -> ExtractPdfResponseDTO:
        # Phase 1: structured extraction
        structured_pages = extract_structured_pdf(request.pdf_path)

        # Phase 2: bilingual normalization
        normalized_blocks_dicts = normalize_phase2_from_pages(structured_pages)

        # Map into DTOs with bilingual content structure
        blocks: List[NormalizedBlockDTO] = []
        for b in normalized_blocks_dicts:
            # Extract content dict from normalized block
            content_dict = b.get("content", {})
            if not isinstance(content_dict, dict):
                # Fallback if content is not a dict
                content_dict = {"ar": None, "en": None}
            
            blocks.append(
                NormalizedBlockDTO(
                    pageNumber=int(b.get("pageNumber", 0)),
                    type=str(b.get("type", "")),
                    content={
                        "ar": content_dict.get("ar"),
                        "en": content_dict.get("en"),
                    },
                    chapter=b.get("chapter"),
                    section=b.get("section"),
                )
            )

        pages_count = max((blk.pageNumber for blk in blocks), default=0)

        return ExtractPdfResponseDTO(
            pdf_path=request.pdf_path,
            pages=pages_count,
            blocks=blocks,
        )


@dataclass
class ChunkingUseCase:
    """
    Semantic chunking with strict rules enforcement.
    
    Takes normalized blocks from extraction pipeline and produces validated chunks
    ready for embedding.
    
    Follows clean architecture: depends on IChunker interface, not concrete implementation.
    """

    chunker: IChunker

    def execute(self, request: ChunkingRequestDTO) -> ChunkingResponseDTO:
        # Convert DTOs to dicts for chunking function
        blocks_dicts = []
        for block_dto in request.blocks:
            blocks_dicts.append({
                "pageNumber": block_dto.pageNumber,
                "type": block_dto.type,
                "content": block_dto.content,
                "chapter": block_dto.chapter,
                "section": block_dto.section,
            })
        
        # Phase 3: Chunk with rules enforcement (via interface)
        chunks_dicts = self.chunker.chunk_blocks(
            blocks=blocks_dicts,
            document_name=request.document_name,
            year=request.year,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            overlap_tokens=request.overlap_tokens,
        )

        # Convert to DTOs
        chunks: List[ChunkDTO] = []
        narrative_count = 0
        table_count = 0
        
        for chunk_dict in chunks_dicts:
            metadata_dict = chunk_dict.get("metadata", {})
            metadata = ChunkMetadataDTO(
                document=metadata_dict.get("document", request.document_name),
                year=metadata_dict.get("year", request.year),
                chapter=metadata_dict.get("chapter"),
                section=metadata_dict.get("section"),
                page_start=metadata_dict.get("page_start", 0),
                page_end=metadata_dict.get("page_end", 0),
                chunk_type=metadata_dict.get("chunk_type", "narrative"),
                language=metadata_dict.get("language", []),
                embedding_allowed=metadata_dict.get("embedding_allowed", True),  # NEW field
            )
            
            chunk = ChunkDTO(
                chunk_id=chunk_dict.get("chunk_id", ""),
                content=chunk_dict.get("content", {}),
                metadata=metadata,
            )
            chunks.append(chunk)
            
            # Count by type
            if metadata.chunk_type == "narrative":
                narrative_count += 1
            elif metadata.chunk_type == "table":
                table_count += 1
        
        return ChunkingResponseDTO(
            chunks=chunks,
            total_chunks=len(chunks),
            narrative_chunks=narrative_count,
            table_chunks=table_count,
        )


@dataclass
class ExtractOcrFullPipelineUseCase:
    """
    Complete OCR pipeline: Phase 1 OCR + Phase 2 OCR + Phase 3 Chunking.
    
    All-in-one processing for scanned/image-based PDFs using pdfplumber + Tesseract OCR.
    Returns RAG-ready chunks in a single execution.
    
    Pipeline:
    1. Phase 1 OCR: Extract with pdfplumber (embedded text + OCR fallback)
    2. Phase 2 OCR: Normalize with OCR-specific fixes (ligatures, Arabic merging)
    3. Phase 3: Chunk with existing ChunkWise chunker (reuses existing infrastructure)
    
    Follows clean architecture: depends on IChunker interface, not concrete implementation.
    """

    chunker: IChunker

    def execute(self, request: ExtractOcrFullRequestDTO) -> ChunkingResponseDTO:
        """
        Execute complete OCR pipeline and return RAG-ready chunks.
        
        Args:
            request: OCR pipeline request with PDF path and chunking parameters
            
        Returns:
            ChunkingResponseDTO with validated chunks ready for embedding
        """
        # Phase 1 OCR: Extract with pdfplumber + Tesseract
        ocr_pages = extract_structured_pdf_ocr(
            request.pdf_path,
            max_pages=request.max_pages,
            ocr_dpi=request.ocr_dpi
        )
        
        # Phase 2 OCR: Normalize OCR artifacts and merge split paragraphs
        normalized_blocks_dicts = normalize_ocr_blocks(ocr_pages)
        
        # Phase 3: Chunk (reuse existing chunker via interface)
        chunks_dicts = self.chunker.chunk_blocks(
            blocks=normalized_blocks_dicts,
            document_name=request.document_name,
            year=request.year,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            overlap_tokens=request.overlap_tokens,
        )

        # Convert to DTOs
        chunks: List[ChunkDTO] = []
        narrative_count = 0
        table_count = 0
        
        for chunk_dict in chunks_dicts:
            metadata_dict = chunk_dict.get("metadata", {})
            metadata = ChunkMetadataDTO(
                document=metadata_dict.get("document", request.document_name),
                year=metadata_dict.get("year", request.year),
                chapter=metadata_dict.get("chapter"),
                section=metadata_dict.get("section"),
                page_start=metadata_dict.get("page_start", 0),
                page_end=metadata_dict.get("page_end", 0),
                chunk_type=metadata_dict.get("chunk_type", "narrative"),
                language=metadata_dict.get("language", []),
                embedding_allowed=metadata_dict.get("embedding_allowed", True),
            )
            
            chunk = ChunkDTO(
                chunk_id=chunk_dict.get("chunk_id", ""),
                content=chunk_dict.get("content", {}),
                metadata=metadata,
            )
            chunks.append(chunk)
            
            # Count by type
            if metadata.chunk_type == "narrative":
                narrative_count += 1
            elif metadata.chunk_type == "table":
                table_count += 1
        
        return ChunkingResponseDTO(
            chunks=chunks,
            total_chunks=len(chunks),
            narrative_chunks=narrative_count,
            table_chunks=table_count,
        )
