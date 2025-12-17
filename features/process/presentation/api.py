"""
FastAPI routes for the PDF processing feature.

Feature: synchronous "extract + Arabic-aware preprocess" for a given on-disk PDF.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from features.process.application.dtos import (
    ExtractPdfRequestDTO,
    ExtractPdfResponseDTO,
    ExtractOcrFullRequestDTO,
    ChunkDTO,
    ChunkMetadataDTO,
    ChunkingRequestDTO,
    ChunkingResponseDTO,
    NormalizedBlockDTO,
)
from features.process.application.use_cases import (
    ExtractPdfUseCase,
    ChunkingUseCase,
    ExtractOcrFullPipelineUseCase,
)
from features.process.infrastructure.phase3_chunker_bilingual import ChunkWiseBilingualChunker


router = APIRouter(prefix="/api/v1/pdfs", tags=["pdfs"])


class ProcessPdfRequest(BaseModel):
    pdf_path: str


class BilingualContent(BaseModel):
    """Bilingual content structure: Arabic and English together."""
    ar: str | None = None
    en: str | None = None


class NormalizedBlock(BaseModel):
    """Single normalized block with bilingual content structure."""
    pageNumber: int
    type: str
    content: BilingualContent  # {"ar": "...", "en": "..."}
    chapter: str | None = None
    section: str | None = None


class ExtractPdfResponse(BaseModel):
    pdf_path: str
    pages: int
    blocks: list[NormalizedBlock]


def build_extract_pdf_use_case() -> ExtractPdfUseCase:
    """Build PDF extraction and normalization use case."""
    return ExtractPdfUseCase()


@router.post("/extract", response_model=ExtractPdfResponse)
def extract_pdf(request: ProcessPdfRequest) -> ExtractPdfResponse:
    """
    Extract and normalize PDF content.
    
    Runs:
      - Phase 1: Structured extraction (page/blocks/bbox, tables preserved)
      - Phase 2: Arabic/English normalization (no chunking or embedding)
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")

    use_case = build_extract_pdf_use_case()
    dto_in = ExtractPdfRequestDTO(pdf_path=request.pdf_path)

    try:
        dto_out: ExtractPdfResponseDTO = use_case.execute(dto_in)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}") from e

    # Convert DTOs to Pydantic models
    blocks = []
    for b in dto_out.blocks:
        blocks.append(
            NormalizedBlock(
                pageNumber=b.pageNumber,
                type=b.type,
                content=BilingualContent(ar=b.content.get("ar"), en=b.content.get("en")),
                chapter=b.chapter,
                section=b.section,
            )
        )
    
    return ExtractPdfResponse(
        pdf_path=dto_out.pdf_path,
        pages=dto_out.pages,
        blocks=blocks,
    )


class ChunkMetadata(BaseModel):
    """Metadata for a chunk."""
    document: str
    year: int
    chapter: str | None = None
    section: str | None = None
    page_start: int
    page_end: int
    chunk_type: str
    language: list[str]
    embedding_allowed: bool  # NEW: Explicit embedding control


class Chunk(BaseModel):
    """Single chunk ready for embedding."""
    chunk_id: str
    content: BilingualContent
    metadata: ChunkMetadata


class ChunkingRequest(BaseModel):
    """Request for chunking normalized blocks."""
    blocks: list[NormalizedBlock]  # From extraction pipeline
    document_name: str = "Statistical Year Book 2025"
    year: int = 2024
    max_tokens: int = 350
    min_tokens: int = 120
    overlap_tokens: int = 40


class ChunkingResponse(BaseModel):
    """Response from chunking pipeline."""
    chunks: list[Chunk]
    total_chunks: int
    narrative_chunks: int
    table_chunks: int


def build_chunking_use_case() -> ChunkingUseCase:
    """Build chunking use case with ChunkWise chunker implementation."""
    chunker = ChunkWiseBilingualChunker()
    return ChunkingUseCase(chunker=chunker)


@router.post("/chunk", response_model=ChunkingResponse)
def chunk_blocks(request: ChunkingRequest) -> ChunkingResponse:
    """
    Chunk normalized blocks with strict rules enforcement.
    
    Takes normalized blocks from extraction pipeline and produces validated chunks
    ready for embedding.
    
    Rules enforced:
    - Never chunk across sections
    - Titles are metadata only
    - TOC/index pages never chunked
    - One chunk = one meaning
    - Arabic + English stay together
    - Sentence boundaries are sacred
    - Tables are isolated
    - No duplicate content
    - No broken text
    - Metadata required
    """
    use_case = build_chunking_use_case()
    
    # Convert API models to DTOs
    blocks_dto = []
    for block in request.blocks:
        blocks_dto.append(
            NormalizedBlockDTO(
                pageNumber=block.pageNumber,
                type=block.type,
                content={
                    "ar": block.content.ar,
                    "en": block.content.en,
                },
                chapter=block.chapter,
                section=block.section,
            )
        )
    
    dto_in = ChunkingRequestDTO(
        blocks=blocks_dto,
        document_name=request.document_name,
        year=request.year,
        max_tokens=request.max_tokens,
        min_tokens=request.min_tokens,
        overlap_tokens=request.overlap_tokens,
    )

    try:
        dto_out: ChunkingResponseDTO = use_case.execute(dto_in)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {e}") from e
    
    # Convert DTOs to API models
    chunks = []
    for chunk_dto in dto_out.chunks:
        chunks.append(
            Chunk(
                chunk_id=chunk_dto.chunk_id,
                content=BilingualContent(
                    ar=chunk_dto.content.get("ar"),
                    en=chunk_dto.content.get("en"),
                ),
                metadata=ChunkMetadata(
                    document=chunk_dto.metadata.document,
                    year=chunk_dto.metadata.year,
                    chapter=chunk_dto.metadata.chapter,
                    section=chunk_dto.metadata.section,
                    page_start=chunk_dto.metadata.page_start,
                    page_end=chunk_dto.metadata.page_end,
                    chunk_type=chunk_dto.metadata.chunk_type,
                    language=chunk_dto.metadata.language,
                    embedding_allowed=chunk_dto.metadata.embedding_allowed,  # NEW field
                ),
            )
        )
    
    return ChunkingResponse(
        chunks=chunks,
        total_chunks=dto_out.total_chunks,
        narrative_chunks=dto_out.narrative_chunks,
        table_chunks=dto_out.table_chunks,
    )


# ==================== OCR Pipeline Endpoint ====================


class ProcessOcrRequest(BaseModel):
    """Request for unified OCR pipeline (Extract + Preprocess + Chunk)."""
    pdf_path: str
    document_name: str = "Statistical Year Book 2025"
    year: int = 2024
    max_tokens: int = 350
    min_tokens: int = 120
    overlap_tokens: int = 40
    ocr_dpi: int = 300
    ocr_threshold: int = 50
    max_pages: int | None = None


def build_ocr_full_pipeline_use_case() -> ExtractOcrFullPipelineUseCase:
    """Build OCR full pipeline use case with ChunkWise chunker."""
    chunker = ChunkWiseBilingualChunker()
    return ExtractOcrFullPipelineUseCase(chunker=chunker)


@router.post("/process-ocr", response_model=ChunkingResponse)
def process_pdf_with_ocr(request: ProcessOcrRequest) -> ChunkingResponse:
    """
    ONE unified endpoint: Extract → Preprocess → Chunk OCR-based PDFs.
    
    Uses pdfplumber + Tesseract OCR for scanned/image-based documents.
    Returns RAG-ready chunks in a single call.
    
    Pipeline:
    1. Phase 1 OCR: Extract with pdfplumber (embedded text when available, OCR for scanned pages)
    2. Phase 2 OCR: Normalize with OCR-specific fixes (ligature artifacts, Arabic paragraph merging)
    3. Phase 3: Chunk with ChunkWise rules (table atomicity, bilingual support, token-based sizing)
    
    Args:
        request: OCR processing request with PDF path and parameters
        
    Returns:
        ChunkingResponse with RAG-ready chunks containing:
        - Bilingual content (Arabic + English separated)
        - Metadata (chapter, section, pages, language)
        - Validated structure (no broken text, complete tables)
        
    OCR Parameters:
        - ocr_dpi: DPI for OCR rendering (300=default, 400-600 for small text/diacritics)
        - ocr_threshold: Min chars to consider page has embedded text (default 50)
        - max_pages: Limit processing to first N pages (None = all pages)
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")

    use_case = build_ocr_full_pipeline_use_case()
    dto_in = ExtractOcrFullRequestDTO(
        pdf_path=request.pdf_path,
        document_name=request.document_name,
        year=request.year,
        max_tokens=request.max_tokens,
        min_tokens=request.min_tokens,
        overlap_tokens=request.overlap_tokens,
        ocr_dpi=request.ocr_dpi,
        ocr_threshold=request.ocr_threshold,
        max_pages=request.max_pages,
    )

    try:
        dto_out: ChunkingResponseDTO = use_case.execute(dto_in)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR pipeline failed: {e}"
        ) from e

    # Convert DTOs to API models
    chunks = []
    for chunk_dto in dto_out.chunks:
        chunks.append(
            Chunk(
                chunk_id=chunk_dto.chunk_id,
                content=BilingualContent(
                    ar=chunk_dto.content.get("ar"),
                    en=chunk_dto.content.get("en"),
                ),
                metadata=ChunkMetadata(
                    document=chunk_dto.metadata.document,
                    year=chunk_dto.metadata.year,
                    chapter=chunk_dto.metadata.chapter,
                    section=chunk_dto.metadata.section,
                    page_start=chunk_dto.metadata.page_start,
                    page_end=chunk_dto.metadata.page_end,
                    chunk_type=chunk_dto.metadata.chunk_type,
                    language=chunk_dto.metadata.language,
                    embedding_allowed=chunk_dto.metadata.embedding_allowed,
                ),
            )
        )
    
    return ChunkingResponse(
        chunks=chunks,
        total_chunks=dto_out.total_chunks,
        narrative_chunks=dto_out.narrative_chunks,
        table_chunks=dto_out.table_chunks,
    )



