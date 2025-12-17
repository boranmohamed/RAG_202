"""
FastAPI routes for the PDF processing feature.

Feature: synchronous "extract + Arabic-aware preprocess" for a given on-disk PDF.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
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
async def extract_pdf(pdf_file: UploadFile = File(...)) -> ExtractPdfResponse:
    """
    Extract and normalize PDF content from uploaded file.
    
    Accepts a PDF file upload and processes it through:
      - Phase 1: Structured extraction (page/blocks/bbox, tables preserved)
      - Phase 2: Arabic/English normalization (no chunking or embedding)
    
    Args:
        pdf_file: PDF file to upload and process
        
    Returns:
        ExtractPdfResponse with normalized blocks
    """
    # Validate file type
    if not pdf_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create temporary file to save uploaded PDF
    temp_path = None
    try:
        # Read uploaded file content
        file_content = await pdf_file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # Process the PDF
        use_case = build_extract_pdf_use_case()
        dto_in = ExtractPdfRequestDTO(pdf_path=temp_path)

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
            pdf_path=pdf_file.filename,  # Return original filename instead of temp path
            pages=dto_out.pages,
            blocks=blocks,
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_path}: {e}")


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
    """Request for chunking normalized blocks (legacy - for direct block input)."""
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


def save_chunks_to_file(chunks: list[Chunk], document_name: str) -> str:
    """
    Save chunks to a txt file in chunks_output folder.
    
    Args:
        chunks: List of chunks to save
        document_name: Document name for filename generation
        
    Returns:
        Path to the saved file
    """
    # Create chunks_output folder if it doesn't exist
    output_dir = Path("chunks_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_doc_name = "".join(c for c in document_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_doc_name = safe_doc_name.replace(' ', '_')
    filename = f"{safe_doc_name}_{timestamp}.txt"
    filepath = output_dir / filename
    
    # Write chunks to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Chunks Output for: {document_name}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, chunk in enumerate(chunks, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CHUNK #{idx}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Chunk ID: {chunk.chunk_id}\n")
            f.write(f"Type: {chunk.metadata.chunk_type}\n")
            f.write(f"Document: {chunk.metadata.document}\n")
            f.write(f"Year: {chunk.metadata.year}\n")
            if chunk.metadata.chapter:
                f.write(f"Chapter: {chunk.metadata.chapter}\n")
            if chunk.metadata.section:
                f.write(f"Section: {chunk.metadata.section}\n")
            f.write(f"Pages: {chunk.metadata.page_start} - {chunk.metadata.page_end}\n")
            f.write(f"Languages: {', '.join(chunk.metadata.language)}\n")
            f.write(f"Embedding Allowed: {chunk.metadata.embedding_allowed}\n")
            f.write(f"\n--- Arabic Content ---\n")
            if chunk.content.ar:
                f.write(f"{chunk.content.ar}\n")
            else:
                f.write("(No Arabic content)\n")
            f.write(f"\n--- English Content ---\n")
            if chunk.content.en:
                f.write(f"{chunk.content.en}\n")
            else:
                f.write("(No English content)\n")
            f.write("\n")
    
    return str(filepath)


def build_chunking_use_case() -> ChunkingUseCase:
    """Build chunking use case with ChunkWise chunker implementation."""
    chunker = ChunkWiseBilingualChunker()
    return ChunkingUseCase(chunker=chunker)


@router.post("/chunk", response_model=ChunkingResponse)
async def chunk_blocks(
    pdf_file: UploadFile = File(...),
    document_name: str = Form("Statistical Year Book 2025"),
    year: int = Form(2024),
    max_tokens: int = Form(350),
    min_tokens: int = Form(120),
    overlap_tokens: int = Form(40),
) -> ChunkingResponse:
    """
    Extract and chunk PDF in one step.
    
    Accepts a PDF file upload, extracts it, and chunks it automatically.
    Chunks are saved to chunks_output folder.
    
    Pipeline:
    1. Phase 1: Structured extraction (page/blocks/bbox, tables preserved)
    2. Phase 2: Arabic/English normalization
    3. Phase 3: Chunking with strict rules enforcement
    
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
    
    Args:
        pdf_file: PDF file to upload and process
        document_name: Name of the document
        year: Year of the document
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk
        overlap_tokens: Token overlap between chunks
        
    Returns:
        ChunkingResponse with validated chunks ready for embedding
    """
    # Validate file type
    if not pdf_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create temporary file to save uploaded PDF
    temp_path = None
    try:
        # Read uploaded file content
        file_content = await pdf_file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # Step 1: Extract PDF (Phase 1 + Phase 2)
        extract_use_case = build_extract_pdf_use_case()
        extract_dto_in = ExtractPdfRequestDTO(pdf_path=temp_path)
        
        try:
            extract_dto_out: ExtractPdfResponseDTO = extract_use_case.execute(extract_dto_in)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}") from e
        
        # Step 2: Chunk the extracted blocks (Phase 3)
        chunking_use_case = build_chunking_use_case()
        chunking_dto_in = ChunkingRequestDTO(
            blocks=extract_dto_out.blocks,
            document_name=document_name,
            year=year,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens,
        )
        
        try:
            chunking_dto_out: ChunkingResponseDTO = chunking_use_case.execute(chunking_dto_in)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chunking failed: {e}") from e
        
        # Convert DTOs to API models
        chunks = []
        for chunk_dto in chunking_dto_out.chunks:
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
        
        # Save chunks to file automatically
        try:
            saved_path = save_chunks_to_file(chunks, document_name)
            print(f"Chunks saved to: {saved_path}")
        except Exception as e:
            # Log error but don't fail the request
            print(f"Warning: Failed to save chunks to file: {e}")
        
        return ChunkingResponse(
            chunks=chunks,
            total_chunks=chunking_dto_out.total_chunks,
            narrative_chunks=chunking_dto_out.narrative_chunks,
            table_chunks=chunking_dto_out.table_chunks,
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_path}: {e}")


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
    
    # Save chunks to file
    try:
        saved_path = save_chunks_to_file(chunks, request.document_name)
        print(f"Chunks saved to: {saved_path}")
    except Exception as e:
        # Log error but don't fail the request
        print(f"Warning: Failed to save chunks to file: {e}")
    
    return ChunkingResponse(
        chunks=chunks,
        total_chunks=dto_out.total_chunks,
        narrative_chunks=dto_out.narrative_chunks,
        table_chunks=dto_out.table_chunks,
    )



