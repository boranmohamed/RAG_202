"""
DTOs (Data Transfer Objects) used by the PDF processing use cases and API.

Following clean architecture principles:
- DTOs are organized by feature/domain
- Each DTO is in its own file for better organization
- Easy to find and maintain specific DTOs
"""

# Extraction & Normalization DTOs
from .extract_pdf_request_dto import ExtractPdfRequestDTO
from .normalized_block_dto import NormalizedBlockDTO
from .extract_pdf_response_dto import ExtractPdfResponseDTO

# OCR Pipeline DTOs
from .extract_ocr_full_request_dto import ExtractOcrFullRequestDTO

# Chunking DTOs
from .chunk_metadata_dto import ChunkMetadataDTO
from .chunk_dto import ChunkDTO
from .chunking_request_dto import ChunkingRequestDTO
from .chunking_response_dto import ChunkingResponseDTO

__all__ = [
    # Extraction & Normalization
    "ExtractPdfRequestDTO",
    "NormalizedBlockDTO",
    "ExtractPdfResponseDTO",
    # OCR Pipeline
    "ExtractOcrFullRequestDTO",
    # Chunking
    "ChunkMetadataDTO",
    "ChunkDTO",
    "ChunkingRequestDTO",
    "ChunkingResponseDTO",
]
