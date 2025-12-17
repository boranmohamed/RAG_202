"""
Phase 1 OCR — OCR-BASED EXTRACTION with pdfplumber + Tesseract

This module uses pdfplumber to extract PDFs with OCR support for scanned/image-based pages.

Features:
  - Extract embedded text from PDF when available
  - Detect scanned pages (insufficient text) and apply OCR
  - Use Tesseract with Arabic + English for bilingual OCR
  - Extract tables using dual-strategy (lines-based + text-based)
  - Separate Arabic and English content at extraction time
  - Return same structure as PyMuPDF Phase 1 (PageExtraction with bilingual blocks)

Processing Flow:
  1. Try to extract embedded text first (fast, high quality)
  2. If page has < 50 chars (likely scanned), render to image and OCR
  3. Separate Arabic/English text into bilingual structure
  4. Extract tables with two strategies and keep best results
  5. Return structured blocks compatible with Phase 2 preprocessor
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

import pdfplumber
from PIL import Image

from features.process.infrastructure.utils.ocr_utils import (
    detect_lang,
    normalize_arabic,
    normalize_english,
    ocr_image,
    split_text_by_language,
    table_to_markdown,
)


BlockType = Literal["header", "paragraph", "table", "toc", "footer"]


@dataclass
class BilingualContent:
    """Bilingual content structure: Arabic and English together."""
    ar: Optional[str] = None
    en: Optional[str] = None


@dataclass
class PageBlock:
    """Single block from OCR extraction with bilingual content."""
    type: BlockType
    content: BilingualContent
    bbox: Optional[List[float]]  # [x1, y1, x2, y2] or None for full-page OCR
    chapter: Optional[str] = None
    section: Optional[str] = None


@dataclass
class PageExtraction:
    """Complete page extraction with blocks."""
    pageNumber: int
    blocks: List[PageBlock]


def extract_text_or_ocr(
    page: pdfplumber.page.Page,
    ocr_dpi: int = 300,
    ocr_threshold: int = 50
) -> Dict[str, Any]:
    """
    Try embedded text first. If too little, render and OCR.
    
    Returns both Arabic + English separated streams when possible.
    
    Args:
        page: pdfplumber page object
        ocr_dpi: DPI for OCR rendering (higher = better quality, slower)
        ocr_threshold: Minimum chars to consider page has embedded text
        
    Returns:
        Dict with keys: page_number, source, ar, en, mixed
    """
    # Try embedded text first
    embedded = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
    embedded = embedded.strip()

    # Heuristic: if embedded text is tiny, page might be scanned or mostly images/tables
    use_ocr = len(embedded) < ocr_threshold

    if use_ocr:
        # Render page to image and OCR it
        pil_img = page.to_image(resolution=ocr_dpi).original
        raw = ocr_image(pil_img, lang="ara+eng", psm=6)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        # Use embedded text
        lines = [ln.strip() for ln in embedded.splitlines() if ln.strip()]

    # Split by language
    ar_lines, en_lines, mixed_lines = split_text_by_language("\n".join(lines))

    # Join and normalize
    ar_text = normalize_arabic("\n".join(ar_lines))
    en_text = normalize_english("\n".join(en_lines))
    mixed_text = "\n".join(mixed_lines).strip()

    return {
        "page_number": page.page_number,
        "source": "ocr" if use_ocr else "embedded",
        "ar": ar_text,
        "en": en_text,
        "mixed": mixed_text,
    }


def extract_tables_best(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    """
    Extract tables with two configurations and keep both outputs.
    
    Strategy 1: Lines-based (works when PDF has ruling lines)
    Strategy 2: Text-based (works when no ruling lines)
    
    Both are returned; downstream can score them and pick best.
    
    Args:
        page: pdfplumber page object
        
    Returns:
        List of table dicts with keys: page_number, mode, table_index, rows
    """
    settings_lines = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 8,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    }

    settings_text = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    }

    out: List[Dict[str, Any]] = []

    for mode_name, settings in [("lines", settings_lines), ("text", settings_text)]:
        try:
            tables = page.extract_tables(table_settings=settings) or []
        except Exception:
            tables = []

        for idx, t in enumerate(tables):
            # Clean cells
            cleaned = []
            for row in t:
                if row is None:
                    continue
                cleaned.append([("" if c is None else str(c).strip()) for c in row])

            if cleaned:
                out.append({
                    "page_number": page.page_number,
                    "mode": mode_name,
                    "table_index": idx,
                    "rows": cleaned,
                })

    return out


def _classify_block_type(
    text: str,
    page_height: float,
    y_position: Optional[float] = None,
    is_table: bool = False,
) -> BlockType:
    """
    Heuristic classification into header / paragraph / table / toc / footer.
    
    Args:
        text: Block text content
        page_height: Total page height
        y_position: Vertical position (None if unknown)
        is_table: Whether this block is a table
        
    Returns:
        Classified block type
    """
    if is_table:
        return "table"
    
    # TOC detection
    if text and re.search(r'(?:المحتويات|Contents|Table of Contents)', text, re.IGNORECASE):
        return "toc"
    
    # Position-based classification (if position known)
    if y_position is not None:
        if y_position <= page_height * 0.15:
            return "header"
        if y_position >= page_height * 0.85:
            return "footer"
    
    # Default
    return "paragraph"


def _parse_chapter_from_text(text: str) -> Optional[str]:
    """
    Extract chapter name from text.
    
    Looks for patterns like:
    - "الفصل 2 : المناخ" → "المناخ"
    - "Climate : Chapter 2" → "Climate"
    - "Chapter 2: Climate" → "Climate"
    """
    if not text:
        return None
    
    patterns = [
        r'(?:الفصل|Chapter)\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+)',
        r'([^\t\n\r]+?)\s*[:：]\s*(?:الفصل|Chapter)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            chapter_name = match.group(1).strip()
            chapter_name = re.sub(r'\s*\d+.*$', '', chapter_name)
            chapter_name = re.sub(r'\s{2,}', ' ', chapter_name)
            return chapter_name.strip() if chapter_name else None
    
    return None


def _parse_section_from_text(text: str) -> Optional[str]:
    """
    Extract section name from text.
    
    Looks for patterns like:
    - "القسم 1 : المناخ" → "المناخ"
    - "Section 1: Main Results" → "Main Results"
    """
    if not text:
        return None
    
    patterns = [
        r'(?:القسم|Section)\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+)',
        r'([^\t\n\r]+?)\s*[:：]\s*(?:القسم|Section)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            section_name = match.group(1).strip()
            section_name = re.sub(r'\s*\d+.*$', '', section_name)
            section_name = re.sub(r'\s{2,}', ' ', section_name)
            return section_name.strip() if section_name else None
    
    return None


def extract_blocks_ocr(
    pdf_path: str,
    max_pages: Optional[int] = None,
    ocr_dpi: int = 300,
    ocr_threshold: int = 50
) -> List[PageExtraction]:
    """
    Extract PDF blocks using pdfplumber + OCR hybrid approach.
    
    This is the main entry point for Phase 1 OCR extraction.
    Returns PageExtraction objects compatible with Phase 2 preprocessor.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Limit processing to first N pages (None = all pages)
        ocr_dpi: DPI for OCR rendering (300 recommended, 400-600 for small text)
        ocr_threshold: Minimum chars to consider page has embedded text
        
    Returns:
        List of PageExtraction objects (one per page)
    """
    page_extractions: List[PageExtraction] = []
    
    # Track chapter/section across pages
    current_chapter: Optional[str] = None
    current_section: Optional[str] = None

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages

        for page in pages:
            page_blocks: List[PageBlock] = []
            page_height = page.height or 792  # Default letter size height
            
            # 1) Extract tables first
            tables = extract_tables_best(page)
            
            for table_dict in tables:
                # Convert to markdown for storage
                md = table_to_markdown(table_dict["rows"])
                
                # Store both markdown and raw rows in bilingual format
                # Tables don't have language separation - store as Arabic content
                content = BilingualContent(
                    ar=md,  # Store markdown in ar field
                    en=None
                )
                
                page_blocks.append(PageBlock(
                    type="table",
                    content=content,
                    bbox=None,  # pdfplumber doesn't provide table bbox easily
                    chapter=current_chapter,
                    section=current_section,
                ))
            
            # 2) Extract text (embedded or OCR)
            text_result = extract_text_or_ocr(
                page,
                ocr_dpi=ocr_dpi,
                ocr_threshold=ocr_threshold
            )
            
            ar_text = text_result["ar"]
            en_text = text_result["en"]
            mixed_text = text_result["mixed"]
            
            # Parse chapter/section from any text (update tracking)
            all_text = f"{ar_text} {en_text} {mixed_text}"
            page_chapter = _parse_chapter_from_text(all_text)
            page_section = _parse_section_from_text(all_text)
            
            if page_chapter:
                current_chapter = page_chapter
            if page_section:
                current_section = page_section
            
            # Determine block type based on content
            block_type = _classify_block_type(
                text=all_text,
                page_height=page_height,
                y_position=None,  # We don't have precise Y position from OCR
                is_table=False
            )
            
            # 3) Create bilingual paragraph block if we have text
            if ar_text or en_text:
                content = BilingualContent(ar=ar_text if ar_text else None, en=en_text if en_text else None)
                
                page_blocks.append(PageBlock(
                    type=block_type,
                    content=content,
                    bbox=None,  # Full page text from OCR doesn't have bbox
                    chapter=current_chapter,
                    section=current_section,
                ))
            
            # 4) Handle mixed lines separately if any
            if mixed_text:
                content = BilingualContent(ar=mixed_text, en=None)
                
                page_blocks.append(PageBlock(
                    type="paragraph",
                    content=content,
                    bbox=None,
                    chapter=current_chapter,
                    section=current_section,
                ))
            
            # Create PageExtraction for this page
            page_extractions.append(PageExtraction(
                pageNumber=page.page_number,
                blocks=page_blocks
            ))

    return page_extractions


def extract_structured_pdf_ocr(pdf_path: str, max_pages: Optional[int] = None, ocr_dpi: int = 300) -> List[Dict[str, Any]]:
    """
    Extract PDF using OCR and return as JSON-serializable dict.
    
    This is a convenience function that returns dicts instead of dataclasses.
    Compatible with Phase 2 preprocessor input format.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Limit processing to first N pages (None = all pages)
        ocr_dpi: DPI for OCR rendering
        
    Returns:
        List of page dicts with structure:
        {
            "pageNumber": int,
            "blocks": [
                {
                    "type": str,
                    "content": {"ar": str | None, "en": str | None},
                    "bbox": [float] | None,
                    "chapter": str | None,
                    "section": str | None
                }
            ]
        }
    """
    page_extractions = extract_blocks_ocr(pdf_path, max_pages=max_pages, ocr_dpi=ocr_dpi)
    
    # Convert to dicts
    result = []
    for page_ext in page_extractions:
        page_dict = {
            "pageNumber": page_ext.pageNumber,
            "blocks": []
        }
        
        for block in page_ext.blocks:
            block_dict = {
                "type": block.type,
                "content": {
                    "ar": block.content.ar,
                    "en": block.content.en,
                },
                "bbox": block.bbox,
                "chapter": block.chapter,
                "section": block.section,
            }
            page_dict["blocks"].append(block_dict)
        
        result.append(page_dict)
    
    return result

