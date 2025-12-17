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

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

import pdfplumber
from PIL import Image

# Setup logger for this module
logger = logging.getLogger(__name__)

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
        Dict with keys: page_number, source, ar, en, mixed, used_ocr
    """
    # Try embedded text first
    embedded = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
    embedded = embedded.strip()
    logger.debug(f"extract_text_or_ocr: Page {page.page_number}: Embedded text length = {len(embedded)}")

    # Heuristic: if embedded text is tiny, page might be scanned or mostly images/tables
    use_ocr = len(embedded) < ocr_threshold

    if use_ocr:
        logger.info(f"extract_text_or_ocr: Page {page.page_number}: Using OCR (embedded={len(embedded)} < threshold={ocr_threshold})")
        # Render page to image and OCR it
        try:
            pil_img = page.to_image(resolution=ocr_dpi).original
            raw = ocr_image(pil_img, lang="ara+eng", psm=6)
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            logger.debug(f"extract_text_or_ocr: Page {page.page_number}: OCR extracted {len(lines)} lines")
        except Exception as e:
            logger.error(f"extract_text_or_ocr: Page {page.page_number}: OCR failed: {e}", exc_info=True)
            lines = []
    else:
        logger.debug(f"extract_text_or_ocr: Page {page.page_number}: Using embedded text ({len(embedded)} chars)")
        # Use embedded text
        lines = [ln.strip() for ln in embedded.splitlines() if ln.strip()]

    # Split by language
    ar_lines, en_lines, mixed_lines = split_text_by_language("\n".join(lines))
    logger.debug(f"extract_text_or_ocr: Page {page.page_number}: Language split - ar={len(ar_lines)}, en={len(en_lines)}, mixed={len(mixed_lines)}")

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
        "used_ocr": use_ocr,
    }


def extract_tables_best(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    """
    Extract tables with two configurations and keep both outputs.
    
    Uses pdfplumber's table extraction methods reliably:
    - Primary: page.extract_tables() - extracts all tables on the page
    - Fallback: page.extract_table() - extracts first table if extract_tables() fails
    
    Strategy 1: Lines-based (works when PDF has ruling lines)
    Strategy 2: Text-based (works when no ruling lines)
    Strategy 3: Explicit lines fallback (if lines fail, use explicit vertical/horizontal lines)
    
    Both strategies are tried; downstream can score them and pick best.
    
    Args:
        page: pdfplumber page object
        
    Returns:
        List of table dicts with keys: page_number, mode, table_index, rows
    """
    settings_lines = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "intersection_tolerance": 5,
        "keep_blank_chars": True,  # Preserve empty cells for better table structure
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
        "keep_blank_chars": True,  # Preserve empty cells for better table structure
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    }

    out: List[Dict[str, Any]] = []

    # Strategy 1: Lines-based (primary)
    tables_lines: List[List[List[Optional[str]]]] = []
    try:
        tables_lines = page.extract_tables(table_settings=settings_lines) or []
        logger.debug(f"extract_tables_best: Page {page.page_number}: Lines strategy found {len(tables_lines)} tables")
    except Exception as e:
        logger.debug(f"extract_tables_best: Page {page.page_number}: Lines strategy failed: {e}")
        # If extract_tables() fails, try extract_table() as fallback (single table)
        try:
            single_table = page.extract_table(table_settings=settings_lines)
            if single_table:
                tables_lines = [single_table]
                logger.debug(f"extract_tables_best: Page {page.page_number}: Single table fallback succeeded")
        except Exception as e2:
            logger.debug(f"extract_tables_best: Page {page.page_number}: Single table fallback failed: {e2}")
            # If lines strategy fails, try explicit lines fallback
            try:
                # Get explicit vertical and horizontal lines from page
                vertical_lines = page.lines if hasattr(page, 'lines') else []
                horizontal_lines = page.horizontal_edges if hasattr(page, 'horizontal_edges') else []
                
                logger.debug(f"extract_tables_best: Page {page.page_number}: Trying explicit lines ({len(vertical_lines)} vertical, {len(horizontal_lines)} horizontal)")
                if vertical_lines or horizontal_lines:
                    settings_explicit = {
                        "explicit_vertical_lines": [line for line in vertical_lines] if vertical_lines else None,
                        "explicit_horizontal_lines": [line for line in horizontal_lines] if horizontal_lines else None,
                        "snap_tolerance": 3,
                        "intersection_tolerance": 5,
                        "keep_blank_chars": True,
                        "join_tolerance": 3,
                        "text_x_tolerance": 2,
                        "text_y_tolerance": 2,
                    }
                    # Remove None values
                    settings_explicit = {k: v for k, v in settings_explicit.items() if v is not None}
                    tables_lines = page.extract_tables(table_settings=settings_explicit) or []
                    logger.debug(f"extract_tables_best: Page {page.page_number}: Explicit lines strategy found {len(tables_lines)} tables")
            except Exception as e3:
                logger.debug(f"extract_tables_best: Page {page.page_number}: Explicit lines strategy failed: {e3}")
                tables_lines = []

    # Strategy 2: Text-based (fallback)
    tables_text: List[List[List[Optional[str]]]] = []
    try:
        tables_text = page.extract_tables(table_settings=settings_text) or []
        logger.debug(f"extract_tables_best: Page {page.page_number}: Text strategy found {len(tables_text)} tables")
    except Exception as e:
        logger.debug(f"extract_tables_best: Page {page.page_number}: Text strategy failed: {e}")
        try:
            single_table = page.extract_table(table_settings=settings_text)
            if single_table:
                tables_text = [single_table]
                logger.debug(f"extract_tables_best: Page {page.page_number}: Single table text fallback succeeded")
        except Exception as e2:
            logger.debug(f"extract_tables_best: Page {page.page_number}: Single table text fallback failed: {e2}")
            tables_text = []

    # Combine results, preferring lines-based
    all_tables = tables_lines if tables_lines else tables_text
    logger.debug(f"extract_tables_best: Page {page.page_number}: Combined {len(all_tables)} tables (using {'lines' if tables_lines else 'text'} strategy)")
    
    # NEW FALLBACK: preserve original PDF order even if no table detected
    if not all_tables:
        try:
            raw = page.extract_words(x_tolerance=3, y_tolerance=3)
            lines = reconstruct_table_like_layout(raw)
            if lines:
                all_tables = [lines]
                logger.info(f"extract_tables_best: Page {page.page_number}: Fallback textgrid reconstruction found {len(lines)} rows")
        except Exception as e:
            logger.debug(f"extract_tables_best: Page {page.page_number}: Fallback textgrid reconstruction failed: {e}")
    
    for idx, t in enumerate(all_tables):
        # Clean cells
        cleaned = []
        for row in t:
            if row is None:
                continue
            cleaned.append([("" if c is None else str(c).strip()) for c in row])

        if cleaned:
            mode_name = "lines" if tables_lines else "text"
            out.append({
                "page_number": page.page_number,
                "mode": mode_name,
                "table_index": idx,
                "rows": cleaned,
            })
            logger.debug(f"extract_tables_best: Page {page.page_number}, Table {idx}: Added ({len(cleaned)} rows, mode={mode_name})")
        else:
            logger.warning(f"extract_tables_best: Page {page.page_number}, Table {idx}: Skipped (empty after cleaning)")

    logger.info(f"extract_tables_best: Page {page.page_number}: Extracted {len(out)} valid tables")
    return out


def reconstruct_table_like_layout(words: List[Dict[str, Any]]) -> Optional[List[List[str]]]:
    """
    Reconstruct table-like layout from extracted words.
    
    Heuristic: group words by y position → produce rows.
    This is a fallback when pdfplumber fails to detect table structure.
    
    Args:
        words: List of word dicts from page.extract_words()
        
    Returns:
        List of rows (each row is a list of cell strings), or None if not table-like
    """
    if not words:
        return None
    
    rows = []
    current_row = []
    last_y = None
    
    # Sort words by y position (top to bottom), then x position (left to right)
    sorted_words = sorted(words, key=lambda w: (w.get("top", 0), w.get("left", 0)))
    
    for w in sorted_words:
        word_text = w.get("text", "").strip()
        if not word_text:
            continue
        
        current_y = w.get("top", 0)
        
        if last_y is None:
            last_y = current_y
        
        # If y position is close (within 3 pixels), same row
        if abs(current_y - last_y) <= 3:
            current_row.append(word_text)
        else:
            # New row
            if current_row:
                rows.append(current_row)
            current_row = [word_text]
            last_y = current_y
    
    # Add final row
    if current_row:
        rows.append(current_row)
    
    # Discard tiny non-table results
    # Must have at least 2 rows and at least one row with more than 2 cells
    if len(rows) >= 2 and any(len(r) > 2 for r in rows):
        return rows
    
    return None


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
    Extract chapter name from text (can be in headers or paragraph blocks).
    
    Looks for patterns like:
    - "الفصل 2 : المناخ" → "المناخ"
    - "Climate : Chapter 2" → "Climate"
    - "Chapter 2: Climate" → "Climate"
    - "Chapter 1 : General Information" → "General Information"
    """
    if not text:
        logger.debug("_parse_chapter_from_text (OCR): Empty text provided")
        return None
    
    # More flexible patterns to handle various formats
    patterns = [
        # Pattern 1: "Chapter N : Name" (English, most common format)
        # Matches: "Chapter 1 : General Information" → "General Information"
        r'Chapter\s*[\d]+\s*[:：]\s*([A-Za-z][^\t\n\r\d]*?)(?:\s*\d+|$|\n)',
        # Pattern 2: "الفصل N : Name" (Arabic)
        r'الفصل\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+?)(?:\s*\d+|$|\n)',
        # Pattern 3: "Name : Chapter N" (reverse order)
        r'([A-Za-z][^\t\n\r]+?)\s*[:：]\s*Chapter\s*[\d]+',
        # Pattern 4: Generic pattern as fallback
        r'(?:الفصل|Chapter)\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+?)(?:\s*\d+|$|\n)',
    ]
    
    for idx, pattern in enumerate(patterns, 1):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            chapter_name = match.group(1).strip()
            logger.debug(f"_parse_chapter_from_text (OCR): Pattern {idx} matched, raw: '{chapter_name}'")
            # Clean up: remove trailing numbers, page numbers, and extra whitespace
            chapter_name = re.sub(r'\s*\d+.*$', '', chapter_name)  # Remove trailing numbers and everything after
            chapter_name = re.sub(r'\s{2,}', ' ', chapter_name)  # Normalize whitespace
            chapter_name = chapter_name.strip()
            # Only return if we got a meaningful name (at least 3 characters, not just numbers/symbols)
            if chapter_name and len(chapter_name) >= 3 and re.search(r'[A-Za-z\u0600-\u06FF]', chapter_name):
                logger.info(f"_parse_chapter_from_text (OCR): Extracted chapter: '{chapter_name}'")
                return chapter_name
            else:
                logger.debug(f"_parse_chapter_from_text (OCR): Pattern {idx} matched but result invalid: '{chapter_name}'")
    
    logger.debug(f"_parse_chapter_from_text (OCR): No chapter pattern matched in text (first 100 chars): '{text[:100]}'")
    return None


def _parse_section_from_text(text: str) -> Optional[str]:
    """
    Extract section name from text (can be in headers or paragraph blocks).
    
    Looks for patterns like:
    - "القسم 1 : المناخ" → "المناخ"
    - "Section 1: Main Results" → "Main Results"
    - "Section 1 : Climate" → "Climate"
    """
    if not text:
        logger.debug("_parse_section_from_text (OCR): Empty text provided")
        return None
    
    patterns = [
        # Pattern 1: "Section N : Name" (English, most common format)
        # Matches: "Section 1 : Climate" → "Climate"
        r'Section\s*[\d]+\s*[:：]\s*([A-Za-z][^\t\n\r\d]*?)(?:\s*\d+|$|\n)',
        # Pattern 2: "القسم N : Name" (Arabic)
        r'القسم\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+?)(?:\s*\d+|$|\n)',
        # Pattern 3: "Name : Section N" (reverse order)
        r'([A-Za-z][^\t\n\r]+?)\s*[:：]\s*Section\s*[\d]+',
        # Pattern 4: Generic pattern as fallback
        r'(?:القسم|Section)\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+?)(?:\s*\d+|$|\n)',
    ]
    
    for idx, pattern in enumerate(patterns, 1):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            section_name = match.group(1).strip()
            logger.debug(f"_parse_section_from_text (OCR): Pattern {idx} matched, raw: '{section_name}'")
            # Clean up: remove trailing numbers, page numbers, and extra whitespace
            section_name = re.sub(r'\s*\d+.*$', '', section_name)  # Remove trailing numbers and everything after
            section_name = re.sub(r'\s{2,}', ' ', section_name)  # Normalize whitespace
            section_name = section_name.strip()
            # Only return if we got a meaningful name (at least 3 characters, not just numbers/symbols)
            if section_name and len(section_name) >= 3 and re.search(r'[A-Za-z\u0600-\u06FF]', section_name):
                logger.info(f"_parse_section_from_text (OCR): Extracted section: '{section_name}'")
                return section_name
            else:
                logger.debug(f"_parse_section_from_text (OCR): Pattern {idx} matched but result invalid: '{section_name}'")
    
    logger.debug(f"_parse_section_from_text (OCR): No section pattern matched in text (first 100 chars): '{text[:100]}'")
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
    logger.info(f"extract_blocks_ocr: Starting OCR extraction from '{pdf_path}' (max_pages={max_pages}, ocr_dpi={ocr_dpi}, ocr_threshold={ocr_threshold})")
    page_extractions: List[PageExtraction] = []
    
    # Track chapter/section across pages
    current_chapter: Optional[str] = None
    current_section: Optional[str] = None

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        total_pages = len(pages)
        logger.info(f"extract_blocks_ocr: PDF has {total_pages} pages to process")

        for page_num, page in enumerate(pages, start=1):
            logger.debug(f"extract_blocks_ocr: Processing page {page_num}/{total_pages}")
            page_blocks: List[PageBlock] = []
            page_height = page.height or 792  # Default letter size height
            
            # 1) Extract tables first
            tables = extract_tables_best(page)
            logger.debug(f"extract_blocks_ocr: Page {page_num}: Found {len(tables)} tables")
            
            for table_idx, table_dict in enumerate(tables):
                # Convert to markdown for storage
                md = table_to_markdown(table_dict["rows"])
                logger.debug(f"extract_blocks_ocr: Page {page_num}, Table {table_idx}: Converted to markdown ({len(md)} chars)")
                
                # Store both markdown and raw rows in bilingual format
                # Store markdown in BOTH ar and en to avoid being filtered by chunker
                # Many chunker rules check if en_text exists
                content = BilingualContent(
                    ar=md,  # Store markdown in ar field
                    en=md   # Store markdown also in english to avoid dropping
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
            used_ocr = text_result.get("used_ocr", False)
            
            if used_ocr:
                logger.info(f"extract_blocks_ocr: Page {page_num}: Used OCR (ar_len={len(ar_text)}, en_len={len(en_text)}, mixed_len={len(mixed_text)})")
            else:
                logger.debug(f"extract_blocks_ocr: Page {page_num}: Used embedded text (ar_len={len(ar_text)}, en_len={len(en_text)}, mixed_len={len(mixed_text)})")
            
            # Parse chapter/section from any text (update tracking)
            # Check both combined text and individual language texts for better detection
            all_text = f"{ar_text} {en_text} {mixed_text}"
            page_chapter = _parse_chapter_from_text(all_text) or _parse_chapter_from_text(en_text) or _parse_chapter_from_text(ar_text)
            page_section = _parse_section_from_text(all_text) or _parse_section_from_text(en_text) or _parse_section_from_text(ar_text)
            
            if page_chapter:
                logger.info(f"extract_blocks_ocr: Page {page_num}: Found chapter '{page_chapter}'")
                current_chapter = page_chapter
            if page_section:
                logger.info(f"extract_blocks_ocr: Page {page_num}: Found section '{page_section}'")
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
                
                # Also try to extract chapter/section from this specific block's content
                block_text = (en_text or "") + " " + (ar_text or "")
                block_chapter = _parse_chapter_from_text(block_text) or current_chapter
                block_section = _parse_section_from_text(block_text) or current_section
                
                # Update current if found in this block
                if block_chapter and block_chapter != current_chapter:
                    logger.info(f"extract_blocks_ocr: Page {page_num}: Block-level chapter update to '{block_chapter}'")
                    current_chapter = block_chapter
                    current_section = None  # Reset section on new chapter
                if block_section and block_section != current_section:
                    logger.info(f"extract_blocks_ocr: Page {page_num}: Block-level section update to '{block_section}'")
                    current_section = block_section
                
                page_blocks.append(PageBlock(
                    type=block_type,
                    content=content,
                    bbox=None,  # Full page text from OCR doesn't have bbox
                    chapter=current_chapter,
                    section=current_section,
                ))
                logger.debug(f"extract_blocks_ocr: Page {page_num}: Added {block_type} block (chapter='{current_chapter}', section='{current_section}')")
            
            # 4) Handle mixed lines separately if any
            if mixed_text:
                content = BilingualContent(ar=mixed_text, en=None)
                
                # Also try to extract chapter/section from mixed text
                block_chapter = _parse_chapter_from_text(mixed_text) or current_chapter
                block_section = _parse_section_from_text(mixed_text) or current_section
                
                # Update current if found in this block
                if block_chapter and block_chapter != current_chapter:
                    logger.info(f"extract_blocks_ocr: Page {page_num}: Mixed-text block chapter update to '{block_chapter}'")
                    current_chapter = block_chapter
                    current_section = None  # Reset section on new chapter
                if block_section and block_section != current_section:
                    logger.info(f"extract_blocks_ocr: Page {page_num}: Mixed-text block section update to '{block_section}'")
                    current_section = block_section
                
                page_blocks.append(PageBlock(
                    type="paragraph",
                    content=content,
                    bbox=None,
                    chapter=current_chapter,
                    section=current_section,
                ))
                logger.debug(f"extract_blocks_ocr: Page {page_num}: Added mixed-text paragraph block ({len(mixed_text)} chars)")
            
            # Create PageExtraction for this page
            blocks_with_chapter = sum(1 for b in page_blocks if b.chapter)
            blocks_with_section = sum(1 for b in page_blocks if b.section)
            logger.info(f"extract_blocks_ocr: Page {page_num}: Created {len(page_blocks)} blocks ({blocks_with_chapter} with chapter, {blocks_with_section} with section)")
            page_extractions.append(PageExtraction(
                pageNumber=page.page_number,
                blocks=page_blocks
            ))

    logger.info(f"extract_blocks_ocr: OCR extraction complete - {len(page_extractions)} pages extracted")
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

