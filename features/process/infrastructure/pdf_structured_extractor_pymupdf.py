"""
Phase 1 — PURE STRUCTURAL EXTRACTION (no preprocessing, no normalization).

This module uses PyMuPDF to extract a PDF into a page-by-page JSON-serializable
structure, preserving:
  - Page boundaries
  - Text blocks with bounding boxes
  - A coarse block type: header | paragraph | table | toc | footer
  - Bilingual semantic alignment (Arabic + English paired together)

IMPORTANT:
  - Arabic / English text is left AS-IS (no fixes, no normalization).
  - We do NOT merge content across pages.
  - Tables are kept as raw multi-line text blocks (one block per table).
  - Arabic and English blocks that represent the same semantic unit are paired together.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

import json

import fitz  # PyMuPDF

# Setup logger for this module
logger = logging.getLogger(__name__)


BlockType = Literal["header", "paragraph", "table", "toc", "footer"]


# Language detection regexes
ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
ENGLISH_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass
class BilingualContent:
    """Bilingual content structure: Arabic and English together."""
    ar: Optional[str] = None
    en: Optional[str] = None


@dataclass
class PageBlock:
    type: BlockType
    content: BilingualContent  # Changed from rawText to content with ar/en
    bbox: Optional[List[float]]  # [x1, y1, x2, y2]
    chapter: Optional[str] = None  # Chapter name extracted from headers
    section: Optional[str] = None  # Section name extracted from headers


@dataclass
class PageExtraction:
    pageNumber: int
    blocks: List[PageBlock]


def _classify_block_type(
    bbox: List[float],
    page_height: float,
    is_toc_page: bool,
    fallback_type: BlockType = "paragraph",
) -> BlockType:
    """
    Heuristic classification into header / paragraph / toc / footer.

    - TOC pages: everything is marked as "toc" to keep them easy to filter later.
    - Header: top ~15% of the page.
    - Footer: bottom ~15% of the page.
    - Otherwise: paragraph (or provided fallback_type, e.g. "table").
    """
    if is_toc_page:
        return "toc"

    y1 = bbox[1]

    if y1 <= page_height * 0.15:
        return "header"
    if y1 >= page_height * 0.85:
        return "footer"

    return fallback_type


def _parse_chapter_from_header(text: str) -> Optional[str]:
    """
    Extract chapter name from text (can be in headers or paragraph blocks).
    
    Looks for patterns like:
    - "الفصل 2 : المناخ" → "المناخ"
    - "Climate : Chapter 2" → "Climate"
    - "Chapter 2: Climate" → "Climate"
    - "Chapter 1 : General Information" → "General Information"
    """
    if not text:
        logger.debug("_parse_chapter_from_header: Empty text provided")
        return None
    
    # Arabic and English chapter patterns
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
            logger.debug(f"_parse_chapter_from_header: Pattern {idx} matched, raw: '{chapter_name}'")
            # Clean up: remove trailing numbers, page numbers, and extra whitespace
            chapter_name = re.sub(r'\s*\d+.*$', '', chapter_name)  # Remove trailing numbers and everything after
            chapter_name = re.sub(r'\s{2,}', ' ', chapter_name)  # Normalize whitespace
            chapter_name = chapter_name.strip()
            # Only return if we got a meaningful name (at least 3 characters, not just numbers/symbols)
            if chapter_name and len(chapter_name) >= 3 and re.search(r'[A-Za-z\u0600-\u06FF]', chapter_name):
                logger.info(f"_parse_chapter_from_header: Extracted chapter: '{chapter_name}'")
                return chapter_name
            else:
                logger.debug(f"_parse_chapter_from_header: Pattern {idx} matched but result invalid: '{chapter_name}'")
    
    logger.debug(f"_parse_chapter_from_header: No chapter pattern matched in text (first 100 chars): '{text[:100]}'")
    return None


def _parse_section_from_header(text: str) -> Optional[str]:
    """
    Extract section name from text (can be in headers or paragraph blocks).
    
    Looks for patterns like:
    - "القسم 1 : المناخ" → "المناخ"
    - "Section 1: Main Results" → "Main Results"
    - "Section 1 : Climate" → "Climate"
    """
    if not text:
        logger.debug("_parse_section_from_header: Empty text provided")
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
            logger.debug(f"_parse_section_from_header: Pattern {idx} matched, raw: '{section_name}'")
            # Clean up: remove trailing numbers, page numbers, and extra whitespace
            section_name = re.sub(r'\s*\d+.*$', '', section_name)  # Remove trailing numbers and everything after
            section_name = re.sub(r'\s{2,}', ' ', section_name)  # Normalize whitespace
            section_name = section_name.strip()
            # Only return if we got a meaningful name (at least 3 characters, not just numbers/symbols)
            if section_name and len(section_name) >= 3 and re.search(r'[A-Za-z\u0600-\u06FF]', section_name):
                logger.info(f"_parse_section_from_header: Extracted section: '{section_name}'")
                return section_name
            else:
                logger.debug(f"_parse_section_from_header: Pattern {idx} matched but result invalid: '{section_name}'")
    
    logger.debug(f"_parse_section_from_header: No section pattern matched in text (first 100 chars): '{text[:100]}'")
    return None


def _detect_block_language(text: str) -> Literal["ar", "en", "mixed"]:
    """
    Detect the primary language of a text block.
    - 'ar': primarily Arabic (has Arabic letters, no/minimal English)
    - 'en': primarily English (has English letters, no/minimal Arabic)
    - 'mixed': both languages present
    """
    if not text:
        return "mixed"
    
    ar_count = len(ARABIC_LETTER_RE.findall(text))
    en_count = len(ENGLISH_LETTER_RE.findall(text))
    
    if ar_count > 0 and en_count == 0:
        return "ar"
    if en_count > 0 and ar_count == 0:
        return "en"
    return "mixed"


def _are_blocks_horizontally_aligned(
    bbox1: List[float], bbox2: List[float], y_threshold: float = 10.0
) -> bool:
    """
    Check if two blocks are horizontally aligned (same semantic unit).
    
    Blocks are considered aligned if their Y coordinates overlap significantly.
    y_threshold: maximum vertical distance (in points) to consider aligned.
    """
    y1_center = (bbox1[1] + bbox1[3]) / 2  # center Y of block 1
    y2_center = (bbox2[1] + bbox2[3]) / 2  # center Y of block 2
    
    # Check if vertical overlap exists
    y1_top, y1_bottom = bbox1[1], bbox1[3]
    y2_top, y2_bottom = bbox2[1], bbox2[3]
    
    # Overlap exists if one block's vertical range intersects the other's
    has_overlap = not (y1_bottom < y2_top or y2_bottom < y1_top)
    
    # Also check if centers are close enough
    center_distance = abs(y1_center - y2_center)
    
    return has_overlap or center_distance <= y_threshold


def _pair_bilingual_blocks(blocks: List[PageBlock], page_width: float) -> List[PageBlock]:
    """
    Pair Arabic and English blocks that represent the same semantic unit.
    
    Strategy:
    1. Group blocks by vertical alignment (similar Y coordinates)
    2. Within each group, identify left (English) and right (Arabic) blocks
    3. Pair them together into bilingual content objects
    4. Unpaired blocks are kept as-is (with only one language)
    
    Args:
        blocks: List of PageBlock objects from extraction
        page_width: Width of the page (unused, kept for future use)
    """
    if not blocks:
        return []
    
    # Initialize paired_blocks list early
    paired_blocks: List[PageBlock] = []
    paired_indices = set()
    
    # Separate blocks by language
    ar_blocks: List[tuple[PageBlock, int]] = []  # (block, index)
    en_blocks: List[tuple[PageBlock, int]] = []
    mixed_blocks: List[tuple[PageBlock, int]] = []
    table_blocks: List[tuple[PageBlock, int]] = []
    
    for idx, block in enumerate(blocks):
        # Tables are kept separate and not paired
        if block.type == "table":
            table_blocks.append((block, idx))
            continue
        
        # Check if already paired (has both ar and en)
        if block.content.ar and block.content.en:
            # Already paired, keep as-is
            paired_blocks.append(block)
            paired_indices.add(idx)
            continue
        
        # Categorize by which language field has content
        if block.content.ar and not block.content.en:
            ar_blocks.append((block, idx))
        elif block.content.en and not block.content.ar:
            en_blocks.append((block, idx))
        else:
            # Mixed or empty - treat as mixed
            mixed_blocks.append((block, idx))
    
    # Pair Arabic and English blocks based on horizontal alignment
    
    for ar_block, ar_idx in ar_blocks:
        if ar_idx in paired_indices:
            continue
        
        # Find the best matching English block (horizontally aligned)
        best_en_match = None
        best_en_idx = None
        best_overlap = 0.0
        
        for en_block, en_idx in en_blocks:
            if en_idx in paired_indices:
                continue
            
            if not ar_block.bbox or not en_block.bbox:
                continue
            
            if _are_blocks_horizontally_aligned(ar_block.bbox, en_block.bbox):
                # Calculate horizontal position to determine left/right
                ar_center_x = (ar_block.bbox[0] + ar_block.bbox[2]) / 2
                en_center_x = (en_block.bbox[0] + en_block.bbox[2]) / 2
                
                # English should be on left, Arabic on right (for typical bilingual layout)
                # But we'll pair if they're aligned regardless
                overlap_score = 1.0 / (abs(ar_center_x - en_center_x) + 1.0)
                
                if overlap_score > best_overlap:
                    best_overlap = overlap_score
                    best_en_match = en_block
                    best_en_idx = en_idx
        
        # Create paired block
        if best_en_match and best_en_idx is not None:
            # Extract text from both blocks
            ar_text = ar_block.content.ar or ""
            en_text = best_en_match.content.en or ""
            
            # Use the type from the Arabic block (or English if Arabic is missing)
            block_type = ar_block.type if ar_block.type != "paragraph" else best_en_match.type
            
            # Use the bbox that encompasses both blocks
            combined_bbox = None
            if ar_block.bbox and best_en_match.bbox:
                combined_bbox = [
                    min(ar_block.bbox[0], best_en_match.bbox[0]),
                    min(ar_block.bbox[1], best_en_match.bbox[1]),
                    max(ar_block.bbox[2], best_en_match.bbox[2]),
                    max(ar_block.bbox[3], best_en_match.bbox[3]),
                ]
            
            paired_blocks.append(
                PageBlock(
                    type=block_type,
                    content=BilingualContent(ar=ar_text, en=en_text),
                    bbox=combined_bbox,
                )
            )
            paired_indices.add(ar_idx)
            paired_indices.add(best_en_idx)
        else:
            # Unpaired Arabic block - keep with Arabic only
            paired_blocks.append(ar_block)
            paired_indices.add(ar_idx)
    
    # Add unpaired English blocks
    for en_block, en_idx in en_blocks:
        if en_idx in paired_indices:
            continue
        paired_blocks.append(en_block)
        paired_indices.add(en_idx)
    
    # Add mixed blocks as-is (they already contain both languages in ar field)
    for mixed_block, mixed_idx in mixed_blocks:
        if mixed_idx in paired_indices:
            continue
        paired_blocks.append(mixed_block)
        paired_indices.add(mixed_idx)
    
    # Add table blocks as-is (tables are not paired)
    for table_block, _ in table_blocks:
        paired_blocks.append(table_block)
    
    return paired_blocks


def _is_toc_like_page(page: "fitz.Page") -> bool:
    """
    Very light heuristic for TOC / index-like pages.

    We look for:
      - Many lines that end with a page number.
      - Or common English / Arabic TOC markers.

    Text is NOT modified; this only inspects it.
    """
    text = page.get_text("text") or ""
    if not text:
        return False

    # Quick markers (English / Arabic)
    lowered = text.lower()
    if "table of contents" in lowered or "contents" in lowered:
        return True

    if "فهرس" in text or "المحتويات" in text:
        return True

    # Line-pattern heuristic: lines ending with digits (page numbers)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    toc_like = 0
    for ln in lines:
        # ends with 1–4 digits, optionally after dots/space
        stripped = ln.rstrip(". ")
        if stripped and stripped[-1:].isdigit():
            toc_like += 1

    # If a significant portion of lines look like TOC entries, label as TOC page
    if len(lines) >= 10 and toc_like / len(lines) >= 0.4:
        return True

    return False


def _extract_text_blocks(page: "fitz.Page", is_toc_page: bool) -> List[PageBlock]:
    """
    Extract text blocks with bounding boxes using PyMuPDF "blocks" layout mode.
    
    Initially creates blocks with content in the detected language only.
    Pairing will happen later to combine Arabic and English.
    """
    page_height = page.rect.height
    blocks_raw = page.get_text("blocks", sort=True) or []
    logger.debug(f"_extract_text_blocks: Found {len(blocks_raw)} raw blocks from PyMuPDF")

    blocks: List[PageBlock] = []
    skipped_count = 0

    for block in blocks_raw:
        # block format: (x0, y0, x1, y1, text, block_no, block_type, ...)
        if len(block) < 7:
            skipped_count += 1
            logger.debug(f"_extract_text_blocks: Skipping block with insufficient data (len={len(block)})")
            continue

        x0, y0, x1, y1, text, _block_no, block_type = block[:7]

        # 0 = text block according to PyMuPDF
        if block_type != 0:
            skipped_count += 1
            logger.debug(f"_extract_text_blocks: Skipping non-text block (type={block_type})")
            continue

        raw_text = (text or "").strip()
        if not raw_text:
            skipped_count += 1
            logger.debug(f"_extract_text_blocks: Skipping empty text block")
            continue

        bbox = [float(x0), float(y0), float(x1), float(y1)]
        block_kind = _classify_block_type(bbox, page_height, is_toc_page, fallback_type="paragraph")

        # Detect language and create bilingual content structure
        lang = _detect_block_language(raw_text)
        if lang == "ar":
            content = BilingualContent(ar=raw_text, en=None)
        elif lang == "en":
            content = BilingualContent(ar=None, en=raw_text)
        else:
            # Mixed: put in Arabic field for now, pairing logic will handle it
            content = BilingualContent(ar=raw_text, en=None)

        logger.debug(f"_extract_text_blocks: Created {block_kind} block (lang={lang}, text_len={len(raw_text)})")
        blocks.append(
            PageBlock(
                type=block_kind,
                content=content,
                bbox=bbox,
            )
        )

    logger.info(f"_extract_text_blocks: Extracted {len(blocks)} text blocks, skipped {skipped_count}")
    return blocks


def _extract_table_blocks(page: "fitz.Page", is_toc_page: bool) -> List[PageBlock]:
    """
    Extract table regions as separate blocks.

    - We rely on PyMuPDF's find_tables (if available).
    - Each table is converted into a multi-line text representation, preserving
      header + rows, without flattening into sentences.
    """
    page_height = page.rect.height

    tables: List[PageBlock] = []

    # PyMuPDF's table API is not available in very old versions, so guard it.
    if not hasattr(page, "find_tables"):
        return tables

    try:
        table_finder = page.find_tables()
    except Exception:
        # If table detection fails for any reason, just skip tables.
        return tables

    for table in getattr(table_finder, "tables", []):
        # table.bbox: Rect or tuple; table.extract() returns a dict-like with "cells"/"header"
        rect = getattr(table, "bbox", getattr(table, "rect", None))
        if rect is None:
            continue

        # Handle both Rect objects and tuples (x0, y0, x1, y1)
        if isinstance(rect, (tuple, list)) and len(rect) >= 4:
            bbox = [float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])]
        elif hasattr(rect, "x0") and hasattr(rect, "y0") and hasattr(rect, "x1") and hasattr(rect, "y1"):
            bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        else:
            # Skip if we can't parse the bbox
            continue

        # Extract raw cell text matrix
        raw_table_text = ""
        try:
            extracted = table.extract()  # API: returns list-of-lists or dict depending on version
            # Normalize to a simple list-of-lists of strings without touching content.
            rows: List[List[str]] = []

            if isinstance(extracted, list):
                # Already a matrix
                rows = [[str(cell) if cell is not None else "" for cell in row] for row in extracted]
            elif isinstance(extracted, dict):
                # Common layout: {"header": [...], "cells": [[...], ...]}
                header = extracted.get("header")
                cells = extracted.get("cells") or extracted.get("rows") or []

                if header:
                    rows.append([str(h) if h is not None else "" for h in header])

                for row in cells:
                    rows.append([str(cell) if cell is not None else "" for cell in row])

            # Build a raw multiline representation: one row per line, tab-separated cells
            lines: List[str] = []
            for row in rows:
                line = "\t".join(row)
                lines.append(line)

            raw_table_text = "\n".join(lines)
        except Exception:
            # If anything goes wrong, fall back to a simple region text extraction.
            try:
                # get_textbox accepts Rect objects or tuples
                raw_table_text = page.get_textbox(rect) or ""
            except Exception:
                # If even textbox extraction fails, skip this table
                continue

        raw_table_text = raw_table_text.strip()
        if not raw_table_text:
            continue

        block_kind = _classify_block_type(bbox, page_height, is_toc_page, fallback_type="table")
        # Tables are kept as-is in the 'ar' field (Phase 2 will handle normalization)
        tables.append(
            PageBlock(
                type=block_kind,
                content=BilingualContent(ar=raw_table_text, en=None),
                bbox=bbox,
            )
        )

    return tables


def extract_structured_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract the given PDF into a list of JSON-serializable page objects:

    [
      {
        "pageNumber": 1,
        "blocks": [
          {
            "type": "header" | "paragraph" | "table" | "toc" | "footer",
            "rawText": "...",
            "bbox": [x1, y1, x2, y2],
            "chapter": "...",  # Extracted from headers
            "section": "..."   # Extracted from headers
          },
          ...
        ]
      },
      ...
    ]
    """
    logger.info(f"extract_structured_pdf: Starting extraction from '{pdf_path}'")
    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        logger.info(f"extract_structured_pdf: PDF has {total_pages} pages")
        pages: List[Dict[str, Any]] = []
        
        # Track current chapter/section across pages
        current_chapter: Optional[str] = None
        current_section: Optional[str] = None

        for i, page in enumerate(doc, start=1):
            logger.debug(f"extract_structured_pdf: Processing page {i}/{total_pages}")
            is_toc = _is_toc_like_page(page)
            if is_toc:
                logger.debug(f"extract_structured_pdf: Page {i} identified as TOC page")

            text_blocks = _extract_text_blocks(page, is_toc_page=is_toc)
            table_blocks = _extract_table_blocks(page, is_toc_page=is_toc)
            logger.debug(f"extract_structured_pdf: Page {i} - {len(text_blocks)} text blocks, {len(table_blocks)} table blocks")

            # Do NOT merge pages; every page stands alone.
            all_blocks = text_blocks + table_blocks
            
            # Pair bilingual blocks (Arabic + English together)
            page_width = page.rect.width
            paired_blocks = _pair_bilingual_blocks(all_blocks, page_width)
            logger.debug(f"extract_structured_pdf: Page {i} - {len(paired_blocks)} blocks after pairing")
            
            # Two-pass approach for better chapter/section extraction:
            # Pass 1: Extract chapter/section from ALL blocks on this page
            # This ensures we find chapter/section even if they appear in paragraph blocks
            page_chapter: Optional[str] = None
            page_section: Optional[str] = None
            
            for block_idx, block in enumerate(paired_blocks):
                text = block.content.ar or block.content.en or ""
                if text:
                    chapter = _parse_chapter_from_header(text)
                    section = _parse_section_from_header(text)
                    
                    if chapter:
                        logger.info(f"extract_structured_pdf: Page {i}, Block {block_idx}: Found chapter '{chapter}'")
                        page_chapter = chapter
                        page_section = None  # Reset section on new chapter
                    elif section:
                        logger.info(f"extract_structured_pdf: Page {i}, Block {block_idx}: Found section '{section}'")
                        page_section = section
            
            # Update current chapter/section if found on this page
            if page_chapter:
                logger.info(f"extract_structured_pdf: Page {i}: Updating current_chapter to '{page_chapter}'")
                current_chapter = page_chapter
                current_section = None  # Reset section on new chapter
            if page_section:
                logger.info(f"extract_structured_pdf: Page {i}: Updating current_section to '{page_section}'")
                current_section = page_section
            
            # Pass 2: Propagate current chapter/section to ALL blocks on this page
            blocks_with_chapter = 0
            blocks_with_section = 0
            for block in paired_blocks:
                block.chapter = current_chapter
                block.section = current_section
                if block.chapter:
                    blocks_with_chapter += 1
                if block.section:
                    blocks_with_section += 1
            
            logger.info(f"extract_structured_pdf: Page {i}: {blocks_with_chapter} blocks with chapter, {blocks_with_section} blocks with section")

            page_extraction = PageExtraction(
                pageNumber=i,
                blocks=paired_blocks,
            )
            pages.append(asdict(page_extraction))

        logger.info(f"extract_structured_pdf: Extraction complete - {len(pages)} pages extracted")
        return pages
    except Exception as e:
        logger.error(f"extract_structured_pdf: Error during extraction: {e}", exc_info=True)
        raise
    finally:
        doc.close()


def save_structured_pdf(
    pdf_path: str,
    output_json_path: str,
    ensure_ascii: bool = False,
    indent: Optional[int] = 2,
) -> None:
    """
    Convenience helper: run extraction and save result to a JSON file.
    """
    data = extract_structured_pdf(pdf_path)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


if __name__ == "__main__":
    # Minimal CLI entry point for ad-hoc runs:
    #   python -m features.process.infrastructure.pdf_structured_extractor_pymupdf \
    #       publicationpdfar1765273617.pdf phase1_structured_output.json
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Phase 1 — structured PDF extraction (page + blocks + bbox, no preprocessing)."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the input PDF (e.g. Statistical Year Book 2025 PDF).",
    )
    parser.add_argument(
        "output_json",
        type=str,
        nargs="?",
        default="phase1_structured_output.json",
        help="Path to write the structured JSON output.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        raise SystemExit(f"PDF not found: {args.pdf_path}")

    save_structured_pdf(args.pdf_path, args.output_json)
    print(f"Structured extraction saved to: {args.output_json}")


