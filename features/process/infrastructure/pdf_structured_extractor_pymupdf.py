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

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

import json

import fitz  # PyMuPDF


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
    Extract chapter name from header text.
    
    Looks for patterns like:
    - "الفصل 2 : المناخ" → "المناخ"
    - "Climate : Chapter 2" → "Climate"
    - "Chapter 2: Climate" → "Climate"
    """
    if not text:
        return None
    
    # Arabic and English chapter patterns
    patterns = [
        r'(?:الفصل|Chapter)\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+)',  # Chapter N: Name (stop at tab/newline/digit)
        r'([^\t\n\r]+?)\s*[:：]\s*(?:الفصل|Chapter)',  # Name: Chapter N (stop at tab/newline)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            chapter_name = match.group(1).strip()
            # Clean up: remove trailing Arabic text, numbers, and extra whitespace
            chapter_name = re.sub(r'\s*\d+.*$', '', chapter_name)  # Remove trailing numbers and everything after
            chapter_name = re.sub(r'\s{2,}', ' ', chapter_name)  # Normalize whitespace
            return chapter_name.strip() if chapter_name else None
    
    return None


def _parse_section_from_header(text: str) -> Optional[str]:
    """
    Extract section name from header text.
    
    Looks for patterns like:
    - "القسم 1 : المناخ" → "المناخ"
    - "Section 1: Main Results" → "Main Results"
    """
    if not text:
        return None
    
    patterns = [
        r'(?:القسم|Section)\s*[\d]+\s*[:：]\s*([^\t\n\r\d]+)',  # Section N: Name (stop at tab/newline/digit)
        r'([^\t\n\r]+?)\s*[:：]\s*(?:القسم|Section)',  # Name: Section N (stop at tab/newline)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            section_name = match.group(1).strip()
            # Clean up: remove trailing Arabic text, numbers, and extra whitespace
            section_name = re.sub(r'\s*\d+.*$', '', section_name)  # Remove trailing numbers and everything after
            section_name = re.sub(r'\s{2,}', ' ', section_name)  # Normalize whitespace
            return section_name.strip() if section_name else None
    
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

    blocks: List[PageBlock] = []

    for block in blocks_raw:
        # block format: (x0, y0, x1, y1, text, block_no, block_type, ...)
        if len(block) < 7:
            continue

        x0, y0, x1, y1, text, _block_no, block_type = block[:7]

        # 0 = text block according to PyMuPDF
        if block_type != 0:
            continue

        raw_text = (text or "").strip()
        if not raw_text:
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

        blocks.append(
            PageBlock(
                type=block_kind,
                content=content,
                bbox=bbox,
            )
        )

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
    doc = fitz.open(pdf_path)
    try:
        pages: List[Dict[str, Any]] = []
        
        # Track current chapter/section across pages
        current_chapter: Optional[str] = None
        current_section: Optional[str] = None

        for i, page in enumerate(doc, start=1):
            is_toc = _is_toc_like_page(page)

            text_blocks = _extract_text_blocks(page, is_toc_page=is_toc)
            table_blocks = _extract_table_blocks(page, is_toc_page=is_toc)

            # Do NOT merge pages; every page stands alone.
            all_blocks = text_blocks + table_blocks
            
            # Pair bilingual blocks (Arabic + English together)
            page_width = page.rect.width
            paired_blocks = _pair_bilingual_blocks(all_blocks, page_width)
            
            # Extract chapter/section from headers and propagate to all blocks
            for block in paired_blocks:
                if block.type == "header":
                    # Try to extract chapter/section from header text
                    text = block.content.ar or block.content.en or ""
                    chapter = _parse_chapter_from_header(text)
                    section = _parse_section_from_header(text)
                    
                    if chapter:
                        current_chapter = chapter
                        current_section = None  # Reset section on new chapter
                    elif section:
                        current_section = section
                
                # Propagate current chapter/section to all blocks
                block.chapter = current_chapter
                block.section = current_section

            page_extraction = PageExtraction(
                pageNumber=i,
                blocks=paired_blocks,
            )
            pages.append(asdict(page_extraction))

        return pages
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


