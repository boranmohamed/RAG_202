"""
Phase 2 OCR — OCR-SPECIFIC PREPROCESSING & NORMALIZATION

Input:
    Structured blocks from Phase 1 OCR (pageNumber + blocks with bilingual content).

Goal:
    - Apply OCR-specific fixes for Arabic text (ligature artifacts, common OCR confusions)
    - Merge split Arabic paragraphs (OCR often splits continuous text)
    - Normalize Arabic text (conservative, preserve meaning)
    - Normalize English text (whitespace, basic cleanup)
    - Keep bilingual alignment
    - DO NOT touch table content (already in markdown)
    - DO NOT chunk or add extra metadata

Output format:
    Same as input (list of normalized block dicts with bilingual content)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

from features.process.infrastructure.utils.ocr_utils import (
    detect_lang,
    normalize_arabic as normalize_arabic_basic,
    normalize_english as normalize_english_basic,
    clean_ocr_text,
)

# Setup logger for this module
logger = logging.getLogger(__name__)


BlockType = Literal["header", "paragraph", "table", "toc", "footer"]


# Arabic end-of-sentence punctuation
AR_END_PUNCT = re.compile(r"[\.!\?؟…:]$")

# Common OCR artifacts specific to Arabic yearbooks/statistical documents
# Based on ligature rendering issues and common OCR mistakes
OCR_ARABIC_FIXES = {
    # Ligature artifacts (ال prefix broken)
    'اجل': 'ال',
    'احل': 'ال',
    'اال': 'ال',
    'الئ': 'الل',
    'امل': 'ال',
    
    # Common OCR confusions in statistical yearbooks
    'احلرارة': 'الحرارة',  # temperature
    'احلجر': 'الحجر',  # stone
    'الناطق': 'المناطق',  # regions
    'اجلمهورية': 'الجمهورية',  # republic
    'الناخ': 'المناخ',  # climate
    'احلافظات': 'المحافظات',  # governorates
    'احلكومة': 'الحكومة',  # government
    'الوقع': 'الموقع',  # location
    'الواليات': 'الولايات',  # wilayats
    'اجلدول': 'الجدول',  # table
    'احلدود': 'الحدود',  # borders
    'الدينة': 'المدينة',  # city
    'الساحة': 'المساحة',  # area
    'مشاال': 'شمالاً',  # north
}


def apply_ocr_arabic_fixes(text: str) -> str:
    """
    Apply OCR-specific fixes for common Arabic recognition errors.
    
    These fixes target:
    - Ligature artifacts (ال prefix recognition issues)
    - Common character confusions in OCR
    - Yearbook-specific vocabulary fixes
    
    Args:
        text: Arabic text with potential OCR errors
        
    Returns:
        Text with OCR artifacts fixed
    """
    if not text:
        return ""
    
    # Apply character-level substitutions
    for incorrect, correct in OCR_ARABIC_FIXES.items():
        text = text.replace(incorrect, correct)
    
    return text


def normalize_arabic_ocr(text: str) -> str:
    """
    OCR-aware Arabic normalization.
    
    Steps:
    1. Apply comprehensive OCR cleaning (Unicode control chars, diacritics, numerals)
    2. Apply OCR-specific fixes (ligatures, common confusions)
    3. Apply basic Arabic normalization (Alef, Ya, etc.)
    4. Normalize whitespace
    
    Args:
        text: Arabic text to normalize
        
    Returns:
        Normalized Arabic text
    """
    if not text or not text.strip():
        return ""
    
    # Do NOT clean markdown tables (we must keep pipes `|`, dashes, alignment, etc.)
    if text.strip().startswith("|") and "|" in text:
        return text
    
    # Step 1: Apply comprehensive OCR cleaning (diacritics, control chars, numerals)
    text = clean_ocr_text(text)
    
    # Step 2: Fix OCR artifacts (ligatures, common confusions)
    text = apply_ocr_arabic_fixes(text)
    
    # Step 3: Apply basic normalization
    text = normalize_arabic_basic(text)
    
    return text


def should_merge_ar(prev: str, nxt: str) -> bool:
    """
    Determine if two consecutive Arabic text segments should be merged.
    
    OCR often splits continuous paragraphs into separate blocks.
    Merge if:
    - Previous text doesn't end with sentence punctuation
    - Next text doesn't look like a heading
    
    Args:
        prev: Previous text segment
        nxt: Next text segment
        
    Returns:
        True if segments should be merged
    """
    if not prev or not nxt:
        return False
    
    prev = prev.strip()
    nxt = nxt.strip()
    
    # Don't merge if previous ends with sentence punctuation
    if AR_END_PUNCT.search(prev):
        return False
    
    # Don't merge if next looks like a heading (short + contains keywords)
    if len(nxt) < 60:
        heading_keywords = ["الفصل", "القسم", "جدول", "الباب", "المحور"]
        if any(keyword in nxt for keyword in heading_keywords):
            return False
    
    # Don't merge if next starts with a number pattern (likely list item or table ref)
    if re.match(r'^\d+[\.\-\):]', nxt):
        return False
    
    # Otherwise, merge
    return True


def merge_arabic_lines(ar_text: str) -> str:
    """
    Merge split Arabic lines that belong to same paragraph.
    
    OCR often breaks continuous text into multiple lines.
    This function rejoins lines that don't have strong sentence boundaries.
    
    Args:
        ar_text: Multi-line Arabic text
        
    Returns:
        Text with split paragraphs merged
    """
    if not ar_text:
        return ""
    
    lines = [ln.strip() for ln in ar_text.splitlines() if ln.strip()]
    
    if not lines:
        return ""
    
    merged: List[str] = []
    
    for ln in lines:
        if merged and should_merge_ar(merged[-1], ln):
            # Merge with previous line (add space between)
            merged[-1] = merged[-1].rstrip() + " " + ln.lstrip()
        else:
            # Start new paragraph
            merged.append(ln)
    
    return "\n".join(merged).strip()


def normalize_english_ocr(text: str) -> str:
    """
    OCR-aware English normalization.
    
    Lighter than Arabic (English OCR is generally better).
    Just normalize whitespace and fix common issues.
    
    Args:
        text: English text to normalize
        
    Returns:
        Normalized English text
    """
    if not text or not text.strip():
        return ""
    
    # Use basic normalization
    text = normalize_english_basic(text)
    
    # Fix common OCR spacing issues around punctuation
    text = re.sub(r'\s+([,\.;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,\.;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
    
    return text


def normalize_ocr_blocks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize OCR-extracted blocks with bilingual content.
    
    This is the main entry point for Phase 2 OCR preprocessing.
    Takes Phase 1 OCR output and applies OCR-specific normalization.
    
    Processing:
    - Arabic: OCR fixes + paragraph merging + normalization
    - English: Basic normalization + spacing fixes
    - Tables: Keep as-is (already in markdown)
    - Preserve bilingual alignment
    - Keep metadata (chapter, section, pageNumber, etc.)
    
    Args:
        pages: List of page dicts from Phase 1 OCR with structure:
               {
                   "pageNumber": int,
                   "blocks": [
                       {
                           "type": str,
                           "content": {"ar": str|None, "en": str|None},
                           "bbox": [...] | None,
                           "chapter": str | None,
                           "section": str | None
                       }
                   ]
               }
    
    Returns:
        List of normalized block dicts (flattened, one dict per block):
        [
            {
                "pageNumber": int,
                "type": str,
                "content": {"ar": str|None, "en": str|None},
                "chapter": str | None,
                "section": str | None
            }
        ]
    """
    logger.info(f"normalize_ocr_blocks: Starting OCR normalization - {len(pages)} pages")
    normalized_blocks: List[Dict[str, Any]] = []
    
    # Track Arabic text across consecutive blocks for merging
    pending_ar: List[str] = []
    pending_metadata: Optional[Dict[str, Any]] = None
    total_blocks = 0
    skipped_blocks = 0
    merged_arabic_count = 0
    
    def flush_pending_arabic():
        """Flush accumulated Arabic text as a single block."""
        nonlocal pending_ar, pending_metadata, merged_arabic_count
        
        if pending_ar and pending_metadata:
            # Merge all pending Arabic lines
            merged_ar = merge_arabic_lines("\n".join(pending_ar))
            
            if merged_ar:
                if len(pending_ar) > 1:
                    merged_arabic_count += 1
                    logger.debug(f"normalize_ocr_blocks: Flushing {len(pending_ar)} merged Arabic blocks (page={pending_metadata['pageNumber']})")
                normalized_blocks.append({
                    "pageNumber": pending_metadata["pageNumber"],
                    "type": pending_metadata["type"],
                    "content": {
                        "ar": merged_ar,
                        "en": None
                    },
                    "chapter": pending_metadata.get("chapter"),
                    "section": pending_metadata.get("section"),
                })
        
        # Reset
        pending_ar = []
        pending_metadata = None
    
    for page in pages:
        page_number = page.get("pageNumber", 0)
        page_blocks = page.get("blocks", [])
        logger.debug(f"normalize_ocr_blocks: Processing page {page_number} with {len(page_blocks)} blocks")
        
        for block_idx, block in enumerate(page_blocks):
            total_blocks += 1
            block_type = block.get("type", "paragraph")
            content = block.get("content", {})
            chapter = block.get("chapter")
            section = block.get("section")
            
            ar = content.get("ar")
            en = content.get("en")
            
            # Skip empty blocks
            if not ar and not en:
                skipped_blocks += 1
                logger.debug(f"normalize_ocr_blocks: Page {page_number}, Block {block_idx}: Skipped (empty)")
                continue
            
            # Tables: Keep as-is (already in markdown from Phase 1)
            if block_type == "table":
                # Flush any pending Arabic before table
                flush_pending_arabic()
                
                normalized_blocks.append({
                    "pageNumber": page_number,
                    "type": "table",
                    "content": {
                        "ar": ar,  # markdown content
                        "en": None
                    },
                    "chapter": chapter,
                    "section": section,
                })
                logger.debug(f"normalize_ocr_blocks: Page {page_number}, Block {block_idx}: Added table block (ar_len={len(ar) if ar else 0})")
                continue
            
            # TOC: Keep as-is (no normalization needed)
            if block_type == "toc":
                flush_pending_arabic()
                
                normalized_blocks.append({
                    "pageNumber": page_number,
                    "type": "toc",
                    "content": {
                        "ar": ar,
                        "en": en
                    },
                    "chapter": chapter,
                    "section": section,
                })
                logger.debug(f"normalize_ocr_blocks: Page {page_number}, Block {block_idx}: Added TOC block")
                continue
            
            # Headers: Process but don't merge across blocks
            if block_type == "header":
                flush_pending_arabic()
                
                norm_ar = normalize_arabic_ocr(ar) if ar else None
                norm_en = normalize_english_ocr(en) if en else None
                normalized_blocks.append({
                    "pageNumber": page_number,
                    "type": "header",
                    "content": {
                        "ar": norm_ar,
                        "en": norm_en
                    },
                    "chapter": chapter,
                    "section": section,
                })
                logger.debug(f"normalize_ocr_blocks: Page {page_number}, Block {block_idx}: Added header block (ar_len={len(norm_ar) if norm_ar else 0}, en_len={len(norm_en) if norm_en else 0})")
                continue
            
            # Paragraphs: Accumulate Arabic for merging, process English separately
            
            # Process English immediately (no merging needed)
            if en:
                # Flush pending Arabic before English block
                flush_pending_arabic()
                
                normalized_en = normalize_english_ocr(en)
                if normalized_en:
                    normalized_blocks.append({
                        "pageNumber": page_number,
                        "type": block_type,
                        "content": {
                            "ar": None,
                            "en": normalized_en
                        },
                        "chapter": chapter,
                        "section": section,
                    })
                    logger.debug(f"normalize_ocr_blocks: Page {page_number}, Block {block_idx}: Added English block (en_len={len(normalized_en)})")
            
            # Accumulate Arabic for merging
            if ar:
                # Normalize first
                normalized_ar = normalize_arabic_ocr(ar)
                
                if normalized_ar:
                    # If metadata changed (new page/chapter/section), flush previous
                    if pending_metadata and (
                        pending_metadata["pageNumber"] != page_number or
                        pending_metadata.get("chapter") != chapter or
                        pending_metadata.get("section") != section
                    ):
                        logger.debug(f"normalize_ocr_blocks: Page {page_number}, Block {block_idx}: Metadata changed, flushing pending Arabic")
                        flush_pending_arabic()
                    
                    # Accumulate
                    pending_ar.append(normalized_ar)
                    pending_metadata = {
                        "pageNumber": page_number,
                        "type": block_type,
                        "chapter": chapter,
                        "section": section,
                    }
                    logger.debug(f"normalize_ocr_blocks: Page {page_number}, Block {block_idx}: Accumulated Arabic (pending_count={len(pending_ar)})")
    
    # Flush any remaining Arabic
    flush_pending_arabic()
    
    logger.info(f"normalize_ocr_blocks: OCR normalization complete - {len(normalized_blocks)} blocks normalized ({skipped_blocks} skipped, {merged_arabic_count} Arabic merges)")
    return normalized_blocks

