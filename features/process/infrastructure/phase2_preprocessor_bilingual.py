"""
Phase 2 — Arabic / English PREPROCESSING & NORMALIZATION

Input:
    Structured blocks from Phase 1 (pageNumber + blocks with type + rawText).

Goal:
    - Normalize Arabic text (Hamza, diacritics, punctuation, spacing).
    - Keep English text unchanged.
    - Preserve bilingual alignment in mixed segments.
    - DO NOT touch table content.
    - DO NOT chunk, summarize, or enrich with extra metadata.

Output format (flattened per block, preserving pageNumber & type):

    {
      "pageNumber": 45,
      "type": "paragraph | header | table | toc | footer",
      "language": "ar | en | mixed",
      "normalizedText": "..."
    }
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional

import json
import re

# ChunkWise imports (with fallback if not available)
# Based on ChunkWise GitHub: https://github.com/h9-tec/ChunkWise
try:
    from chunkwise.language.arabic.preprocessor import (
        normalize_arabic,
        remove_diacritics,
        normalize_alef,
    )
    from chunkwise.language.detector import detect_language as chunkwise_detect_language
    CHUNKWISE_AVAILABLE = True
except ImportError:
    CHUNKWISE_AVAILABLE = False
    # Fallback: define dummy functions
    def normalize_arabic(text: str) -> str:  # noqa: ARG001
        return text
    def remove_diacritics(text: str) -> str:  # noqa: ARG001
        return text
    def normalize_alef(text: str) -> str:  # noqa: ARG001
        return text
    def chunkwise_detect_language(text: str) -> str:  # noqa: ARG001
        return "mixed"


LangCode = Literal["ar", "en", "mixed"]
BlockType = Literal["header", "paragraph", "table", "toc", "footer"]


ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
EN_LETTER_RE = re.compile(r"[A-Za-z]")

# Arabic diacritics (tashkeel) + related marks
ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


@dataclass
class BilingualNormalizedContent:
    """Normalized bilingual content structure."""
    ar: Optional[str] = None
    en: Optional[str] = None


@dataclass
class NormalizedBlock:
    pageNumber: int
    type: BlockType
    content: BilingualNormalizedContent  # Changed from language + normalizedText to content with ar/en
    chapter: Optional[str] = None  # Preserved from Phase 1
    section: Optional[str] = None  # Preserved from Phase 1


def detect_language(text: str) -> LangCode:
    """
    Language detection for a block of text.
    
    Uses ChunkWise language detection if available, otherwise falls back to
    basic character counting.
    
    - 'ar'     : contains Arabic letters and no Latin letters.
    - 'en'     : contains Latin letters and no Arabic letters.
    - 'mixed'  : both Arabic and Latin letters present, or neither (fallback).
    """
    if not text:
        return "mixed"
    
    # Try ChunkWise language detection first
    if CHUNKWISE_AVAILABLE:
        try:
            chunkwise_lang = chunkwise_detect_language(text)
            # Map ChunkWise language codes to our LangCode type
            if chunkwise_lang == "ar":
                return "ar"
            elif chunkwise_lang == "en":
                return "en"
            # Otherwise fall through to basic detection
        except Exception:
            # If ChunkWise fails, fall back to basic detection
            pass
    
    # Fallback: basic character counting
    ar_count = len(ARABIC_LETTER_RE.findall(text))
    en_count = len(EN_LETTER_RE.findall(text))

    if ar_count > 0 and en_count == 0:
        return "ar"
    if en_count > 0 and ar_count == 0:
        return "en"
    return "mixed"


# Common OCR character confusion patterns for Arabic
# Based on font rendering issues and common OCR mistakes
# Note: These are used for reference, actual fixes are done via regex patterns below

# Character-level substitutions for ligature artifacts
# These are applied globally as direct string replacements
OCR_CHAR_SUBSTITUTIONS = {
    # Ligature artifacts (ال prefix broken as اجل, احل, etc.)
    'اجل': 'ال',
    'احل': 'ال',
    'اال': 'ال',
    'الئ': 'الل',
    'امل': 'ال',
    # Common OCR confusions in Yearbook
    'مشاال': 'شمالاً',
    'الساحة': 'المساحة',
    # Additional Yearbook-specific artifacts (from validation)
    'احلرارة': 'الحرارة',  # temperature
    'احلجر': 'الحجر',  # stone
    'الناطق': 'المناطق',  # regions
    'اجلمهورية': 'الجمهورية',  # republic
    'الناخ': 'المناخ',  # climate
    'احلافظات': 'المحافظات',  # governorates
    'احلكومة': 'الحكومة',  # government
    'الوقع': 'الموقع',  # location
    'الواليات': 'الولايات',  # wilayats
}

# Common word patterns that frequently get OCR'd incorrectly
# Format: (regex_pattern, replacement)
OCR_WORD_PATTERNS = [
    # Prepositions and common words (most frequent first)
    (r'\bيف\b', 'في'),  # in
    (r'\bامل\b', 'ال'),  # the (standalone)
    (r'\bامل([^\s]+)\b', r'ال\1'),  # ال prefix
    (r'\بامل([^\s]+)\b', r'بال\1'),  # بال prefix
    (r'\بامل\b', 'بال'),  # بال standalone
    
    # Common words with missing hamza
    (r'\bاهم\b', 'أهم'),  # most important
    (r'\bاهل\b', 'أهل'),  # people/family
    (r'\bاى\b', 'أي'),  # any
    (r'\bاين\b', 'أين'),  # where
    (r'\bاهتا\b', 'اتها'),  # her/its
    
    # Common words with ال prefix errors
    (r'\bاملقدمة\b', 'المقدمة'),
    (r'\باملقدمة\b', 'بالمقدمة'),
    
    # Date/month names (common OCR errors)
    (r'\bديسمرب\b', 'ديسمبر'),  # December
    (r'\بينابر\b', 'يناير'),  # January
    (r'\بفبرابر\b', 'فبراير'),  # February
    (r'\بابريل\b', 'أبريل'),  # April
    (r'\باغسطس\b', 'أغسطس'),  # August
    (r'\باكتوبر\b', 'أكتوبر'),  # October
    
    # Common suffix/prefix errors
    (r'\ببنهايه\b', 'بنهاية'),  # at the end
    (r'\بتوزيعاهتا\b', 'توزيعاتها'),  # their distributions
    (r'\بتوزيعاهتم\b', 'توزيعاتهم'),  # their distributions (plural)
    (r'([^\s])اهتا\b', r'\1اتها'),  # اتها suffix
    (r'([^\s])اهتم\b', r'\1اتهم'),  # اتهام suffix
    
    # Geographic terms (common in Yearbook)
    (r'\bالناطق\b', 'المناطق'),  # regions
    (r'\bالناخ\b', 'المناخ'),  # climate
    (r'\bالحافظات\b', 'المحافظات'),  # governorates
    (r'\bالساحة\b', 'المساحة'),  # area
    (r'\bالوقع\b', 'الموقع'),  # location
    
    # Statistical terms
    (r'\bالصائية\b', 'الإحصائية'),  # statistical
    (r'\bالياانات\b', 'البيانات'),  # data
    (r'\bاحلكومة\b', 'الحكومة'),  # government
    
    # Morphological fixes (conservative - only clear cases)
    # Note: ه → ة conversion is tricky, so we're very conservative
]

# Character-level substitution rules (applied globally)
OCR_CHAR_RULES = [
    # Fix common character confusions
    (r'([^\u0600-\u06FF])يف([^\u0600-\u06FF])', r'\1في\2'),  # يف → في in context
    (r'^يف([^\u0600-\u06FF])', r'في\1'),  # يف → في at start
    (r'([^\u0600-\u06FF])يف$', r'\1في'),  # يف → في at end
    
    # Fix ال prefix errors
    (r'([^\u0600-\u06FF])امل([^\u0600-\u06FF])', r'\1ال\2'),  # امل → ال in context
    (r'^امل([^\u0600-\u06FF])', r'ال\1'),  # امل → ال at start
    (r'([^\u0600-\u06FF])امل$', r'\1ال'),  # امل → ال at end
    
    # Fix missing hamza on ا when followed by certain letters
    (r'\bا([هع])', r'أ\1'),  # ا → أ before ه or ع
    (r'\bا([ي])', r'أ\1'),  # ا → أ before ي (in some contexts)
]


def _fix_arabic_ligatures(text: str) -> str:
    """
    Fix broken ligatures in Arabic text (Layer 0 - before all other fixes).
    
    PDF extraction sometimes breaks the ال (al) prefix ligature into:
    - اجل (ajl)
    - احل (ahl)  
    - اال (aal)
    - امل (aml)
    
    This function repairs these at word boundaries.
    """
    if not text:
        return text
    
    # Fix ligature patterns at word boundaries
    # Pattern: word starting with اجل, احل, اال, امل, or الئ should be ال or الل
    text = re.sub(r'\bاجل(\w+)', r'ال\1', text)  # اجل → ال (already exists)
    text = re.sub(r'\bاحل(\w+)', r'ال\1', text)  # احل → ال
    text = re.sub(r'\bاال(\w+)', r'ال\1', text)  # اال → ال
    text = re.sub(r'\bامل(\w+)', r'ال\1', text)  # امل → ال
    text = re.sub(r'\bالئ(\w+)', r'الل\1', text)  # الئ → الل
    
    # Mid-word ligature artifacts (common words that appear in Yearbook)
    text = re.sub(r'احلرارة', 'الحرارة', text)  # temperature
    text = re.sub(r'احلجر', 'الحجر', text)  # stone
    text = re.sub(r'الناطق', 'المناطق', text)  # regions
    text = re.sub(r'اجلمهورية', 'الجمهورية', text)  # republic
    text = re.sub(r'الناخ', 'المناخ', text)  # climate
    
    return text


def _fix_ocr_errors_arabic(text: str) -> str:
    """
    Comprehensive OCR error correction for Arabic text using pattern-based rules.
    
    This function applies multiple layers of correction:
    0. Ligature repair (most fundamental - fixes broken ال prefix)
    1. Character-level substitutions (common OCR character confusions)
    2. Word-level pattern matching (common words that get OCR'd incorrectly)
    3. Morphological pattern fixes (common suffix/prefix errors)
    
    Strategy:
    - Apply ligature fixes FIRST (Layer 0)
    - Apply character rules (most general - Layer 1)
    - Then apply word patterns (more specific - Layer 2)
    - Preserve legitimate text (conservative approach)
    """
    if not text:
        return text
    
    # Layer 0: Fix ligatures FIRST (most fundamental)
    text = _fix_arabic_ligatures(text)
    
    # Apply direct character substitutions from dictionary
    for wrong, correct in OCR_CHAR_SUBSTITUTIONS.items():
        text = text.replace(wrong, correct)
    
    # Layer 1: Character-level substitutions (most general)
    for pattern, replacement in OCR_CHAR_RULES:
        text = re.sub(pattern, replacement, text)
    
    # Layer 2: Word-level pattern fixes (more specific)
    for pattern, replacement in OCR_WORD_PATTERNS:
        text = re.sub(pattern, replacement, text)
    
    # Layer 3: Fix common morphological patterns and compound words
    # Fix توزيعاتها pattern: توزيع + اتها → توزيعاتها
    text = re.sub(r'توزيع([^\s]*)اهتا\b', r'توزيعاتها', text)
    text = re.sub(r'توزيع([^\s]*)اهتم\b', r'توزيعاتهم', text)
    
    # Fix common compound words with ال
    text = re.sub(r'([^\s])امل([^\s]+)\b', r'\1ال\2', text)  # ال in middle of word
    
    # Fix double ا → أ (common OCR error)
    text = re.sub(r'([^\u0600-\u06FF])اا([^\u0600-\u06FF])', r'\1أ\2', text)
    
    return text


def _normalize_arabic_with_chunkwise(text: str) -> str:
    """
    Normalize Arabic text using ChunkWise preprocessing utilities.
    
    Based on ChunkWise API from https://github.com/h9-tec/ChunkWise:
    - remove_diacritics: Strip Arabic diacritics
    - normalize_alef: Normalize hamza forms (أ, إ, آ -> ا)
    - normalize_arabic: General Arabic normalization
    
    Falls back to basic normalization if ChunkWise is not available.
    """
    if not text:
        return text
    
    if CHUNKWISE_AVAILABLE:
        try:
            # Apply ChunkWise normalization functions in sequence
            # Step 1: Remove diacritics
            normalized = remove_diacritics(text)
            
            # Step 2: Normalize alef/hamza forms
            normalized = normalize_alef(normalized)
            
            # Step 3: General Arabic normalization
            normalized = normalize_arabic(normalized)
            
            return normalized
        except Exception:
            # If ChunkWise fails, fall back to basic normalization
            return _normalize_arabic_chars_basic(text)
    else:
        # Fallback: use basic normalization
        return _normalize_arabic_chars_basic(text)


def _normalize_arabic_chars_basic(text: str) -> str:
    """
    Fallback normalization when ChunkWise is not available.
    
    Normalize core Arabic character variants while preserving meaning:
      - Normalize Alef forms (أ, إ, آ, ٱ -> ا).
      - Normalize Yeh / Alef Maqsura (ى -> ي).
      - Strip Tatweel (ـ).
      - Remove diacritics.
    """
    if not text:
        return text

    # Alef variants
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Yeh vs Alef Maqsura
    text = text.replace("ى", "ي")
    # Tatweel
    text = text.replace("ـ", "")
    # Strip diacritics
    text = ARABIC_DIACRITICS_RE.sub("", text)

    return text


def _normalize_arabic_punctuation(text: str, lang: LangCode) -> str:
    """
    Normalize Arabic punctuation:
      - Question mark: ? -> ؟
      - Comma: , -> ،
      - Semicolon: ; -> ؛

    For clearly English-only text, we do NOT change these.
    For Arabic or mixed blocks, we bias towards Arabic punctuation.
    """
    if not text:
        return text

    if lang == "en":
        # English punctuation is left untouched.
        return text

    # For 'ar' and 'mixed', prefer Arabic forms.
    text = text.replace("?", "؟")
    text = text.replace(",", "،")
    text = text.replace(";", "؛")
    return text


def _normalize_spacing(text: str) -> str:
    """
    Fix broken spacing:
      - Collapse multiple spaces.
      - Normalize all whitespace to simple spaces (except newlines).
      - Remove spaces directly before punctuation.
    """
    if not text:
        return text

    # Preserve newlines; normalize other whitespace to spaces.
    # Replace any whitespace except newline with a single space.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    # Collapse newlines where there are too many (3+ -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # No space before punctuation
    text = re.sub(r"\s+([،؟؛\.,!])", r"\1", text)

    # Single space after punctuation where appropriate
    text = re.sub(r"([،؟؛\.,!])([^\s\n])", r"\1 \2", text)

    # Trim per-line leading/trailing spaces while preserving line breaks
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def _normalize_block_text(text: str, block_type: BlockType) -> str:
    """
    Apply normalization rules to a single block of text, respecting safety rules.

    Normalization pipeline:
    1. OCR error correction (fixes common PDF extraction errors)
    2. ChunkWise Arabic normalization (diacritics, hamza, punctuation, spacing, RTL)
    3. Basic spacing normalization (fallback if ChunkWise not available)

    Safety rules:
    - Tables: returned unchanged (CRITICAL: statistical accuracy preserved)
    - Headers: minimal normalization (preserve structure)
    - Paragraph/toc/footer: full normalization
    """
    if not text:
        return ""

    # TABLE SAFETY RULE: do not modify content.
    # This is CRITICAL - tables must never be normalized to preserve numbers.
    if block_type == "table":
        return text

    # TITLE SAFETY: keep titles essentially unchanged, except for very light
    # whitespace cleanup to avoid obvious extraction glitches.
    if block_type == "header":
        # Only strip redundant surrounding whitespace and excessive internal spaces.
        cleaned = re.sub(r"[ \t]+", " ", text)
        lines = [ln.strip() for ln in cleaned.splitlines()]
        return "\n".join(lines).strip()

    # For other types (paragraph, toc, footer), apply full normalization pipeline
    lang = detect_language(text)

    # Step 1: Fix OCR errors first (before ChunkWise normalization)
    if lang in ("ar", "mixed"):
        text = _fix_ocr_errors_arabic(text)
    
    # Step 2: Apply ChunkWise normalization (handles diacritics, hamza, punctuation, spacing, RTL)
    if lang in ("ar", "mixed"):
        text = _normalize_arabic_with_chunkwise(text)
        # ChunkWise handles punctuation, but we apply our own as fallback if needed
        if not CHUNKWISE_AVAILABLE:
            text = _normalize_arabic_punctuation(text, lang=lang)
    else:
        # English-only: preserve text exactly with minimal spacing fix.
        pass

    # Step 3: Final spacing normalization (language-agnostic, mild)
    # ChunkWise already handles spacing, but we apply basic normalization as safety net
    text = _normalize_spacing(text)

    return text


def _is_fragmented_arabic_paragraph(
    current_block: Dict[str, Any],
    next_block: Dict[str, Any],
) -> bool:
    """
    Detect if next_block is a fragment of current_block's Arabic paragraph.
    
    A fragment is identified if:
    1. Both are paragraphs on the same page
    2. Next block starts with punctuation (. ، … :) or mid-sentence
    3. Next block has no English content (or current has English but next doesn't)
    4. They share the same semantic flow (same page, consecutive)
    5. Current block's Arabic doesn't end with sentence-ending punctuation
    """
    # Must be same page
    if current_block.get("pageNumber") != next_block.get("pageNumber"):
        return False
    
    # Both must be paragraph type
    if current_block.get("type") != "paragraph" or next_block.get("type") != "paragraph":
        return False
    
    # Get Arabic content
    current_content = current_block.get("content", {})
    next_content = next_block.get("content", {})
    
    current_ar = (current_content.get("ar") or "").strip()
    next_ar = (next_content.get("ar") or "").strip()
    
    # Must have Arabic content
    if not current_ar or not next_ar:
        return False
    
    # Check if current block ends mid-sentence (doesn't end with sentence punctuation)
    # This indicates it's likely incomplete
    current_ends_sentence = bool(re.search(r'[\.!؟]\s*$', current_ar))
    
    # Check if next block starts with punctuation (indicating continuation)
    next_starts_with_punct = bool(re.match(r'^[\.،…:;]\s*', next_ar))
    
    # Check if next block has no English (fragment indicator)
    next_en = (next_content.get("en") or "").strip()
    current_en = (current_content.get("en") or "").strip()
    
    has_english_mismatch = (current_en and not next_en) or (not current_en and not next_en)
    
    # Check if next block starts mid-sentence (starts with lowercase/arabic, not capital)
    # For Arabic, check if it starts with a connecting word or continues the sentence
    next_starts_mid_sentence = (
        len(next_ar) < 30 or  # Short fragment
        bool(re.match(r'^[^\u0600-\u06FFA-Z]', next_ar)) or  # Starts with non-Arabic/non-capital
        bool(re.match(r'^[و،ف]', next_ar))  # Starts with connecting words (و, ف)
    )
    
    # Fragment if:
    # 1. Current doesn't end sentence AND next starts with punctuation, OR
    # 2. Current doesn't end sentence AND next has no English AND next is short/mid-sentence
    return (
        (not current_ends_sentence and next_starts_with_punct) or
        (not current_ends_sentence and has_english_mismatch and next_starts_mid_sentence)
    )


def _reconstruct_fragmented_paragraphs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge fragmented Arabic paragraphs into single semantic units.
    
    This fixes the issue where line breaks are incorrectly treated as paragraph breaks.
    """
    if not blocks:
        return blocks
    
    reconstructed: List[Dict[str, Any]] = []
    i = 0
    
    while i < len(blocks):
        current = blocks[i]
        
        # Check if this block should be merged with the next one(s)
        fragments_to_merge = [current]
        j = i + 1
        
        while j < len(blocks):
            next_block = blocks[j]
            
            # Check if next_block is a fragment of current
            if _is_fragmented_arabic_paragraph(current, next_block):
                fragments_to_merge.append(next_block)
                j += 1
                # Update current to the merged block so far for next iteration
                current = _merge_blocks(fragments_to_merge)
            else:
                break
        
        # Merge all fragments into one block
        if len(fragments_to_merge) > 1:
            merged = _merge_blocks(fragments_to_merge)
            reconstructed.append(merged)
        else:
            reconstructed.append(current)
        
        i = j
    
    return reconstructed


def _merge_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple blocks into a single block, combining their Arabic and English content.
    """
    if not blocks:
        return {}
    
    if len(blocks) == 1:
        return blocks[0]
    
    # Use the first block as the base
    merged = blocks[0].copy()
    
    # Combine Arabic text (join with space, remove leading punctuation from fragments)
    ar_parts = []
    en_parts = []
    
    for blk in blocks:
        content = blk.get("content", {})
        ar = (content.get("ar") or "").strip()
        en = (content.get("en") or "").strip()
        
        if ar:
            # Remove leading punctuation from fragments (except first)
            if ar_parts and re.match(r'^[\.،…:;]\s*', ar):
                ar = re.sub(r'^[\.،…:;]\s*', '', ar)
            ar_parts.append(ar)
        
        if en:
            en_parts.append(en)
    
    # Join Arabic parts with space
    merged_ar = " ".join(ar_parts) if ar_parts else None
    
    # Join English parts with space (if all have English)
    merged_en = " ".join(en_parts) if en_parts and all(en_parts) else (en_parts[0] if en_parts else None)
    
    # Update content
    merged["content"] = {
        "ar": merged_ar,
        "en": merged_en,
    }
    
    return merged


def normalize_phase2_from_pages(
    pages: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Core Phase 2 entrypoint when working from in-memory Phase 1 structures.

    Expected Phase 1 input (per page):
      {
        "pageNumber": int,
        "blocks": [
          { 
            "type": "header|paragraph|table|toc|footer", 
            "content": {"ar": "...", "en": "..."}, 
            "bbox": [...] 
          },
          ...
        ]
      }

    Returns a flat list of normalized blocks with bilingual content structure.
    """
    normalized: List[Dict[str, Any]] = []

    for page in pages:
        page_number = int(page.get("pageNumber", 0))
        blocks = page.get("blocks") or []

        for blk in blocks:
            blk_type_str = blk.get("type", "paragraph")

            # Safely coerce to BlockType (falling back to paragraph)
            blk_type: BlockType = (
                blk_type_str
                if blk_type_str in {"header", "paragraph", "table", "toc", "footer"}
                else "paragraph"
            )

            # Extract bilingual content from Phase 1 structure
            content_dict = blk.get("content", {})
            if isinstance(content_dict, dict):
                raw_ar = content_dict.get("ar") or ""
                raw_en = content_dict.get("en") or ""
            else:
                # Fallback for old structure (shouldn't happen, but be safe)
                raw_ar = ""
                raw_en = ""

            # Normalize Arabic text (if present)
            norm_ar = None
            if raw_ar:
                norm_ar = _normalize_block_text(raw_ar, block_type=blk_type)

            # Normalize English text (if present) - minimal normalization
            norm_en = None
            if raw_en:
                # English gets minimal normalization (spacing only, no character changes)
                if blk_type == "table":
                    # Tables pass through untouched
                    norm_en = raw_en
                else:
                    # Light spacing normalization for English
                    norm_en = _normalize_spacing(raw_en)

            nb = NormalizedBlock(
                pageNumber=page_number,
                type=blk_type,
                content=BilingualNormalizedContent(ar=norm_ar, en=norm_en),
                chapter=blk.get("chapter"),  # Preserve from Phase 1
                section=blk.get("section"),  # Preserve from Phase 1
            )
            normalized.append(asdict(nb))

    # Reconstruct fragmented paragraphs BEFORE returning
    normalized = _reconstruct_fragmented_paragraphs(normalized)

    return normalized


def normalize_phase2_from_file(
    input_structured_json: str,
    output_normalized_json: str,
    ensure_ascii: bool = False,
    indent: int | None = 2,
) -> None:
    """
    Convenience helper: load Phase 1 JSON from disk, run Phase 2 normalization,
    and write the normalized blocks to disk.
    """
    with open(input_structured_json, "r", encoding="utf-8") as f:
        pages = json.load(f)

    normalized = normalize_phase2_from_pages(pages)

    with open(output_normalized_json, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=ensure_ascii, indent=indent)


if __name__ == "__main__":
    # Example CLI:
    #   python -m features.process.infrastructure.phase2_preprocessor_bilingual \
    #       phase1_structured_output.json phase2_normalized_blocks.json
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 — Arabic/English preprocessing & normalization for "
            "structured Year Book PDF blocks."
        )
    )
    parser.add_argument(
        "input_structured_json",
        type=str,
        help="Path to Phase 1 structured JSON (output of pdf_structured_extractor_pymupdf).",
    )
    parser.add_argument(
        "output_normalized_json",
        type=str,
        nargs="?",
        default="phase2_normalized_blocks.json",
        help="Path to write the Phase 2 normalized JSON blocks.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_structured_json):
        raise SystemExit(f"Structured JSON not found: {args.input_structured_json}")

    normalize_phase2_from_file(args.input_structured_json, args.output_normalized_json)
    print(f"Phase 2 normalized blocks saved to: {args.output_normalized_json}")


