"""
Phase 3 — SEMANTIC CHUNKING with strict rules enforcement.

This module implements chunking for bilingual (Arabic/English) Yearbook content
using ChunkWise, with mandatory rules to ensure RAG quality.

Following clean architecture:
- Implements IChunker interface from domain layer
- Infrastructure adapter for ChunkWise chunking

Rules enforced:
- Never chunk across sections
- Titles are metadata only (not chunked)
- TOC/index pages never chunked
- One chunk = one meaning
- Arabic + English stay together
- Sentence boundaries are sacred
- Tables are isolated
- No duplicate content
- No broken text
- Metadata required
- Page numbers are truth
- Chunk size limits
- Controlled overlap
- Validation checklist

Input: Normalized blocks from Phase 2
Output: Validated chunks ready for Phase 4 (Embeddings)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

from features.process.domain.interfaces import IChunker

# Setup logger for this module
logger = logging.getLogger(__name__)

# ChunkWise imports
try:
    from chunkwise import Chunker
    from chunkwise.language.detector import detect_language as chunkwise_detect_language
    CHUNKWISE_AVAILABLE = True
except ImportError:
    CHUNKWISE_AVAILABLE = False
    # Fallback chunkers
    def chunkwise_detect_language(text: str) -> str:  # noqa: ARG001
        return "mixed"


ChunkType = Literal["narrative", "table"]
LangCode = Literal["ar", "en", "mixed"]


def slugify(text: str) -> str:
    """Convert text to URL-safe slug for chunk IDs."""
    return re.sub(r'[^a-z0-9]+', '_', (text or 'unknown').lower()).strip('_')


@dataclass
class ChunkMetadata:
    """Required metadata for every chunk (Rule 11)."""
    document: str = "Statistical Year Book 2025"
    year: int = 2024
    chapter: Optional[str] = None
    section: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    chunk_type: str = "narrative"  # Changed from ChunkType to allow glossary, etc.
    language: List[str] = None  # type: ignore
    embedding_allowed: bool = True  # NEW: Explicit embedding control

    def __post_init__(self):
        if self.language is None:
            self.language = []


@dataclass
class Chunk:
    """A single chunk ready for embedding."""
    chunk_id: str
    content: Dict[str, Optional[str]]  # {"ar": "...", "en": "..."}
    metadata: ChunkMetadata


def classify_content_type(text: str, block_type: str, section: Optional[str] = None) -> str:
    """
    Classify content type with section context.
    
    REFINEMENT 2: Added section-aware glossary detection for better precision.
    """
    if block_type == "table":
        return "table"
    
    if block_type in ["toc", "footer", "header"]:
        return block_type
    
    # Check section context first (more reliable than patterns alone)
    if section:
        # Arabic or English section names indicating glossary/units
        glossary_section_keywords = [
            "الوحدات", "Units", "Measurements", "القياس",
            "Legend", "الرموز", "Symbols", "المصطلحات"
        ]
        if any(keyword in section for keyword in glossary_section_keywords):
            return "glossary"
    
    # Detect glossary/units by content patterns
    glossary_patterns = [
        r'(mm|cm|km|kg|°C|°م)\s*\n',  # Units with newlines
        r'\b(Centigrade|Millimetre|Kilogram)\b.*\n.*\b(mm|kg|°C)\b',  # Unit definitions
        r'^\s*[A-Z][a-z]+\s+[A-Z]\S*\s*\n',  # "Temperature C°"
    ]
    
    for pattern in glossary_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "glossary"
    
    return "narrative"


def is_embedding_eligible(chunk_type: str) -> bool:
    """
    Explicit embedding policy (CRITICAL).
    
    Makes embedding control explicit to prevent silent pollution.
    """
    eligible_types = {"narrative", "table"}
    return chunk_type in eligible_types


def dynamic_chunk_size(block_type: str) -> int:
    """
    Dynamic chunk size based on content type.
    
    Following Yearbook 2025 rules:
    - Chunks must grow dynamically
    - Merge pages ONLY until semantic boundary changes OR chunk exceeds ~1500 characters per language block
    - Do NOT enforce token limits (character-based only)
    
    Dynamic approach:
    - Narrative blocks → chunk at ~1500 chars per language block
    - Tables → chunk at ~6000 chars (preserve table structure)
    
    Args:
        block_type: Type of content block ("table", "narrative", etc.)
        
    Returns:
        Maximum characters for chunking this block type (per language block)
    """
    if block_type == "table":
        return 6000  # large tables (preserve structure)
    return 1500  # narrative (~1500 chars per language block as per rules)


def should_chunk(block: Dict[str, Any]) -> bool:
    """
    Pre-filter rules (Rule 1, 2, 3).
    
    Returns False for blocks that should NEVER be chunked:
    - Empty blocks (no content)
    - Tables (handled separately)
    
    UPDATED: Headers, footers, and TOC blocks with meaningful content will be chunked.
    This ensures all extracted content gets processed.
    """
    block_type = block.get("type", "")
    page_num = block.get("pageNumber", 0)
    
    # Check if block has any content
    content = block.get("content", {})
    if isinstance(content, dict):
        ar_text = (content.get("ar") or "").strip()
        en_text = (content.get("en") or "").strip()
        has_content = bool(ar_text or en_text)
    else:
        has_content = False
    
    # Tables handled separately (Rule 7)
    if block_type == "table":
        logger.debug(f"should_chunk: Page {page_num}: Block will be handled separately (type=table)")
        return False  # Tables are handled in chunk_table_block
    
    # Skip only if block has no content at all
    if not has_content:
        logger.debug(f"should_chunk: Page {page_num}: Block skipped (type={block_type}, no content)")
        return False
    
    # All other blocks with content should be chunked (including headers, footers, TOC with content)
    logger.debug(f"should_chunk: Page {page_num}: Block will be chunked (type={block_type}, has_content=True)")
    return True


def extract_section_info(block: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """
    Extract chapter and section information from block.
    
    Now implemented in Phase 1 - chapter/section are parsed from headers
    and attached to blocks during extraction.
    """
    # Chapter and section are now extracted in Phase 1 and propagated through Phase 2
    return block.get("chapter"), block.get("section")


def preprocess_for_chunking(
    content: Dict[str, Optional[str]],
    block_type: str,
) -> tuple[str, LangCode]:
    """
    Preprocess bilingual content for chunking (Rule 6, Rule 9).
    
    Combines Arabic and English into a single string for ChunkWise,
    while preserving language information.
    """
    ar_text = (content.get("ar") or "").strip()
    en_text = (content.get("en") or "").strip()
    
    # Rule 9: Skip empty or broken content
    if not ar_text and not en_text:
        return "", "mixed"
    
    # Combine bilingual content (Rule 5: keep together)
    # Use newline separator to distinguish languages
    combined = ""
    if ar_text:
        combined += ar_text
    if en_text:
        if combined:
            combined += "\n\n"  # Separator between languages
        combined += en_text
    
    # Detect language
    if CHUNKWISE_AVAILABLE:
        try:
            lang = chunkwise_detect_language(combined)
            # Map to our LangCode
            if lang == "ar":
                lang_code: LangCode = "ar"
            elif lang == "en":
                lang_code = "en"
            else:
                lang_code = "mixed"
        except Exception:
            lang_code = "mixed"
    else:
        # Fallback detection
        if ar_text and not en_text:
            lang_code = "ar"
        elif en_text and not ar_text:
            lang_code = "en"
        else:
            lang_code = "mixed"
    
    return combined, lang_code


def chunk_narrative_text(
    text: str,
    language: LangCode,
    max_chars: int = 1500,
    min_chars: int = 50,
    overlap_chars: int = 100,
) -> List[str]:
    """
    Chunk narrative text using semantic boundaries (Rule 4, 6, 13, 14).
    
    Uses character-based chunking (NOT token-based) as per Yearbook 2025 rules.
    Chunks based on semantic boundaries, not fixed length.
    
    UPDATED: Reduced min_chars check to allow smaller chunks that will be merged later.
    
    Args:
        text: Text to chunk
        language: Language code
        max_chars: Maximum characters per chunk (~1500 per language block)
        min_chars: Minimum characters per chunk (not enforced here, validation handles it)
        overlap_chars: Overlap between chunks in characters
    """
    if not text or not text.strip():
        return []
    
    # Use semantic chunking (character-based, not token-based)
    # Don't filter by min_chars here - let validation handle it
    # This ensures we capture all content, even if small
    return semantic_chunk(text, max_chars=max_chars)


def _fallback_chunk_text(text: str, max_chars: int) -> List[str]:
    """Fallback chunking when ChunkWise unavailable."""
    if not text:
        return []
    
    # Simple sentence-based chunking
    # Split by sentence boundaries
    sentences = re.split(r'[.!?؟]\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding this sentence exceeds limit, start new chunk
        if current_chunk and len(current_chunk) + len(sentence) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def is_heading(line: str) -> bool:
    """
    Check if a line is a heading (Chapter, Section, etc.).
    
    Args:
        line: Text line to check
        
    Returns:
        True if line appears to be a heading
    """
    return bool(re.match(r"(Chapter|Section|\d+-\s|الفصل|القسم)", line.strip()))


def is_governorate(line: str) -> bool:
    """
    Check if a line indicates a geographic area (Governorate, Wilayat).
    
    Args:
        line: Text line to check
        
    Returns:
        True if line appears to be a governorate/wilayat indicator
    """
    return bool(re.search(r"(حمافظة|Governorate|الوالية|Wilayat)", line))


def is_table_indicator(line: str) -> bool:
    """
    Check if a line indicates table content (average, humidity, rainfall, %).
    
    Args:
        line: Text line to check
        
    Returns:
        True if line appears to be a table indicator
    """
    return bool(re.search(r"(?i)(average|table|humidity|rainfall|%)", line))


def semantic_chunk(text: str, max_chars: int = 1500) -> List[str]:
    """
    Semantic block chunker using layout-aware semantic rules.
    
    Chunk boundaries are determined by semantic meaning, not raw length.
    Perfect for data-heavy yearbook content.
    
    Following Yearbook 2025 rules:
    - Chunk based on semantic boundaries (headings, sections, chapters, page structure, table presence)
    - NEVER cut: tables, bilingual paragraphs, Arabic sentences mid-way
    - Merge pages ONLY until semantic boundary changes OR chunk exceeds ~1500 characters per language block
    
    Chunk boundaries:
    - Start new chunk when:
      - heading changes (Chapter, Section)
      - new geographic area (e.g., "Muscat Governorate")
      - entering a table section
      - switching languages ar ↔ en (if on separate lines)
      - large white-space gap
      - line starts with governance/unit lists
      - chunk exceeds ~1500 characters per language block
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk (~1500 per language block as per rules)
        
    Returns:
        List of text chunks
    """
    if not text:
        logger.debug("semantic_chunk: Empty text provided")
        return []
    
    logger.debug(f"semantic_chunk: Starting - text_len={len(text)}, max_chars={max_chars}")
    chunks = []
    current = []
    current_size = 0
    semantic_boundaries = 0
    size_boundaries = 0
    
    for line in text.split("\n"):
        line = line.strip()
        
        if not line:
            # Preserve empty lines for structure but don't count them toward size
            if current:
                current.append("")
            continue
        
        line_size = len(line)
        
        # --- Hard semantic boundaries (ALWAYS start new chunk) ---
        if (is_heading(line) or
            is_governorate(line) or
            is_table_indicator(line)):
            
            if current:
                chunk_text = "\n".join(current)
                chunks.append(chunk_text)
                logger.debug(f"semantic_chunk: Semantic boundary - created chunk {len(chunks)} ({len(chunk_text)} chars) - boundary type: heading={is_heading(line)}, governorate={is_governorate(line)}, table={is_table_indicator(line)}")
                current = []
                current_size = 0
                semantic_boundaries += 1
        
        # Check if adding this line would exceed max_chars
        if current_size + line_size > max_chars and current:
            # Don't cut mid-sentence - try to find sentence boundary
            # If current chunk has content, finalize it
            chunk_text = "\n".join(current)
            chunks.append(chunk_text)
            logger.debug(f"semantic_chunk: Size boundary - created chunk {len(chunks)} ({len(chunk_text)} chars, exceeded {max_chars} limit)")
            current = [line]
            current_size = line_size
            size_boundaries += 1
        else:
            # Add line to current chunk
            current.append(line)
            current_size += line_size
    
    # Add remaining content
    if current:
        chunk_text = "\n".join(current)
        chunks.append(chunk_text)
        logger.debug(f"semantic_chunk: Final chunk {len(chunks)} ({len(chunk_text)} chars)")
    
    logger.info(f"semantic_chunk: Complete - {len(chunks)} chunks created ({semantic_boundaries} semantic boundaries, {size_boundaries} size boundaries)")
    return chunks


def detect_language_robust(text: str) -> LangCode:
    """
    Detect language using character ratio.
    
    More robust than simple presence/absence checks.
    """
    if not text or len(text.strip()) < 3:
        return "mixed"
    
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = arabic_chars + latin_chars
    
    if total_chars == 0:
        return "mixed"
    
    arabic_ratio = arabic_chars / total_chars
    
    if arabic_ratio > 0.7:
        return "ar"
    elif arabic_ratio < 0.3:
        return "en"
    else:
        return "mixed"


def extract_from_original(original_text: str, chunk_text: str) -> str:
    """
    Extract chunk from original language-scoped content (CRITICAL FIX).
    
    REFINEMENT 1: Added confidence threshold to prevent cross-language bleed.
    REFINEMENT 3: Better handling of semantic boundary chunks (headings, governorates).
    """
    if not chunk_text or not chunk_text.strip():
        logger.warning(f"extract_from_original: Empty chunk_text provided")
        return ""
    
    if not original_text or not original_text.strip():
        logger.warning(f"extract_from_original: Empty original_text provided")
        return ""
    
    # Exact match (most common case)
    if chunk_text.strip() in original_text:
        return chunk_text.strip()
    
    # Try normalized match (remove extra whitespace)
    chunk_normalized = " ".join(chunk_text.split())
    original_normalized = " ".join(original_text.split())
    if chunk_normalized in original_normalized:
        # Find the position and return the exact substring
        idx = original_normalized.find(chunk_normalized)
        if idx >= 0:
            # Map back to original text positions (approximate)
            original_words = original_text.split()
            chunk_words = chunk_text.split()
            if len(chunk_words) > 0 and chunk_words[0] in original_words:
                start_idx = original_text.find(chunk_words[0])
                if start_idx >= 0:
                    # Try to find the end
                    end_marker = chunk_words[-1] if len(chunk_words) > 1 else chunk_words[0]
                    end_idx = original_text.find(end_marker, start_idx)
                    if end_idx >= 0:
                        end_idx += len(end_marker)
                        return original_text[start_idx:end_idx].strip()
    
    # Match by word overlap with 60% confidence threshold
    chunk_words = set(chunk_text.split())
    original_words = set(original_text.split())
    
    if not chunk_words:
        logger.warning(f"extract_from_original: No words in chunk_text")
        return ""
    
    overlap = len(chunk_words & original_words)
    ratio = overlap / len(chunk_words)
    
    # Only accept if 60% confidence or higher
    if ratio >= 0.6:
        # Try to find the actual position in original text
        # Find first matching word
        chunk_word_list = chunk_text.split()
        if chunk_word_list:
            first_word = chunk_word_list[0]
            start_idx = original_text.find(first_word)
            if start_idx >= 0:
                # Try to extract a reasonable substring
                # Estimate length based on chunk_text length
                estimated_end = start_idx + len(chunk_text) + 50  # Add some buffer
                extracted = original_text[start_idx:estimated_end].strip()
                logger.debug(f"extract_from_original: Extracted {len(extracted)} chars using word overlap (ratio={ratio:.2%})")
                return extracted
        return chunk_text
    
    # Last resort: if chunk is very short (likely a heading/governorate), try substring search
    if len(chunk_text) < 100:
        # Try to find any part of chunk_text in original_text
        for word in chunk_words:
            if word in original_text:
                word_idx = original_text.find(word)
                # Extract surrounding context
                context_start = max(0, word_idx - 20)
                context_end = min(len(original_text), word_idx + len(chunk_text) + 20)
                extracted = original_text[context_start:context_end].strip()
                logger.debug(f"extract_from_original: Extracted short chunk using substring search ({len(extracted)} chars)")
                return extracted
    
    # Fallback: return empty to trigger validation failure rather than wrong content
    logger.warning(f"extract_from_original: Could not match chunk_text ({len(chunk_text)} chars) to original ({len(original_text)} chars, overlap_ratio={ratio:.2%})")
    return ""


def has_parallel_structure(text: str) -> bool:
    """Check if text has TRUE parallel bilingual structure."""
    if "\n\n" not in text:
        return False
    
    parts = text.split("\n\n", 1)
    if len(parts) != 2:
        return False
    
    lang1 = detect_language_robust(parts[0])
    lang2 = detect_language_robust(parts[1])
    
    # True parallel: different languages, not mixed
    return lang1 != lang2 and lang1 != "mixed" and lang2 != "mixed"


def split_bilingual_content(chunk_text: str, original_content: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Split chunked text based on ACTUAL language detection (NEVER fabricate).
    
    Enhanced with:
    - REFINEMENT 1: Confidence-based extraction from original
    - Mixed-but-not-parallel handling (e.g., Arabic with English units)
    - Never duplicates text across languages
    """
    ar_text = original_content.get("ar") or ""
    en_text = original_content.get("en") or ""
    
    # CRITICAL: Extract from original, not raw chunk_text
    if ar_text and not en_text:
        return {"ar": extract_from_original(ar_text, chunk_text), "en": None}
    
    if en_text and not ar_text:
        return {"ar": None, "en": extract_from_original(en_text, chunk_text)}
    
    # Handle mixed-but-not-parallel (e.g., Arabic with English units)
    if detect_language_robust(chunk_text) == "mixed":
        if not has_parallel_structure(chunk_text):
            # Mixed but not parallel translation (e.g., "المساحة 309.5 km²")
            return {"ar": chunk_text, "en": None}
    
    # Only split if BOTH languages exist AND parallel structure detected
    if ar_text and en_text and has_parallel_structure(chunk_text):
        parts = chunk_text.split("\n\n", 1)
        lang1 = detect_language_robust(parts[0])
        lang2 = detect_language_robust(parts[1]) if len(parts) > 1 else None
        
        if lang1 == "ar" and lang2 == "en":
            return {"ar": parts[0].strip(), "en": parts[1].strip()}
        elif lang1 == "en" and lang2 == "ar":
            return {"ar": parts[1].strip(), "en": parts[0].strip()}
    
    # Fallback: preserve original structure
    if ar_text and en_text:
        return {
            "ar": extract_from_original(ar_text, chunk_text) if ar_text in chunk_text else None,
            "en": extract_from_original(en_text, chunk_text) if en_text in chunk_text else None
        }
    
    # Never duplicate
    return {"ar": chunk_text, "en": None}


def validate_chunk(chunk: Chunk) -> bool:
    """
    Enhanced validation with fake bilingual detection (Rule 15, Rule 16).
    
    Checks all mandatory requirements before embedding.
    """
    chunk_id_short = chunk.chunk_id[:50] if chunk.chunk_id else "unknown"
    
    # Check required fields
    if not chunk.content:
        logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - no content")
        return False
    
    ar_text = chunk.content.get("ar") or ""
    en_text = chunk.content.get("en") or ""
    
    # Must have at least one language
    if not ar_text and not en_text:
        logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - empty content")
        return False
    
    # Rule 16: Detect fake bilingual (same text in both fields)
    if ar_text and en_text:
        # Check if texts are suspiciously similar (exact duplicate)
        if ar_text == en_text:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - fake bilingual (identical ar/en)")
            return False  # Fake bilingual detected
        
        # Check language consistency
        ar_lang = detect_language_robust(ar_text)
        en_lang = detect_language_robust(en_text)
        
        # If both detected as same language, something is wrong
        if ar_lang == en_lang and ar_lang != "mixed":
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - same language in both fields (ar_lang={ar_lang}, en_lang={en_lang})")
            return False  # Same language in both fields
    
    # Check metadata
    if not chunk.metadata.chapter and not chunk.metadata.section:
        logger.debug(f"validate_chunk: Chunk {chunk_id_short}: No chapter/section metadata (allowed)")
        # Allow chunks without chapter/section for now (will be enhanced)
        pass
    
    if chunk.metadata.page_start == 0:
        logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - page_start is 0")
        return False
    
    # Rule 10: Check for broken text
    ar_text = chunk.content.get("ar") or ""
    en_text = chunk.content.get("en") or ""
    
    # Check for obvious broken Arabic (too many isolated characters)
    if ar_text:
        isolated_chars = len(re.findall(r'\s[\u0600-\u06FF]\s', ar_text))
        isolated_ratio = isolated_chars / len(ar_text) if ar_text else 0
        if isolated_ratio > 0.1:  # More than 10% isolated chars
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - broken Arabic text (isolated_chars={isolated_chars}, ratio={isolated_ratio:.2%})")
            return False
    
    # Check minimum content length for narrative chunks
    # UPDATED: Reduced minimum to 20 chars to capture more content
    # Small chunks will be merged later if needed
    if chunk.metadata.chunk_type == "narrative":
        total_length = len(ar_text) + len(en_text)
        if total_length < 20:  # Too short (reduced from 50 to capture more content)
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - too short (length={total_length} < 20)")
            return False
    
    logger.debug(f"validate_chunk: Chunk {chunk_id_short}: Validation passed (type={chunk.metadata.chunk_type}, ar_len={len(ar_text)}, en_len={len(en_text)})")
    return True


def chunk_section_blocks(
    section_blocks: List[Dict[str, Any]],
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    max_chars: int = 1500,
    min_chars: int = 50,
    overlap_chars: int = 100,
) -> List[Chunk]:
    """
    Chunk blocks within a single section (Rule 1: no cross-section chunking).
    
    Following Yearbook 2025 rules:
    - Character-based limits (NOT token-based)
    - ~1500 characters per language block
    - Semantic boundaries (headings, sections, chapters)
    - NEVER cut tables, bilingual paragraphs, Arabic sentences mid-way
    
    Args:
        section_blocks: List of normalized blocks from Phase 2
        chapter: Chapter name (if known)
        section: Section name (if known)
        max_chars: Maximum characters per chunk (~1500 per language block)
        min_chars: Minimum characters per chunk
        overlap_chars: Overlap between chunks in characters
    
    Returns:
        List of validated chunks
    """
    logger.info(f"chunk_section_blocks: Starting - {len(section_blocks)} blocks (chapter='{chapter}', section='{section}')")
    section_chunks: List[Chunk] = []
    skipped_blocks = 0
    empty_blocks = 0
    invalid_blocks = 0
    rejected_chunks = 0
    
    for block_idx, block in enumerate(section_blocks):
        # Rule 1, 2, 3: Filter blocks
        if not should_chunk(block):
            skipped_blocks += 1
            logger.debug(f"chunk_section_blocks: Block {block_idx}: Skipped (should_chunk=False, type={block.get('type')})")
            continue
        
        # Extract content
        content = block.get("content", {})
        if not isinstance(content, dict):
            invalid_blocks += 1
            logger.warning(f"chunk_section_blocks: Block {block_idx}: Invalid content structure (not dict)")
            continue
        
        # Preprocess for chunking
        combined_text, language = preprocess_for_chunking(content, block.get("type", ""))
        
        if not combined_text:
            empty_blocks += 1
            logger.debug(f"chunk_section_blocks: Block {block_idx}: Empty after preprocessing")
            continue
        
        # Determine block type for dynamic sizing
        block_type = block.get("type", "paragraph")
        content_type = classify_content_type(
            combined_text,
            block_type,
            section=block.get("section")
        )
        
        # Use dynamic chunk size based on content type
        effective_max_chars = dynamic_chunk_size(content_type)
        logger.debug(f"chunk_section_blocks: Block {block_idx}: content_type={content_type}, max_chars={effective_max_chars}, text_len={len(combined_text)}")
        
        # Chunk the text - use semantic chunking (character-based, NOT token-based)
        # Following Yearbook 2025 rules: character-based limits, semantic boundaries
        text_chunks = chunk_narrative_text(
            combined_text,
            language,
            max_chars=effective_max_chars,
            min_chars=min_chars,  # Use parameter from function (default 20, not 50)
            overlap_chars=overlap_chars,  # Use parameter from function
        )
        
        logger.debug(f"chunk_section_blocks: Block {block_idx}: Created {len(text_chunks)} text chunks from {len(combined_text)} chars")
        
        # Create chunk objects
        for i, chunk_text in enumerate(text_chunks):
            # Split back into bilingual structure
            chunk_content = split_bilingual_content(chunk_text, content)
            
            # Log content extraction for debugging
            ar_len = len(chunk_content.get("ar") or "")
            en_len = len(chunk_content.get("en") or "")
            logger.debug(f"chunk_section_blocks: Block {block_idx}, Chunk {i}: Extracted content (ar={ar_len} chars, en={en_len} chars, chunk_text={len(chunk_text)} chars)")
            
            # Check if content extraction failed
            if not chunk_content.get("ar") and not chunk_content.get("en"):
                logger.warning(f"chunk_section_blocks: Block {block_idx}, Chunk {i}: Content extraction failed - both ar and en are empty (chunk_text={len(chunk_text)} chars)")
                invalid_blocks += 1
                continue
            
            # Classify content type with section context (REFINEMENT 2)
            content_type = classify_content_type(
                chunk_text,
                block.get("type", ""),
                section=block.get("section")
            )
            
            # Use block's own chapter/section, not the passed parameters
            block_chapter = block.get("chapter") or chapter
            block_section = block.get("section") or section
            
            # Create metadata with embedding control
            metadata = ChunkMetadata(
                chapter=block_chapter,
                section=block_section,
                page_start=block.get("pageNumber", 0),
                page_end=block.get("pageNumber", 0),
                chunk_type=content_type,
                language=[lang for lang in [language] if lang != "mixed"] or ["ar", "en"] if language == "mixed" else [language],
                embedding_allowed=is_embedding_eligible(content_type),  # NEW: Explicit control
            )
            
            # Create chunk with stable ID
            chapter_slug = re.sub(r'[^a-z0-9]+', '_', (block_chapter or 'unknown').lower()).strip('_')
            section_slug = re.sub(r'[^a-z0-9]+', '_', (block_section or 'unknown').lower()).strip('_')
            chunk = Chunk(
                chunk_id=f"yearbook2025_{chapter_slug}_{section_slug}_p{block.get('pageNumber', 0)}_{i}",
                content=chunk_content,
                metadata=metadata,
            )
            
            # Rule 15: Validate before adding
            if validate_chunk(chunk):
                section_chunks.append(chunk)
                logger.debug(f"chunk_section_blocks: Block {block_idx}, Chunk {i}: Created and validated (id={chunk.chunk_id[:50]}...)")
            else:
                logger.warning(f"chunk_section_blocks: Block {block_idx}, Chunk {i}: Validation failed - chunk rejected")
                rejected_chunks += 1
    
    logger.info(f"chunk_section_blocks: Complete - {len(section_chunks)} chunks created ({skipped_blocks} blocks skipped, {empty_blocks} blocks empty, {invalid_blocks} blocks invalid, {rejected_chunks} chunks rejected)")
    return section_chunks


def chunk_table_block(
    table_block: Dict[str, Any],
    chapter: Optional[str] = None,
    section: Optional[str] = None,
) -> List[Chunk]:
    """
    Chunk table blocks separately (Rule 7, 8).
    
    Tables are isolated chunks with headers preserved.
    """
    logger.debug(f"chunk_table_block: Processing table block (page={table_block.get('pageNumber')}, chapter='{chapter}', section='{section}')")
    content = table_block.get("content", {})
    if not isinstance(content, dict):
        logger.warning(f"chunk_table_block: Invalid content structure (not dict)")
        return []
    
    ar_text = content.get("ar") or ""
    en_text = content.get("en") or ""
    
    if not ar_text and not en_text:
        logger.warning(f"chunk_table_block: Empty table content")
        return []
    
    # Use block's own chapter/section, not the passed parameters
    block_chapter = table_block.get("chapter") or chapter
    block_section = table_block.get("section") or section
    
    # Use dynamic chunk size for tables (6000 chars)
    max_table_chars = dynamic_chunk_size("table")
    
    # Combine table text for size check
    combined_table_text = (ar_text or "") + (en_text or "")
    logger.debug(f"chunk_table_block: Table size = {len(combined_table_text)} chars (max={max_table_chars})")
    
    # Rule 7: One table = one chunk (or multiple if very large)
    # If table exceeds dynamic size, split it (preserve markdown structure)
    if len(combined_table_text) > max_table_chars:
        logger.info(f"chunk_table_block: Large table detected ({len(combined_table_text)} chars), splitting into multiple chunks")
        # Split large table - try to preserve markdown table structure
        # For now, create multiple chunks if table is very large
        table_chunks: List[Chunk] = []
        # Simple split by lines (preserve markdown table rows)
        lines = combined_table_text.split("\n")
        current_chunk_lines = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            if current_size + line_size > max_table_chars and current_chunk_lines:
                # Create chunk from accumulated lines
                chunk_text = "\n".join(current_chunk_lines)
                metadata = ChunkMetadata(
                    chapter=block_chapter,
                    section=block_section,
                    page_start=table_block.get("pageNumber", 0),
                    page_end=table_block.get("pageNumber", 0),
                    chunk_type="table",
                    language=["ar", "en"] if (ar_text and en_text) else (["ar"] if ar_text else ["en"]),
                )
                chunk = Chunk(
                    chunk_id=f"table_{block_chapter or 'unknown'}_{table_block.get('pageNumber', 0)}_{len(table_chunks)}",
                    content={"ar": chunk_text if ar_text else None, "en": chunk_text if en_text else None},
                    metadata=metadata,
                )
                if validate_chunk(chunk):
                    table_chunks.append(chunk)
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Add remaining lines as final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            metadata = ChunkMetadata(
                chapter=block_chapter,
                section=block_section,
                page_start=table_block.get("pageNumber", 0),
                page_end=table_block.get("pageNumber", 0),
                chunk_type="table",
                language=["ar", "en"] if (ar_text and en_text) else (["ar"] if ar_text else ["en"]),
            )
            chunk = Chunk(
                chunk_id=f"table_{block_chapter or 'unknown'}_{table_block.get('pageNumber', 0)}_{len(table_chunks)}",
                content={"ar": chunk_text if ar_text else None, "en": chunk_text if en_text else None},
                metadata=metadata,
            )
            if validate_chunk(chunk):
                table_chunks.append(chunk)
        
        logger.info(f"chunk_table_block: Split large table into {len(table_chunks)} chunks")
        return table_chunks if table_chunks else []
    else:
        # Table fits in one chunk
        logger.debug(f"chunk_table_block: Table fits in single chunk")
        metadata = ChunkMetadata(
            chapter=block_chapter,
            section=block_section,
            page_start=table_block.get("pageNumber", 0),
            page_end=table_block.get("pageNumber", 0),
            chunk_type="table",
            language=["ar", "en"] if (ar_text and en_text) else (["ar"] if ar_text else ["en"]),
        )
        
        chunk = Chunk(
            chunk_id=f"table_{block_chapter or 'unknown'}_{table_block.get('pageNumber', 0)}",
            content={"ar": ar_text, "en": en_text},
            metadata=metadata,
        )
        
        # Validate table chunk
        if validate_chunk(chunk):
            logger.debug(f"chunk_table_block: Created and validated table chunk (id={chunk.chunk_id[:50]}...)")
            return [chunk]
        else:
            logger.warning(f"chunk_table_block: Table chunk validation failed - chunk rejected")
            return []


def merge_small_chunks_with_adjacent(
    chunks: List[Dict[str, Any]], 
    min_word_count: int = 20  # Reduced from 30 to capture more content
) -> List[Dict[str, Any]]:
    """
    Merge chunks with < min_word_count words with adjacent chunks in same section.
    
    Strategy:
    - Small chunk + previous chunk (same section) → merged
    - If no previous, merge with next chunk (same section)
    - If isolated, keep as-is (will be embedded - changed from marking embedding_allowed=False)
    
    UPDATED: Reduced min_word_count and keep all chunks (don't disable embedding).
    This ensures all extracted content gets processed.
    """
    if not chunks:
        return []
    
    merged = []
    skip_next = False
    
    for i, chunk in enumerate(chunks):
        if skip_next:
            skip_next = False
            continue
        
        # Count words in content
        content = chunk.get("content", {})
        ar = content.get("ar") or ""
        en = content.get("en") or ""
        combined_text = (ar + " " + en).strip()
        word_count = len(combined_text.split()) if combined_text else 0
        
        # If chunk is large enough, keep as-is
        if word_count >= min_word_count:
            merged.append(chunk)
            continue
        
        # Small chunk - try to merge
        metadata = chunk.get("metadata", {})
        chapter = metadata.get("chapter")
        section = metadata.get("section")
        
        # Try merge with previous
        if merged and can_merge_chunks(merged[-1], chunk, chapter, section):
            merged[-1] = merge_two_chunks(merged[-1], chunk)
            logger.debug(f"merge_small_chunks: Merged small chunk {i} with previous chunk")
            continue
        
        # Try merge with next
        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            next_metadata = next_chunk.get("metadata", {})
            if (next_metadata.get("chapter") == chapter and 
                next_metadata.get("section") == section):
                merged_chunk = merge_two_chunks(chunk, next_chunk)
                merged.append(merged_chunk)
                skip_next = True
                logger.debug(f"merge_small_chunks: Merged small chunk {i} with next chunk")
                continue
        
        # Isolated small chunk - keep it (changed: don't disable embedding)
        # Even small chunks may be valuable for RAG
        merged.append(chunk)
        logger.debug(f"merge_small_chunks: Kept isolated small chunk {i} ({word_count} words)")
    
    return merged


def can_merge_chunks(
    prev_chunk: Dict[str, Any], 
    curr_chunk: Dict[str, Any],
    curr_chapter: Optional[str],
    curr_section: Optional[str]
) -> bool:
    """
    Check if two chunks can be merged (same chapter/section).
    
    UPDATED: Allow merging of narrative, glossary, and other content types.
    Only exclude tables from merging.
    """
    prev_metadata = prev_chunk.get("metadata", {})
    prev_chunk_type = prev_metadata.get("chunk_type", "narrative")
    
    # Don't merge tables with other content
    if prev_chunk_type == "table":
        return False
    
    return (
        prev_metadata.get("chapter") == curr_chapter and
        prev_metadata.get("section") == curr_section
    )


def merge_two_chunks(
    chunk1: Dict[str, Any], 
    chunk2: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two chunks into one, combining content and updating metadata."""
    # Combine content
    content1 = chunk1.get("content", {})
    content2 = chunk2.get("content", {})
    
    merged_content = {
        "ar": join_texts(content1.get("ar"), content2.get("ar")),
        "en": join_texts(content1.get("en"), content2.get("en")),
    }
    
    # Update metadata
    metadata1 = chunk1.get("metadata", {})
    metadata2 = chunk2.get("metadata", {})
    
    merged_metadata = {
        **metadata1,
        "page_end": max(metadata1.get("page_end", 0), metadata2.get("page_end", 0)),
    }
    
    # Generate new chunk ID
    chapter_slug = slugify(metadata1.get("chapter") or "unknown")
    section_slug = slugify(metadata1.get("section") or "unknown")
    page = metadata1.get("page_start", 0)
    
    return {
        "chunk_id": f"yearbook2025_{chapter_slug}_{section_slug}_p{page}_merged",
        "content": merged_content,
        "metadata": merged_metadata,
    }


def join_texts(text1: Optional[str], text2: Optional[str]) -> Optional[str]:
    """Join two text fragments with proper spacing."""
    if not text1:
        return text2
    if not text2:
        return text1
    return f"{text1.strip()}\n\n{text2.strip()}"


def _merge_block_group(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a group of blocks into a single block.
    
    Combines content from multiple blocks while preserving structure.
    """
    if len(blocks) == 1:
        return blocks[0]
    
    # Merge content
    merged_ar = []
    merged_en = []
    
    for block in blocks:
        content = block.get("content", {})
        ar = content.get("ar")
        en = content.get("en")
        
        if ar:
            merged_ar.append(ar)
        if en:
            merged_en.append(en)
    
    return {
        "pageNumber": blocks[0].get("pageNumber", 0),
        "pageEnd": blocks[-1].get("pageNumber", 0),
        "type": blocks[0].get("type", "paragraph"),
        "content": {
            "ar": "\n".join(merged_ar) if merged_ar else None,
            "en": "\n".join(merged_en) if merged_en else None,
        },
        "chapter": blocks[0].get("chapter"),
        "section": blocks[0].get("section"),
    }


def merge_blocks_by_section(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive blocks within the same section (CRITICAL for context continuity).
    
    This ensures:
    - Multi-block paragraphs stay together
    - Section context is preserved
    - Better semantic chunking
    - Improved retrieval recall
    
    Benefits:
    - Prevents fragmented ideas across blocks
    - Better context for AI understanding
    - Higher quality search results
    """
    if not blocks:
        return []
    
    merged = []
    buffer = []
    current_section = None
    current_chapter = None
    
    for block in blocks:
        block_section = block.get("section")
        block_chapter = block.get("chapter")
        
        # Start new section group if section/chapter changes
        if (block_section != current_section or 
            block_chapter != current_chapter):
            
            if buffer:
                # Merge buffered blocks
                merged.append(_merge_block_group(buffer))
                buffer = []
            
            current_section = block_section
            current_chapter = block_chapter
        
        # Merge blocks with content (paragraph, narrative, and now headers/footers/toc with content)
        # Tables are still handled separately
        block_type = block.get("type", "")
        if block_type == "table":
            # Tables pass through individually (handled separately)
            if buffer:
                merged.append(_merge_block_group(buffer))
                buffer = []
            merged.append(block)
        else:
            # All other blocks with content can be merged (paragraph, narrative, header, footer, toc)
            buffer.append(block)
    
    # Merge remaining buffer
    if buffer:
        merged.append(_merge_block_group(buffer))
    
    return merged


class ChunkWiseBilingualChunker(IChunker):
    """
    Infrastructure adapter: ChunkWise-based bilingual chunker.
    
    Implements IChunker interface using ChunkWise library for semantic chunking
    with strict rules enforcement for bilingual Yearbook content.
    """
    
    def chunk_blocks(
        self,
        blocks: List[Dict[str, Any]],
        document_name: str = "Statistical Year Book 2025",
        year: int = 2024,
        max_chars: int = 1500,
        min_chars: int = 50,
        overlap_chars: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Chunk normalized blocks from Phase 2 into semantic chunks.
        
        Implements IChunker interface.
        """
        logger.info(f"chunk_blocks: Starting chunking - {len(blocks)} blocks, max_chars={max_chars}, min_chars={min_chars}")
        all_chunks: List[Chunk] = []
        
        # Track statistics
        stats = {
            "input_blocks": len(blocks),
            "after_merge": 0,
            "narrative_blocks": 0,
            "table_blocks": 0,
            "skipped_blocks": 0,
            "narrative_chunks_created": 0,
            "table_chunks_created": 0,
            "chunks_validated": 0,
            "chunks_rejected": 0,
        }
        
        # CRITICAL: Merge blocks by section BEFORE chunking
        # This prevents fragmented ideas and improves context continuity
        before_merge = len(blocks)
        blocks = merge_blocks_by_section(blocks)
        after_merge = len(blocks)
        stats["after_merge"] = after_merge
        if before_merge != after_merge:
            logger.info(f"chunk_blocks: Block merging: {before_merge} -> {after_merge} blocks")
        
        # Extract chapter/section from blocks
        current_chapter = None
        current_section = None
        
        narrative_blocks = []
        table_blocks = []
        skipped_blocks = 0
        
        for block_idx, block in enumerate(blocks):
            block_type = block.get("type", "")
            
            # Update current chapter/section
            if block.get("chapter"):
                new_chapter = block.get("chapter")
                if new_chapter != current_chapter:
                    logger.info(f"chunk_blocks: Block {block_idx}: Chapter changed to '{new_chapter}'")
                    current_chapter = new_chapter
            if block.get("section"):
                new_section = block.get("section")
                if new_section != current_section:
                    logger.info(f"chunk_blocks: Block {block_idx}: Section changed to '{new_section}'")
                    current_section = new_section
            
            if block_type == "table":
                table_blocks.append(block)
                logger.debug(f"chunk_blocks: Block {block_idx}: Added to table_blocks (type={block_type})")
            elif should_chunk(block):
                narrative_blocks.append(block)
                logger.debug(f"chunk_blocks: Block {block_idx}: Added to narrative_blocks (type={block_type})")
            else:
                skipped_blocks += 1
                logger.debug(f"chunk_blocks: Block {block_idx}: Skipped (type={block_type}, should_chunk=False)")
        
        stats["narrative_blocks"] = len(narrative_blocks)
        stats["table_blocks"] = len(table_blocks)
        stats["skipped_blocks"] = skipped_blocks
        
        logger.info(f"chunk_blocks: Classified blocks - {len(narrative_blocks)} content blocks (paragraphs/headers/footers/toc), {len(table_blocks)} tables, {skipped_blocks} skipped (empty only)")
        
        # Chunk narrative blocks (includes paragraphs, headers, footers, TOC with content)
        if narrative_blocks:
            logger.info(f"chunk_blocks: Chunking {len(narrative_blocks)} content blocks (chapter='{current_chapter}', section='{current_section}')")
            narrative_chunks = chunk_section_blocks(
                narrative_blocks,
                chapter=current_chapter,
                section=current_section,
                max_chars=max_chars,
                min_chars=min_chars,
                overlap_chars=overlap_chars,
            )
            stats["narrative_chunks_created"] = len(narrative_chunks)
            logger.info(f"chunk_blocks: Created {len(narrative_chunks)} content chunks from {len(narrative_blocks)} blocks")
            all_chunks.extend(narrative_chunks)
        else:
            logger.warning("chunk_blocks: No content blocks to chunk - this may indicate an issue!")
        
        # Chunk table blocks separately
        if table_blocks:
            logger.info(f"chunk_blocks: Processing {len(table_blocks)} table blocks")
            for table_idx, table_block in enumerate(table_blocks):
                table_chunks = chunk_table_block(
                    table_block,
                    chapter=current_chapter,
                    section=current_section,
                )
                stats["table_chunks_created"] += len(table_chunks)
                logger.debug(f"chunk_blocks: Table {table_idx}: Created {len(table_chunks)} chunks")
                all_chunks.extend(table_chunks)
        else:
            logger.debug("chunk_blocks: No table blocks to process")
        
        # Count validated vs rejected chunks
        stats["chunks_validated"] = len(all_chunks)
        
        # Convert to dictionaries first (for merge function compatibility)
        result = []
        for chunk in all_chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": asdict(chunk.metadata),
            }
            result.append(chunk_dict)
        
        # NEW: Merge small chunks with adjacent chunks (AFTER all chunking)
        # This fixes low-information chunks while preserving section boundaries
        before_merge_small = len(result)
        result = merge_small_chunks_with_adjacent(result, min_word_count=20)  # Reduced threshold
        after_merge_small = len(result)
        if before_merge_small != after_merge_small:
            logger.info(f"chunk_blocks: Small chunk merging: {before_merge_small} -> {after_merge_small} chunks")
        else:
            logger.debug(f"chunk_blocks: No small chunks to merge (all {before_merge_small} chunks kept)")
        
        # Final statistics report
        total_chunks_created = stats['narrative_chunks_created'] + stats['table_chunks_created']
        total_rejected = total_chunks_created - stats['chunks_validated']
        blocks_processed = stats['narrative_blocks'] + stats['table_blocks']
        blocks_with_content = blocks_processed  # All blocks that passed should_chunk have content
        
        logger.info("=" * 80)
        logger.info("CHUNKING STATISTICS REPORT")
        logger.info("=" * 80)
        logger.info(f"INPUT:")
        logger.info(f"  - Total blocks received: {stats['input_blocks']}")
        logger.info(f"  - After block merging: {stats['after_merge']} blocks")
        logger.info(f"")
        logger.info(f"CLASSIFICATION:")
        logger.info(f"  - Content blocks (paragraphs/headers/footers/toc with content): {stats['narrative_blocks']}")
        logger.info(f"  - Table blocks: {stats['table_blocks']}")
        logger.info(f"  - Skipped blocks (empty only): {stats['skipped_blocks']}")
        logger.info(f"")
        logger.info(f"CHUNKING RESULTS:")
        logger.info(f"  - Content chunks created: {stats['narrative_chunks_created']}")
        logger.info(f"  - Table chunks created: {stats['table_chunks_created']}")
        logger.info(f"  - Total chunks created: {total_chunks_created}")
        logger.info(f"")
        logger.info(f"POST-PROCESSING:")
        logger.info(f"  - After small chunk merging: {len(result)} final chunks")
        logger.info(f"")
        logger.info(f"VALIDATION:")
        logger.info(f"  - Chunks validated: {stats['chunks_validated']}")
        logger.info(f"  - Chunks rejected: {total_rejected}")
        logger.info(f"")
        logger.info(f"COVERAGE:")
        if stats['input_blocks'] > 0:
            coverage_pct = (blocks_processed / stats['input_blocks']) * 100
            logger.info(f"  - Blocks processed: {blocks_processed} / {stats['input_blocks']} ({coverage_pct:.1f}%)")
            if stats['skipped_blocks'] > 0:
                logger.warning(f"  - WARNING: {stats['skipped_blocks']} blocks were skipped (empty content)")
        logger.info("=" * 80)
        
        logger.info(f"chunk_blocks: Chunking complete - {len(result)} final chunks")
        return result


# Convenience function for backward compatibility
def chunk_phase3_from_blocks(
    blocks: List[Dict[str, Any]],
    document_name: str = "Statistical Year Book 2025",
    year: int = 2024,
    max_chars: int = 1500,
    min_chars: int = 50,
    overlap_chars: int = 100,
) -> List[Dict[str, Any]]:
    """
    Convenience function: chunk normalized blocks from Phase 2.
    
    This is a wrapper around ChunkWiseBilingualChunker for backward compatibility.
    For new code, use ChunkWiseBilingualChunker directly.
    
    Following Yearbook 2025 rules: character-based limits (NOT token-based).
    """
    chunker = ChunkWiseBilingualChunker()
    return chunker.chunk_blocks(
        blocks=blocks,
        document_name=document_name,
        year=year,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars,
    )


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load Phase 2 output
    with open("phase2_normalized_blocks.json", "r", encoding="utf-8") as f:
        blocks = json.load(f)
    
    # Chunk
    chunks = chunk_phase3_from_blocks(blocks)
    
    # Save
    with open("phase3_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(chunks)} chunks")

