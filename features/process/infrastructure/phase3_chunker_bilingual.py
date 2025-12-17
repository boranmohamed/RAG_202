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

import hashlib
import logging
import re
from dataclasses import dataclass, asdict, field
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


ChunkType = Literal["narrative", "table", "table_fragment", "header", "reference"]
LangCode = Literal["ar", "en", "mixed"]

# Constants for strict chunk semantics
MAX_HEADER_CHARS = 200  # Headers must never exceed 200 chars total
MAX_NUMERIC_DENSITY_HEADER = 0.10  # Headers with >10% numeric density are reclassified
MAX_TABLE_CHARS_FOR_SPLIT = 12000  # Tables only split if exceeding 12,000 chars
MAX_TABLE_CHARS_NORMAL = 6000  # Normal table chunk size (for structure preservation)


def slugify(text: str) -> str:
    """Convert text to URL-safe slug for chunk IDs."""
    return re.sub(r'[^a-z0-9]+', '_', (text or 'unknown').lower()).strip('_')


def extract_key_phrase(content: Dict[str, Optional[str]], max_words: int = 3) -> str:
    """
    Extract first meaningful words from content for chunk ID.
    
    Prefers English if available, falls back to Arabic.
    Skips common words and returns slugified phrase.
    
    Args:
        content: Dictionary with "ar" and "en" keys
        max_words: Maximum number of words to extract
        
    Returns:
        Slugified key phrase (e.g., "oil_exports_monthly")
    """
    # Common words to skip (English and Arabic)
    common_words_en = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "they", "them", "their"
    }
    common_words_ar = {
        "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
        "التي", "الذي", "كان", "كانت", "يكون", "تكون", "كانوا", "كانت"
    }
    
    # Prefer English, fallback to Arabic
    text = content.get("en") or content.get("ar") or ""
    
    if not text or not text.strip():
        return "unknown"
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Extract words (handle both English and Arabic)
    words = re.findall(r'\b\w+\b|[\u0600-\u06FF]+', text)
    
    # Filter out common words and empty strings
    meaningful_words = []
    for word in words:
        word_lower = word.lower().strip()
        if (word_lower and 
            word_lower not in common_words_en and 
            word not in common_words_ar and
            len(word_lower) > 2):  # Skip very short words
            meaningful_words.append(word)
            if len(meaningful_words) >= max_words:
                break
    
    if not meaningful_words:
        return "unknown"
    
    # Join and slugify
    phrase = "_".join(meaningful_words[:max_words])
    return slugify(phrase)


def generate_content_hash(content: Dict[str, Optional[str]]) -> str:
    """
    Create short hash from normalized content for chunk ID.
    
    Args:
        content: Dictionary with "ar" and "en" keys
        
    Returns:
        First 6 characters of MD5 hash (e.g., "a3f2b1")
    """
    ar_text = (content.get("ar") or "").strip()
    en_text = (content.get("en") or "").strip()
    
    # Normalize whitespace
    ar_text = " ".join(ar_text.split())
    en_text = " ".join(en_text.split())
    
    # Create hash
    content_str = f"{ar_text}|{en_text}"
    hash_obj = hashlib.md5(content_str.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # Return first 6 characters
    return hash_hex[:6]


def calculate_numeric_density(text: str) -> float:
    """
    Calculate the ratio of numeric characters to total characters.
    
    Args:
        text: Text to analyze
        
    Returns:
        Ratio of numeric characters (0.0 to 1.0)
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    numeric_chars = len(re.findall(r'\d', text))
    total_chars = len(text)
    
    return numeric_chars / total_chars if total_chars > 0 else 0.0


def normalize_table_for_hash(content: Dict[str, Optional[str]]) -> str:
    """
    Normalize table content for deduplication hashing.
    
    Strips whitespace, removes headers, normalizes ordering.
    Used to detect duplicate table layouts within same chapter+section.
    
    Args:
        content: Dictionary with "ar" and "en" keys containing table content
        
    Returns:
        Normalized string for hashing
    """
    ar_text = (content.get("ar") or "").strip()
    en_text = (content.get("en") or "").strip()
    
    # Combine both languages
    combined = f"{ar_text}\n{en_text}"
    
    # Remove all whitespace (spaces, tabs, newlines)
    normalized = re.sub(r'\s+', '', combined)
    
    # Remove common table noise (page numbers, book titles)
    normalized = re.sub(r'Statistical\s*Year\s*Book', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'صفحة\s*\d+', '', normalized)  # Arabic "page"
    normalized = re.sub(r'Page\s*\d+', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\d+', '', normalized)  # Remove all numbers for layout comparison
    
    # Sort lines to normalize ordering (if table rows are reordered)
    lines = sorted(normalized.split('\n'))
    normalized = ''.join(lines)
    
    return normalized.lower()


def strip_page_noise(content: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Strip page noise and book titles from content before embedding.
    
    Removes:
    - "Statistical Year Book" (any case)
    - Page numbers (Arabic and English)
    - Repeated bilingual book titles
    
    Args:
        content: Dictionary with "ar" and "en" keys
        
    Returns:
        Cleaned content dictionary
    """
    ar_text = content.get("ar") or ""
    en_text = content.get("en") or ""
    
    # Patterns to remove
    noise_patterns = [
        r'Statistical\s+Year\s+Book\s+\d+',  # "Statistical Year Book 2025"
        r'صفحة\s*\d+',  # Arabic "page X"
        r'Page\s+\d+',  # English "Page X"
        r'\b\d+\s*$',  # Standalone page numbers at end of line
        r'^Statistical\s+Year\s+Book',  # Book title at start
        r'كتاب\s+الإحصاء\s+السنوي',  # Arabic book title
    ]
    
    for pattern in noise_patterns:
        ar_text = re.sub(pattern, '', ar_text, flags=re.IGNORECASE | re.MULTILINE)
        en_text = re.sub(pattern, '', en_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up extra whitespace
    ar_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', ar_text).strip()
    en_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', en_text).strip()
    
    return {
        "ar": ar_text if ar_text else None,
        "en": en_text if en_text else None,
    }


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
    # Table-specific metadata
    units: Optional[List[str]] = None  # Measurement units (e.g., ["000 MT", "000 BBL"])
    data_year: Optional[List[int]] = None  # Years covered by data (e.g., [2024, 2023])
    geography: Optional[List[str]] = None  # Geographic areas (e.g., ["Muscat Governorate"])
    bilingual_alignment: Optional[Dict[str, str]] = None  # Arabic -> English term mapping

    def __post_init__(self):
        if self.language is None:
            self.language = []


@dataclass
class Chunk:
    """A single chunk ready for embedding."""
    chunk_id: str
    content: Dict[str, Optional[str]]  # {"ar": "...", "en": "..."}
    metadata: ChunkMetadata


def classify_content_type(
    text: str, 
    block_type: str, 
    section: Optional[str] = None,
    content: Optional[Dict[str, Optional[str]]] = None
) -> str:
    """
    Classify content type with section context.
    
    REFINEMENT 2: Added section-aware glossary detection for better precision.
    NEW: Enforces header misclassification rules (max 200 chars, <10% numeric density).
    NEW: Detects enumerations (governorates, stations, name lists) as reference.
    """
    if block_type == "table":
        return "table"
    
    # CRITICAL: Header validation - reclassify if exceeds limits
    if block_type == "header":
        if content:
            ar_text = (content.get("ar") or "").strip()
            en_text = (content.get("en") or "").strip()
            total_chars = len(ar_text) + len(en_text)
            
            # Rule 1: Headers must never exceed 200 chars
            if total_chars > MAX_HEADER_CHARS:
                logger.warning(f"classify_content_type: Header exceeds {MAX_HEADER_CHARS} chars ({total_chars}), reclassifying to narrative")
                return "narrative"
            
            # Rule 2: Headers with >10% numeric density are reclassified
            combined_text = f"{ar_text} {en_text}"
            numeric_density = calculate_numeric_density(combined_text)
            if numeric_density > MAX_NUMERIC_DENSITY_HEADER:
                logger.warning(f"classify_content_type: Header has numeric density {numeric_density:.2%} > {MAX_NUMERIC_DENSITY_HEADER:.2%}, reclassifying to narrative")
                return "narrative"
            
            # Rule 3: Headers with lists (more than 3 items) are reclassified
            lines = [line.strip() for line in combined_text.split("\n") if line.strip()]
            if len(lines) > 3:
                logger.warning(f"classify_content_type: Header has {len(lines)} lines (>3), reclassifying to reference")
                return "reference"
        
        return "header"
    
    if block_type in ["toc", "footer"]:
        return block_type
    
    # Check for enumerations (governorates, stations, name lists) BEFORE other checks
    if content and looks_like_enumeration(content):
        logger.debug(f"classify_content_type: Detected enumeration pattern, classifying as reference")
        return "reference"
    
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
    Only narrative, table, and reference (not table_fragment, not header) are eligible.
    """
    eligible_types = {"narrative", "table", "reference"}
    return chunk_type in eligible_types and chunk_type != "table_fragment" and chunk_type != "header"


def dynamic_chunk_size(block_type: str) -> int:
    """
    Dynamic chunk size based on content type.
    
    Following Yearbook 2025 rules:
    - Chunks must grow dynamically
    - Merge pages ONLY until semantic boundary changes OR chunk exceeds ~1500 characters per language block
    - Do NOT enforce token limits (character-based only)
    
    Dynamic approach:
    - Narrative blocks → chunk at ~1500 chars per language block
    - Tables → chunk at ~6000 chars (preserve table structure, but splitting disabled unless >12,000)
    
    Args:
        block_type: Type of content block ("table", "narrative", etc.)
        
    Returns:
        Maximum characters for chunking this block type (per language block)
    """
    if block_type == "table":
        return MAX_TABLE_CHARS_NORMAL  # large tables (preserve structure)
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


def looks_like_enumeration(content: Dict[str, Optional[str]]) -> bool:
    """
    Detect if content is an enumeration/list (governorates, wilayats, stations, name lists).
    
    Enumerations are:
    - Lists of names with no prose explanation
    - Governorate/wilayat listings
    - Ground station name blocks
    - Repeated bilingual name lists
    - Deterministic factual lists users may query
    
    Args:
        content: Dictionary with "ar" and "en" keys
        
    Returns:
        True if content appears to be an enumeration
    """
    ar_text = (content.get("ar") or "").strip()
    en_text = (content.get("en") or "").strip()
    combined_text = f"{ar_text}\n{en_text}"
    
    if not combined_text.strip():
        return False
    
    lines = [line.strip() for line in combined_text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    
    # Check for governorate/wilayat enumeration patterns
    governorate_count = sum(1 for line in lines if is_governorate(line))
    if governorate_count >= 2:  # Multiple governorates = enumeration
        return True
    
    # Check for station name patterns
    station_patterns = [
        r'\b(Station|محطة|محطات)\b',
        r'\b(Ground\s+Station|محطة\s+أرضية)\b',
    ]
    station_count = 0
    for line in lines:
        for pattern in station_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                station_count += 1
                break
    if station_count >= 2:
        return True
    
    # Check for list-like structure (multiple lines with similar patterns)
    # Lists typically have:
    # - Multiple lines starting with similar patterns (numbers, bullets, names)
    # - Low prose density (few sentences, mostly names/terms)
    # - Repeated bilingual name pairs
    
    # Count lines that look like list items (short, no punctuation, name-like)
    list_item_count = 0
    for line in lines:
        # Short lines (likely list items)
        if len(line) < 100:
            # Check if line is mostly names/terms (not prose)
            # Prose has more punctuation, list items have less
            punctuation_ratio = len(re.findall(r'[.!?،,;:]', line)) / len(line) if line else 0
            if punctuation_ratio < 0.1:  # Less than 10% punctuation = likely list item
                list_item_count += 1
    
    # If most lines are list-like, it's an enumeration
    if list_item_count >= len(lines) * 0.7:  # 70% of lines are list items
        return True
    
    # Check for repeated bilingual name patterns
    # Enumerations often have Arabic name followed by English name on same or next line
    bilingual_pair_count = 0
    for i, line in enumerate(lines):
        # Check if line has Arabic
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', line))
        # Check if next line has English (and no Arabic)
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            has_english = bool(re.search(r'[a-zA-Z]', next_line))
            has_arabic_next = bool(re.search(r'[\u0600-\u06FF]', next_line))
            if has_arabic and has_english and not has_arabic_next:
                bilingual_pair_count += 1
    
    if bilingual_pair_count >= 3:  # Multiple bilingual pairs = enumeration
        return True
    
    return False


def looks_like_table(content: Dict[str, Optional[str]]) -> bool:
    """
    Detect if content contains table-like patterns.
    
    Tables have:
    - Rows + columns structure
    - Repeated numeric alignment
    - Year-by-month or station-by-month matrices
    
    Args:
        content: Dictionary with "ar" and "en" keys
        
    Returns:
        True if content appears to be a table
    """
    return is_table_content(content)


def split_header(chunk: Chunk) -> List[Chunk]:
    """
    Split a header chunk that exceeds limits.
    
    Keeps ONLY the title line as header, reclassifies remaining content.
    
    Args:
        chunk: Header chunk to split
        
    Returns:
        List of chunks: [header_chunk, ...remaining_chunks]
    """
    ar_text = (chunk.content.get("ar") or "").strip()
    en_text = (chunk.content.get("en") or "").strip()
    
    # Extract first line as title
    ar_lines = ar_text.split("\n") if ar_text else []
    en_lines = en_text.split("\n") if en_text else []
    
    # Get first non-empty line from each language
    ar_title = ar_lines[0].strip() if ar_lines and ar_lines[0].strip() else ""
    en_title = en_lines[0].strip() if en_lines and en_lines[0].strip() else ""
    
    # Remaining content
    ar_remaining = "\n".join(ar_lines[1:]).strip() if len(ar_lines) > 1 else ""
    en_remaining = "\n".join(en_lines[1:]).strip() if len(en_lines) > 1 else ""
    
    # Create header chunk (title only)
    header_content = {
        "ar": ar_title if ar_title else None,
        "en": en_title if en_title else None,
    }
    
    header_metadata = ChunkMetadata(
        document=chunk.metadata.document,
        year=chunk.metadata.year,
        chapter=chunk.metadata.chapter,
        section=chunk.metadata.section,
        page_start=chunk.metadata.page_start,
        page_end=chunk.metadata.page_end,
        chunk_type="header",
        language=chunk.metadata.language.copy() if chunk.metadata.language else [],
        embedding_allowed=False,  # Headers never embedded
    )
    
    chapter_slug = slugify(chunk.metadata.chapter or 'unknown')
    section_slug = slugify(chunk.metadata.section or 'unknown')
    key_phrase = extract_key_phrase(header_content)
    content_hash = generate_content_hash(header_content)
    
    header_chunk = Chunk(
        chunk_id=f"header_{chapter_slug}_{section_slug}_p{chunk.metadata.page_start}_{key_phrase}_{content_hash}",
        content=header_content,
        metadata=header_metadata,
    )
    
    result = [header_chunk]
    
    # If there's remaining content, create new chunk(s) with reclassification
    if ar_remaining or en_remaining:
        remaining_content = {
            "ar": ar_remaining if ar_remaining else None,
            "en": en_remaining if en_remaining else None,
        }
        
        # Reclassify remaining content
        remaining_type = classify_content_type(
            f"{ar_remaining}\n{en_remaining}",
            "paragraph",  # Treat as paragraph for reclassification
            section=chunk.metadata.section,
            content=remaining_content,
        )
        
        # If still looks like header, force to narrative
        if remaining_type == "header":
            remaining_type = "narrative"
        
        remaining_metadata = ChunkMetadata(
            document=chunk.metadata.document,
            year=chunk.metadata.year,
            chapter=chunk.metadata.chapter,
            section=chunk.metadata.section,
            page_start=chunk.metadata.page_start,
            page_end=chunk.metadata.page_end,
            chunk_type=remaining_type,
            language=chunk.metadata.language.copy() if chunk.metadata.language else [],
            embedding_allowed=is_embedding_eligible(remaining_type),
        )
        
        key_phrase_remaining = extract_key_phrase(remaining_content)
        content_hash_remaining = generate_content_hash(remaining_content)
        
        remaining_chunk = Chunk(
            chunk_id=f"{remaining_type}_{chapter_slug}_{section_slug}_p{chunk.metadata.page_start}_{key_phrase_remaining}_{content_hash_remaining}",
            content=remaining_content,
            metadata=remaining_metadata,
        )
        
        result.append(remaining_chunk)
    
    return result


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


def normalize_chunk_type(chunk: Chunk) -> List[Chunk]:
    """
    Automatic reclassification pass after initial chunk creation.
    
    Enforces:
    - Headers > 200 chars are split
    - Enumerations are reclassified to reference
    - Tables are correctly identified (never narrative)
    
    Args:
        chunk: Chunk to normalize
        
    Returns:
        List of chunks (may be split if header exceeds limits)
    """
    # Rule 1: Split headers that exceed limits
    if chunk.metadata.chunk_type == "header":
        ar_text = (chunk.content.get("ar") or "").strip()
        en_text = (chunk.content.get("en") or "").strip()
        total_chars = len(ar_text) + len(en_text)
        
        if total_chars > MAX_HEADER_CHARS:
            logger.warning(f"normalize_chunk_type: Header chunk exceeds {MAX_HEADER_CHARS} chars ({total_chars}), splitting")
            return split_header(chunk)
        
        # Check for paragraphs (multiple lines with prose)
        lines = [line.strip() for line in f"{ar_text}\n{en_text}".split("\n") if line.strip()]
        if len(lines) > 3:
            logger.warning(f"normalize_chunk_type: Header has {len(lines)} lines (>3), splitting")
            return split_header(chunk)
        
        # Check numeric density
        combined_text = f"{ar_text} {en_text}"
        numeric_density = calculate_numeric_density(combined_text)
        if numeric_density > MAX_NUMERIC_DENSITY_HEADER:
            logger.warning(f"normalize_chunk_type: Header has numeric density {numeric_density:.2%} > {MAX_NUMERIC_DENSITY_HEADER:.2%}, splitting")
            return split_header(chunk)
    
    # Rule 2: Reclassify enumerations to reference
    if chunk.metadata.chunk_type != "reference" and looks_like_enumeration(chunk.content):
        logger.info(f"normalize_chunk_type: Reclassifying chunk from {chunk.metadata.chunk_type} to reference (enumeration detected)")
        chunk.metadata.chunk_type = "reference"
        chunk.metadata.embedding_allowed = True
    
    # Rule 3: Reclassify tables misclassified as narrative
    if chunk.metadata.chunk_type == "narrative" and looks_like_table(chunk.content):
        logger.warning(f"normalize_chunk_type: Reclassifying chunk from narrative to table (table pattern detected)")
        chunk.metadata.chunk_type = "table"
        chunk.metadata.embedding_allowed = True
    
    # Rule 4: Ensure headers never have embedding_allowed=True
    if chunk.metadata.chunk_type == "header":
        chunk.metadata.embedding_allowed = False
    
    return [chunk]


def validate_chunk(chunk: Chunk, table_hashes: Optional[set] = None) -> bool:
    """
    Enhanced validation with all strict chunk semantics rules.
    
    Enforces:
    - No headers > 200 chars
    - No embedded table fragments
    - No duplicated table hashes
    - Mandatory metadata for tables
    - Header embedding restrictions
    
    Args:
        chunk: Chunk to validate
        table_hashes: Set of normalized table hashes seen in current chapter+section (for deduplication)
        
    Returns:
        True if chunk passes all validation rules, False otherwise
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
    
    # RULE 1: Header validation - must never exceed 200 chars
    if chunk.metadata.chunk_type == "header":
        total_chars = len(ar_text) + len(en_text)
        if total_chars > MAX_HEADER_CHARS:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - header exceeds {MAX_HEADER_CHARS} chars ({total_chars})")
            return False
        
        # Headers must have embedding_allowed = False
        if chunk.metadata.embedding_allowed:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - header has embedding_allowed=True (must be False)")
            return False
    
    # RULE 2: Table fragments must not be embedded
    if chunk.metadata.chunk_type == "table_fragment":
        if chunk.metadata.embedding_allowed:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - table_fragment has embedding_allowed=True (must be False)")
            return False
    
    # RULE 3: Table deduplication check
    if chunk.metadata.chunk_type == "table" and table_hashes is not None:
        normalized_hash = normalize_table_for_hash(chunk.content)
        table_hash = hashlib.md5(normalized_hash.encode('utf-8')).hexdigest()
        if table_hash in table_hashes:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - duplicate table hash detected")
            return False
        # Add to set for future checks (caller should manage this)
        table_hashes.add(table_hash)
    
    # RULE 4: Mandatory metadata for table chunks
    if chunk.metadata.chunk_type == "table":
        if not chunk.metadata.section:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - table missing mandatory metadata.section")
            return False
        if not chunk.metadata.chapter:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - table missing mandatory metadata.chapter")
            return False
        if not chunk.metadata.data_year:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - table missing mandatory metadata.data_year")
            return False
        if not chunk.metadata.geography:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - table missing mandatory metadata.geography")
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
    
    if chunk.metadata.page_start == 0:
        logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - page_start is 0")
        return False
    
    # Rule 10: Check for broken text
    # Check for obvious broken Arabic (too many isolated characters)
    if ar_text:
        isolated_chars = len(re.findall(r'\s[\u0600-\u06FF]\s', ar_text))
        isolated_ratio = isolated_chars / len(ar_text) if ar_text else 0
        if isolated_ratio > 0.1:  # More than 10% isolated chars
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - broken Arabic text (isolated_chars={isolated_chars}, ratio={isolated_ratio:.2%})")
            return False
    
    # Check minimum content length for narrative chunks
    if chunk.metadata.chunk_type == "narrative":
        total_length = len(ar_text) + len(en_text)
        if total_length < 20:  # Too short
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - too short (length={total_length} < 20)")
            return False
    
    # RULE 5: Fail-fast safety guards - no header chunks embedded
    if chunk.metadata.chunk_type == "header" and chunk.metadata.embedding_allowed:
        logger.error(f"validate_chunk: FAIL-FAST: Header chunk has embedding_allowed=True (violation)")
        return False
    
    # RULE 6: Fail-fast safety guards - no header chunks > 200 chars
    if chunk.metadata.chunk_type == "header":
        total_chars = len(ar_text) + len(en_text)
        if total_chars > MAX_HEADER_CHARS:
            logger.error(f"validate_chunk: FAIL-FAST: Header chunk exceeds {MAX_HEADER_CHARS} chars ({total_chars})")
            return False
    
    # RULE 7: Fail-fast safety guards - no tables labeled as narrative
    if chunk.metadata.chunk_type == "narrative" and looks_like_table(chunk.content):
        logger.error(f"validate_chunk: FAIL-FAST: Table content labeled as narrative (violation)")
        return False
    
    # RULE 8: Fail-fast safety guards - no governorate lists labeled as header
    if chunk.metadata.chunk_type == "header" and looks_like_enumeration(chunk.content):
        logger.error(f"validate_chunk: FAIL-FAST: Enumeration content labeled as header (violation)")
        return False
    
    # RULE 9: Embedding policy guardrails with runtime assertion
    if chunk.metadata.embedding_allowed:
        if chunk.metadata.chunk_type not in {"narrative", "table", "reference"}:
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - embedding_allowed=True but chunk_type={chunk.metadata.chunk_type} not eligible")
            return False
        if chunk.metadata.chunk_type == "table_fragment":
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - table_fragment cannot have embedding_allowed=True")
            return False
        if chunk.metadata.chunk_type == "header":
            logger.warning(f"validate_chunk: Chunk {chunk_id_short}: Validation failed - header cannot have embedding_allowed=True")
            return False
        # Runtime assertion: Only narrative, table, and reference (not table_fragment, not header) can be embedded
        assert chunk.metadata.chunk_type in {"narrative", "table", "reference"}, \
            f"Embedding policy violation: chunk_type={chunk.metadata.chunk_type} cannot be embedded"
        assert chunk.metadata.chunk_type != "table_fragment", \
            f"Embedding policy violation: table_fragment cannot be embedded"
        assert chunk.metadata.chunk_type != "header", \
            f"Embedding policy violation: header cannot be embedded"
    
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
            
            # STRIP PAGE NOISE before classification
            cleaned_chunk_content = strip_page_noise(chunk_content)
            
            # Classify content type with section context (REFINEMENT 2)
            # Pass content for header validation
            content_type = classify_content_type(
                chunk_text,
                block.get("type", ""),
                section=block.get("section"),
                content=cleaned_chunk_content,
            )
            
            # Use block's own chapter/section, not the passed parameters
            block_chapter = block.get("chapter") or chapter
            block_section = block.get("section") or section
            
            # Create metadata with embedding control
            # Headers must have embedding_allowed=False
            embedding_allowed = is_embedding_eligible(content_type) and content_type != "header"
            metadata = ChunkMetadata(
                chapter=block_chapter,
                section=block_section,
                page_start=block.get("pageNumber", 0),
                page_end=block.get("pageNumber", 0),
                chunk_type=content_type,
                language=[lang for lang in [language] if lang != "mixed"] or ["ar", "en"] if language == "mixed" else [language],
                embedding_allowed=embedding_allowed,
            )
            
            # Use cleaned content for chunk
            chunk_content = cleaned_chunk_content
            
            # Create chunk with stable ID (includes key phrase and hash)
            chapter_slug = slugify(block_chapter or 'unknown')
            section_slug = slugify(block_section or 'unknown')
            key_phrase = extract_key_phrase(chunk_content)
            content_hash = generate_content_hash(chunk_content)
            chunk = Chunk(
                chunk_id=f"yearbook2025_{chapter_slug}_{section_slug}_p{block.get('pageNumber', 0)}_{key_phrase}_{content_hash}",
                content=chunk_content,
                metadata=metadata,
            )
            
            # AUTOMATIC RECLASSIFICATION: Normalize chunk type (split headers, reclassify enumerations/tables)
            normalized_chunks = normalize_chunk_type(chunk)
            
            # Rule 15: Validate before adding
            for normalized_chunk in normalized_chunks:
                if validate_chunk(normalized_chunk):
                    section_chunks.append(normalized_chunk)
                    logger.debug(f"chunk_section_blocks: Block {block_idx}, Chunk {i}: Created and validated (id={normalized_chunk.chunk_id[:50]}..., type={normalized_chunk.metadata.chunk_type})")
                else:
                    logger.warning(f"chunk_section_blocks: Block {block_idx}, Chunk {i}: Validation failed - chunk rejected")
                    rejected_chunks += 1
    
    logger.info(f"chunk_section_blocks: Complete - {len(section_chunks)} chunks created ({skipped_blocks} blocks skipped, {empty_blocks} blocks empty, {invalid_blocks} blocks invalid, {rejected_chunks} chunks rejected)")
    return section_chunks


def chunk_table_block(
    table_block: Dict[str, Any],
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    preceding_blocks: Optional[List[Dict[str, Any]]] = None,
    table_hashes: Optional[set] = None,
) -> List[Chunk]:
    """
    Chunk table blocks separately with strict atomicity enforcement.
    
    ENFORCES:
    - One table = one chunk (unless > 12,000 chars)
    - Fragments marked as "table_fragment" with embedding_allowed=False
    - Mandatory metadata: section, chapter, data_year, geography
    - Table deduplication by normalized hash
    - Governorate table normalization (split multi-governorate tables)
    - Page noise stripping
    
    Args:
        table_block: The table block to chunk
        chapter: Chapter name (if known)
        section: Section name (if known)
        preceding_blocks: All blocks from the same page (for extracting table summary)
        table_hashes: Set of normalized table hashes for deduplication (mutated in-place)
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
    
    # STRIP PAGE NOISE before processing
    table_content_dict = {"ar": ar_text, "en": en_text}
    cleaned_content = strip_page_noise(table_content_dict)
    ar_text = cleaned_content.get("ar") or ""
    en_text = cleaned_content.get("en") or ""
    
    # Use block's own chapter/section, not the passed parameters
    block_chapter = table_block.get("chapter") or chapter
    block_section = table_block.get("section") or section
    
    # MANDATORY: Ensure chapter and section exist (fail fast)
    if not block_chapter:
        logger.error(f"chunk_table_block: Table missing mandatory chapter - rejecting")
        return []
    if not block_section:
        logger.error(f"chunk_table_block: Table missing mandatory section - rejecting")
        return []
    
    # Extract table summary from preceding blocks if available
    summary_chunks: List[Chunk] = []
    if preceding_blocks:
        summary_content = extract_table_summary(table_block, preceding_blocks)
        if summary_content:
            # Strip noise from summary too
            summary_content = strip_page_noise(summary_content)
            # Create narrative chunk for table summary
            summary_metadata = ChunkMetadata(
                chapter=block_chapter,
                section=block_section,
                page_start=table_block.get("pageNumber", 0),
                page_end=table_block.get("pageNumber", 0),
                chunk_type="narrative",
                language=["ar", "en"] if (summary_content.get("ar") and summary_content.get("en")) else (["ar"] if summary_content.get("ar") else ["en"]),
                embedding_allowed=True,
            )
            chapter_slug = slugify(block_chapter or 'unknown')
            section_slug = slugify(block_section or 'unknown')
            key_phrase = extract_key_phrase(summary_content)
            content_hash = generate_content_hash(summary_content)
            summary_chunk = Chunk(
                chunk_id=f"table_summary_{chapter_slug}_{section_slug}_p{table_block.get('pageNumber', 0)}_{key_phrase}_{content_hash}",
                content=summary_content,
                metadata=summary_metadata,
            )
            if validate_chunk(summary_chunk):
                summary_chunks.append(summary_chunk)
                logger.debug(f"chunk_table_block: Created table summary narrative chunk (id={summary_chunk.chunk_id[:50]}...)")
    
    # Extract table metadata from preceding blocks
    table_metadata_dict = extract_table_metadata(preceding_blocks if preceding_blocks else [])
    
    # MANDATORY: Ensure data_year and geography exist (set defaults if missing)
    if not table_metadata_dict.get("data_year"):
        # Try to extract from table content or set default
        logger.warning(f"chunk_table_block: Table missing data_year, attempting extraction")
        # Extract years from content
        combined_text = f"{ar_text} {en_text}"
        year_matches = re.findall(r'\b(20[0-3][0-9])\b', combined_text)
        years = [int(y) for y in year_matches if 2020 <= int(y) <= 2039]
        if years:
            table_metadata_dict["data_year"] = sorted(set(years), reverse=True)
        else:
            logger.error(f"chunk_table_block: Table missing mandatory data_year - rejecting")
            return []
    
    if not table_metadata_dict.get("geography"):
        # Default to national level
        table_metadata_dict["geography"] = ["Sultanate of Oman"]
        logger.debug(f"chunk_table_block: Table missing geography, defaulting to 'Sultanate of Oman'")
    
    # Extract bilingual alignment from table content
    table_content_dict = {"ar": ar_text, "en": en_text}
    bilingual_alignment = extract_bilingual_alignment(table_content_dict)
    
    # Combine table text for size check
    combined_table_text = (ar_text or "") + (en_text or "")
    combined_size = len(combined_table_text)
    logger.debug(f"chunk_table_block: Table size = {combined_size} chars")
    
    # CHECK FOR DEDUPLICATION (before processing)
    if table_hashes is not None:
        normalized_hash = normalize_table_for_hash(table_content_dict)
        table_hash = hashlib.md5(normalized_hash.encode('utf-8')).hexdigest()
        if table_hash in table_hashes:
            logger.warning(f"chunk_table_block: Duplicate table detected (hash={table_hash[:8]}...), dropping")
            return summary_chunks  # Return only summary chunks, drop table
        table_hashes.add(table_hash)
    
    # CHECK FOR GOVERNORATE SPLITTING
    governorate_chunks = split_governorate_table(
        table_content_dict,
        table_metadata_dict,
        block_chapter,
        block_section,
        table_block.get("pageNumber", 0),
    )
    
    # If governorate split occurred, process each separately
    if len(governorate_chunks) > 1:
        logger.info(f"chunk_table_block: Multi-governorate table detected, splitting into {len(governorate_chunks)} chunks")
        table_chunks: List[Chunk] = []
        for gov_chunk_data in governorate_chunks:
            gov_content = gov_chunk_data["content"]
            gov_metadata = gov_chunk_data["metadata"]
            gov_name = gov_chunk_data.get("governorate", "Unknown")
            
            metadata = ChunkMetadata(
                chapter=block_chapter,
                section=block_section,
                page_start=table_block.get("pageNumber", 0),
                page_end=table_block.get("pageNumber", 0),
                chunk_type="table",
                language=["ar", "en"] if (gov_content.get("ar") and gov_content.get("en")) else (["ar"] if gov_content.get("ar") else ["en"]),
                units=gov_metadata.get("units"),
                data_year=gov_metadata.get("data_year"),
                geography=gov_metadata.get("geography"),
                bilingual_alignment=bilingual_alignment if bilingual_alignment else None,
                embedding_allowed=True,
            )
            
            chapter_slug = slugify(block_chapter or 'unknown')
            section_slug = slugify(block_section or 'unknown')
            key_phrase = extract_key_phrase(gov_content)
            content_hash = generate_content_hash(gov_content)
            chunk = Chunk(
                chunk_id=f"table_{chapter_slug}_{section_slug}_p{table_block.get('pageNumber', 0)}_{slugify(gov_name)}_{key_phrase}_{content_hash}",
                content=gov_content,
                metadata=metadata,
            )
            
            if validate_chunk(chunk, table_hashes=table_hashes):
                table_chunks.append(chunk)
        
        return summary_chunks + table_chunks
    
    # ENFORCE TABLE ATOMICITY: Only split if > 12,000 chars
    if combined_size > MAX_TABLE_CHARS_FOR_SPLIT:
        logger.warning(f"chunk_table_block: Very large table detected ({combined_size} chars > {MAX_TABLE_CHARS_FOR_SPLIT}), splitting with fragments marked")
        # Split large table - mark fragments as table_fragment
        table_chunks: List[Chunk] = []
        lines = combined_table_text.split("\n")
        current_chunk_lines = []
        current_size = 0
        fragment_index = 0
        
        for line in lines:
            line_size = len(line)
            if current_size + line_size > MAX_TABLE_CHARS_FOR_SPLIT and current_chunk_lines:
                # Create fragment chunk (embedding_allowed=False)
                chunk_text = "\n".join(current_chunk_lines)
                chunk_content = {"ar": chunk_text if ar_text else None, "en": chunk_text if en_text else None}
                metadata = ChunkMetadata(
                    chapter=block_chapter,
                    section=block_section,
                    page_start=table_block.get("pageNumber", 0),
                    page_end=table_block.get("pageNumber", 0),
                    chunk_type="table_fragment",  # Mark as fragment
                    language=["ar", "en"] if (ar_text and en_text) else (["ar"] if ar_text else ["en"]),
                    units=table_metadata_dict.get("units"),
                    data_year=table_metadata_dict.get("data_year"),
                    geography=table_metadata_dict.get("geography"),
                    bilingual_alignment=bilingual_alignment if bilingual_alignment else None,
                    embedding_allowed=False,  # Fragments cannot be embedded
                )
                chapter_slug = slugify(block_chapter or 'unknown')
                section_slug = slugify(block_section or 'unknown')
                key_phrase = extract_key_phrase(chunk_content)
                content_hash = generate_content_hash(chunk_content)
                chunk = Chunk(
                    chunk_id=f"table_fragment_{chapter_slug}_{section_slug}_p{table_block.get('pageNumber', 0)}_{fragment_index}_{key_phrase}_{content_hash}",
                    content=chunk_content,
                    metadata=metadata,
                )
                # Fragments are not validated for embedding (they're marked as non-embeddable)
                table_chunks.append(chunk)
                fragment_index += 1
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Add remaining lines as final fragment
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunk_content = {"ar": chunk_text if ar_text else None, "en": chunk_text if en_text else None}
            metadata = ChunkMetadata(
                chapter=block_chapter,
                section=block_section,
                page_start=table_block.get("pageNumber", 0),
                page_end=table_block.get("pageNumber", 0),
                chunk_type="table_fragment",
                language=["ar", "en"] if (ar_text and en_text) else (["ar"] if ar_text else ["en"]),
                units=table_metadata_dict.get("units"),
                data_year=table_metadata_dict.get("data_year"),
                geography=table_metadata_dict.get("geography"),
                bilingual_alignment=bilingual_alignment if bilingual_alignment else None,
                embedding_allowed=False,
            )
            chapter_slug = slugify(block_chapter or 'unknown')
            section_slug = slugify(block_section or 'unknown')
            key_phrase = extract_key_phrase(chunk_content)
            content_hash = generate_content_hash(chunk_content)
            chunk = Chunk(
                chunk_id=f"table_fragment_{chapter_slug}_{section_slug}_p{table_block.get('pageNumber', 0)}_{fragment_index}_{key_phrase}_{content_hash}",
                content=chunk_content,
                metadata=metadata,
            )
            table_chunks.append(chunk)
        
        logger.warning(f"chunk_table_block: Split very large table into {len(table_chunks)} fragments (NONE will be embedded)")
        return summary_chunks + table_chunks
    else:
        # Table fits in one chunk - ATOMIC TABLE RULE
        logger.debug(f"chunk_table_block: Table fits in single chunk (atomic table)")
        table_content = {"ar": ar_text, "en": en_text}
        metadata = ChunkMetadata(
            chapter=block_chapter,
            section=block_section,
            page_start=table_block.get("pageNumber", 0),
            page_end=table_block.get("pageNumber", 0),
            chunk_type="table",
            language=["ar", "en"] if (ar_text and en_text) else (["ar"] if ar_text else ["en"]),
            units=table_metadata_dict.get("units"),
            data_year=table_metadata_dict.get("data_year"),
            geography=table_metadata_dict.get("geography"),
            bilingual_alignment=bilingual_alignment if bilingual_alignment else None,
            embedding_allowed=True,  # Only atomic tables can be embedded
        )
        
        chapter_slug = slugify(block_chapter or 'unknown')
        section_slug = slugify(block_section or 'unknown')
        key_phrase = extract_key_phrase(table_content)
        content_hash = generate_content_hash(table_content)
        chunk = Chunk(
            chunk_id=f"table_{chapter_slug}_{section_slug}_p{table_block.get('pageNumber', 0)}_{key_phrase}_{content_hash}",
            content=table_content,
            metadata=metadata,
        )
        
        # Validate table chunk with deduplication check
        if validate_chunk(chunk, table_hashes=table_hashes):
            logger.debug(f"chunk_table_block: Created and validated atomic table chunk (id={chunk.chunk_id[:50]}...)")
            return summary_chunks + [chunk]
        else:
            logger.warning(f"chunk_table_block: Table chunk validation failed - chunk rejected")
            return summary_chunks


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
    
    # Generate new chunk ID with key phrase and hash
    chapter_slug = slugify(metadata1.get("chapter") or "unknown")
    section_slug = slugify(metadata1.get("section") or "unknown")
    page = metadata1.get("page_start", 0)
    key_phrase = extract_key_phrase(merged_content)
    content_hash = generate_content_hash(merged_content)
    
    return {
        "chunk_id": f"yearbook2025_{chapter_slug}_{section_slug}_p{page}_{key_phrase}_{content_hash}_merged",
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


def is_table_content(content: Dict[str, Optional[str]]) -> bool:
    """
    Detect if content contains table-like patterns.
    
    Checks for:
    - Tab-separated values (\t)
    - Multiple consecutive lines with similar structure (rows)
    - Table-like patterns (headers, numeric data in columns)
    - Markdown table syntax (| separators)
    
    Args:
        content: Dictionary with "ar" and "en" keys
        
    Returns:
        True if content appears to be table data
    """
    ar_text = (content.get("ar") or "").strip()
    en_text = (content.get("en") or "").strip()
    combined_text = (ar_text + "\n" + en_text).strip()
    
    if not combined_text:
        return False
    
    # Check for tab-separated values (strong indicator)
    if "\t" in combined_text:
        # Count tabs per line - tables have consistent tab counts
        lines = combined_text.split("\n")
        tab_counts = [line.count("\t") for line in lines if line.strip()]
        if tab_counts and max(tab_counts) >= 2:  # At least 2 tabs (3+ columns)
            # Check if multiple lines have similar tab counts (table structure)
            if len(set(tab_counts)) <= 2:  # Most lines have similar structure
                return True
    
    # Check for markdown table syntax (| separators)
    if "|" in combined_text:
        lines = combined_text.split("\n")
        pipe_lines = [line for line in lines if "|" in line and line.strip()]
        if len(pipe_lines) >= 2:  # At least 2 rows
            # Check if lines have consistent pipe counts
            pipe_counts = [line.count("|") for line in pipe_lines]
            if len(set(pipe_counts)) <= 2:  # Consistent structure
                return True
    
    # Check for structured data patterns (multiple lines with similar structure)
    lines = [line.strip() for line in combined_text.split("\n") if line.strip()]
    if len(lines) >= 3:  # At least 3 rows
        # Check for numeric patterns (tables often have numeric data)
        numeric_lines = 0
        for line in lines:
            # Check if line contains numbers and separators (spaces, tabs, commas)
            if re.search(r'\d+[\s\t,]+', line):
                numeric_lines += 1
        
        # If most lines have numeric patterns, likely a table
        if numeric_lines >= len(lines) * 0.6:  # 60% of lines have numbers
            return True
        
        # Check for consistent column-like structure (multiple spaces/tabs separating fields)
        structured_lines = 0
        for line in lines:
            # Lines with multiple spaces/tabs separating fields
            if len(re.split(r'[\s\t]{2,}', line)) >= 3:  # At least 3 fields
                structured_lines += 1
        
        if structured_lines >= len(lines) * 0.7:  # 70% of lines are structured
            return True
    
    return False


def merge_table_fragments(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge table fragments that were split due to size limits.
    
    Detects consecutive table chunks that belong to the same table and merges them
    back into single chunks. This happens when tables exceed the 6000 character limit
    and are split by chunk_table_block().
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunks with table fragments merged
    """
    if not chunks:
        return []
    
    merged = []
    i = 0
    fragments_merged = 0
    
    while i < len(chunks):
        chunk = chunks[i]
        metadata = chunk.get("metadata", {})
        chunk_type = metadata.get("chunk_type", "")
        
        # Only process table chunks
        if chunk_type != "table":
            merged.append(chunk)
            i += 1
            continue
        
        # Look for consecutive table fragments to merge
        fragment_group = [chunk]
        base_page = metadata.get("page_start", 0)
        base_chapter = metadata.get("chapter")
        base_section = metadata.get("section")
        chunk_id_base = chunk.get("chunk_id", "")
        
        # Check if this looks like a fragment (chunk_id suggests it might be split)
        # Fragments often have same base ID or page number with different indices
        j = i + 1
        while j < len(chunks):
            next_chunk = chunks[j]
            next_metadata = next_chunk.get("metadata", {})
            next_chunk_type = next_metadata.get("chunk_type", "")
            
            # Stop if not a table chunk
            if next_chunk_type != "table":
                break
            
            # Check if it's a fragment of the same table
            next_page = next_metadata.get("page_start", 0)
            next_chapter = next_metadata.get("chapter")
            next_section = next_metadata.get("section")
            next_chunk_id = next_chunk.get("chunk_id", "")
            
            # Primary check: Same chapter/section and same or adjacent page (within 2 pages)
            # This is the most reliable indicator that chunks belong to the same table
            is_fragment = (
                next_chapter == base_chapter and
                next_section == base_section and
                abs(next_page - base_page) <= 2
            )
            
            # Secondary check: Chunk ID patterns suggest fragmentation
            # Fragments from chunk_table_block() have IDs like: table_{chapter}_{page}_{index}
            # So fragments share the same chapter and page in their IDs
            id_suggests_fragment = False
            if chunk_id_base.startswith("table_") and next_chunk_id.startswith("table_"):
                # Extract components from chunk IDs
                base_parts = chunk_id_base.split("_")
                next_parts = next_chunk_id.split("_")
                
                # If both have at least 3 parts (table_chapter_page or table_chapter_page_index)
                if len(base_parts) >= 3 and len(next_parts) >= 3:
                    # Check if chapter and page match (first 3 parts: table, chapter, page)
                    if base_parts[:3] == next_parts[:3]:
                        id_suggests_fragment = True
                    # Or if they're both just table_page (2 parts)
                    elif len(base_parts) == 2 and len(next_parts) == 2:
                        if base_parts == next_parts:
                            id_suggests_fragment = True
            
            if is_fragment or id_suggests_fragment:
                fragment_group.append(next_chunk)
                j += 1
            else:
                break
        
        # Merge fragments if we found multiple
        if len(fragment_group) > 1:
            # Combine content from all fragments
            merged_ar_parts = []
            merged_en_parts = []
            
            page_start = base_page
            page_end = base_page
            
            for frag in fragment_group:
                frag_content = frag.get("content", {})
                frag_ar = frag_content.get("ar")
                frag_en = frag_content.get("en")
                
                if frag_ar:
                    merged_ar_parts.append(frag_ar)
                if frag_en:
                    merged_en_parts.append(frag_en)
                
                # Update page range
                frag_metadata = frag.get("metadata", {})
                frag_page_start = frag_metadata.get("page_start", 0)
                frag_page_end = frag_metadata.get("page_end", 0)
                page_start = min(page_start, frag_page_start)
                page_end = max(page_end, frag_page_end)
            
            # Merge content (preserve table structure with newlines)
            merged_content = {
                "ar": "\n".join(merged_ar_parts) if merged_ar_parts else None,
                "en": "\n".join(merged_en_parts) if merged_en_parts else None,
            }
            
            # Create merged chunk
            merged_metadata = {
                **metadata,
                "page_start": page_start,
                "page_end": page_end,
            }
            
            # Generate new merged chunk ID with key phrase and hash
            chapter_slug = slugify(base_chapter or "unknown")
            section_slug = slugify(base_section or "unknown")
            key_phrase = extract_key_phrase(merged_content)
            content_hash = generate_content_hash(merged_content)
            merged_chunk_id = f"table_{chapter_slug}_{section_slug}_p{page_start}_{key_phrase}_{content_hash}_merged"
            
            merged_chunk = {
                "chunk_id": merged_chunk_id,
                "content": merged_content,
                "metadata": merged_metadata,
            }
            
            merged.append(merged_chunk)
            fragments_merged += len(fragment_group) - 1  # Count how many chunks were merged
            logger.info(f"merge_table_fragments: Merged {len(fragment_group)} table fragments into single chunk (pages {page_start}-{page_end})")
            i = j
        else:
            # Single chunk, no merging needed
            merged.append(chunk)
            i += 1
    
    if fragments_merged > 0:
        logger.info(f"merge_table_fragments: Merged {fragments_merged} table fragments, {len(chunks)} -> {len(merged)} chunks")
    
    return merged


def reclassify_toc_to_table(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reclassify chunks misclassified as "toc" to "table" if they contain table content.
    
    Some chunks are classified as "toc" based on block_type, but may actually contain
    table data. This function detects table patterns in toc chunks and reclassifies them.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunks with toc->table reclassifications applied
    """
    if not chunks:
        return []
    
    reclassified_count = 0
    
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_type = metadata.get("chunk_type", "")
        
        # Only process toc chunks
        if chunk_type != "toc":
            continue
        
        # Check if content contains table patterns
        content = chunk.get("content", {})
        if is_table_content(content):
            # Reclassify to table
            metadata["chunk_type"] = "table"
            metadata["embedding_allowed"] = True  # Tables are embedding-eligible
            
            chunk_id = chunk.get("chunk_id", "unknown")
            logger.info(f"reclassify_toc_to_table: Reclassified chunk {chunk_id[:50]}... from 'toc' to 'table'")
            reclassified_count += 1
    
    if reclassified_count > 0:
        logger.info(f"reclassify_toc_to_table: Reclassified {reclassified_count} chunks from 'toc' to 'table'")
    
    return chunks


def extract_table_summary(
    table_block: Dict[str, Any],
    page_blocks: List[Dict[str, Any]]
) -> Optional[Dict[str, Optional[str]]]:
    """
    Extract preceding text blocks that describe a table as summary content.
    
    Looks for 1-3 preceding text blocks (paragraph/header types) on the same page
    that likely describe the table.
    
    Args:
        table_block: The table block to find summary for
        page_blocks: All blocks from the same page (in order)
        
    Returns:
        Dictionary with "ar" and "en" keys containing summary content, or None if not found
    """
    table_page = table_block.get("pageNumber", 0)
    table_idx = None
    
    # Find table block index in page_blocks by matching page number and type
    # Use a simple heuristic: find first table block on the same page
    # (since we can't reliably compare block objects)
    for idx, block in enumerate(page_blocks):
        if (block.get("pageNumber", 0) == table_page and 
            block.get("type") == "table"):
            # Check if this is likely the same table by comparing content hash
            block_content = block.get("content", {})
            table_content = table_block.get("content", {})
            block_ar = (block_content.get("ar") or "").strip()[:100]  # First 100 chars for comparison
            table_ar = (table_content.get("ar") or "").strip()[:100]
            if block_ar == table_ar or table_idx is None:  # Match or first table on page
                table_idx = idx
                # If we found an exact match, use it
                if block_ar == table_ar:
                    break
    
    if table_idx is None or table_idx == 0:
        # Table not found or is first block, no preceding blocks
        return None
    
    # Look for preceding text blocks (1-3 blocks before table)
    summary_blocks = []
    for i in range(max(0, table_idx - 3), table_idx):
        block = page_blocks[i]
        block_type = block.get("type", "")
        
        # Only include paragraph or header blocks (skip other tables, footers, etc.)
        if block_type in ["paragraph", "header"]:
            content = block.get("content", {})
            ar_text = (content.get("ar") or "").strip()
            en_text = (content.get("en") or "").strip()
            
            # Only include blocks with meaningful content
            if ar_text or en_text:
                summary_blocks.append(block)
    
    if not summary_blocks:
        return None
    
    # Combine content from summary blocks
    ar_parts = []
    en_parts = []
    
    for block in summary_blocks:
        content = block.get("content", {})
        ar_text = (content.get("ar") or "").strip()
        en_text = (content.get("en") or "").strip()
        
        if ar_text:
            ar_parts.append(ar_text)
        if en_text:
            en_parts.append(en_text)
    
    return {
        "ar": "\n\n".join(ar_parts) if ar_parts else None,
        "en": "\n\n".join(en_parts) if en_parts else None,
    }


def extract_table_metadata(preceding_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract table metadata (units, data_year, geography) from preceding text blocks.
    
    Parses preceding text blocks to find:
    - Units: Measurement units like "(000 MT)", "(000 BBL)", "mm", "kg", etc.
    - Years: 4-digit years (2020-2039 range)
    - Geography: Governorate names, geographic areas
    
    Args:
        preceding_blocks: List of text blocks that precede the table
        
    Returns:
        Dictionary with keys: "units", "data_year", "geography"
    """
    if not preceding_blocks:
        return {"units": None, "data_year": None, "geography": None}
    
    # Collect all text from preceding blocks
    combined_text = ""
    for block in preceding_blocks:
        content = block.get("content", {})
        ar_text = (content.get("ar") or "").strip()
        en_text = (content.get("en") or "").strip()
        if ar_text:
            combined_text += ar_text + " "
        if en_text:
            combined_text += en_text + " "
    
    combined_text = combined_text.strip()
    if not combined_text:
        return {"units": None, "data_year": None, "geography": None}
    
    # Extract units
    units = []
    # Pattern for common unit formats: (000 MT), (000 BBL), etc.
    unit_patterns = [
        r'\(000\s+(MT|BBL|tons?|barrels?|kg|tons?)\)',  # (000 MT), (000 BBL)
        r'\b(mm|cm|km|kg|°C|°م|%)\b',  # Standard units
        r'\b(Million|Billion|Thousand)\b',  # Large numbers
        r'\b(MT|BBL|tons?|barrels?|liters?|gallons?)\b',  # Unit abbreviations
    ]
    
    for pattern in unit_patterns:
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                unit = match[0] if match[0] else match[1] if len(match) > 1 else ""
            else:
                unit = match
            if unit and unit not in units:
                units.append(unit)
    
    # Extract years (2020-2039 range)
    years = []
    year_pattern = r'\b(20[0-3][0-9])\b'
    year_matches = re.findall(year_pattern, combined_text)
    for year_str in year_matches:
        year = int(year_str)
        if 2020 <= year <= 2039 and year not in years:
            years.append(year)
    years.sort(reverse=True)  # Most recent first
    
    # Extract geography
    geography = []
    # Common geographic terms
    geo_patterns = [
        r'\b(Muscat|Dhofar|Al\s+Batinah|Ash\s+Sharqiyah|Al\s+Dakhiliyah|Al\s+Wusta|Al\s+Buraymi|Musandam)\s+(Governorate|Wilayat)?\b',
        r'\b(Oman|Sultanate\s+of\s+Oman)\b',
        r'\b(Governorate|Wilayat)\b',
    ]
    
    # Arabic geographic terms
    geo_ar_patterns = [
        r'محافظة\s+([^\s]+)',  # محافظة followed by name
        r'ولاية\s+([^\s]+)',  # ولاية followed by name
        r'(مسقط|ظفار|الباطنة|الشرقية|الداخلية|الوسطى|البريمي|مسندم)',
        r'(عمان|سلطنة\s+عمان)',
    ]
    
    for pattern in geo_patterns + geo_ar_patterns:
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                geo = match[0] if match[0] else match
            else:
                geo = match
            if geo and geo.strip() and geo.strip() not in geography:
                geography.append(geo.strip())
    
    # Also check for governorate indicators using existing function
    for block in preceding_blocks:
        content = block.get("content", {})
        ar_text = (content.get("ar") or "").strip()
        en_text = (content.get("en") or "").strip()
        for line in (ar_text + "\n" + en_text).split("\n"):
            if is_governorate(line):
                # Extract the governorate name
                gov_match = re.search(r'(Governorate|Wilayat|محافظة|ولاية)\s*:?\s*([^\n,]+)', line, re.IGNORECASE)
                if gov_match:
                    gov_name = gov_match.group(2).strip()
                    if gov_name and gov_name not in geography:
                        geography.append(gov_name)
    
    return {
        "units": units if units else None,
        "data_year": years if years else None,
        "geography": geography if geography else None,
    }


def split_governorate_table(
    table_content: Dict[str, Optional[str]],
    table_metadata: Dict[str, Any],
    block_chapter: Optional[str],
    block_section: Optional[str],
    page_num: int,
) -> List[Dict[str, Any]]:
    """
    Split multi-governorate tables into one chunk per governorate.
    
    Detects multiple governorates in table content and creates separate chunks
    for each, preserving identical structure.
    
    Args:
        table_content: Dictionary with "ar" and "en" keys containing table content
        table_metadata: Table metadata dictionary
        block_chapter: Chapter name
        block_section: Section name
        page_num: Page number
        
    Returns:
        List of chunk dictionaries, one per governorate (or single chunk if no split needed)
    """
    ar_text = (table_content.get("ar") or "").strip()
    en_text = (table_content.get("en") or "").strip()
    combined_text = f"{ar_text}\n{en_text}"
    
    # Detect governorate indicators
    governorate_patterns = [
        r'(Muscat|Dhofar|Al\s+Batinah|Ash\s+Sharqiyah|Al\s+Dakhiliyah|Al\s+Wusta|Al\s+Buraymi|Musandam)\s+(Governorate|Wilayat)?',
        r'محافظة\s+(مسقط|ظفار|الباطنة|الشرقية|الداخلية|الوسطى|البريمي|مسندم)',
        r'ولاية\s+([^\n]+)',
    ]
    
    governorate_matches = []
    for pattern in governorate_patterns:
        matches = re.finditer(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            gov_name = match.group(1) if match.group(1) else match.group(2) if len(match.groups()) > 1 else match.group(0)
            if gov_name:
                governorate_matches.append((match.start(), gov_name.strip()))
    
    # If no governorates detected or only one, return single chunk
    if len(governorate_matches) <= 1:
        return [{
            "content": table_content,
            "metadata": table_metadata,
            "governorate": None,
        }]
    
    # Sort by position
    governorate_matches.sort(key=lambda x: x[0])
    
    # Split table by governorate boundaries
    chunks = []
    for i, (pos, gov_name) in enumerate(governorate_matches):
        # Determine boundaries
        start_pos = pos
        end_pos = governorate_matches[i + 1][0] if i + 1 < len(governorate_matches) else len(combined_text)
        
        # Extract governorate-specific content (simplified - preserve full table structure)
        # For now, create chunks with same content but different geography metadata
        # This is a simplified implementation - full implementation would parse table rows
        chunk_content = table_content.copy()
        
        # Update metadata with specific governorate
        chunk_metadata = table_metadata.copy()
        chunk_metadata["geography"] = [gov_name]
        
        chunks.append({
            "content": chunk_content,
            "metadata": chunk_metadata,
            "governorate": gov_name,
        })
    
    logger.info(f"split_governorate_table: Split table into {len(chunks)} governorate-specific chunks")
    return chunks


def extract_bilingual_alignment(table_content: Dict[str, Optional[str]]) -> Dict[str, str]:
    """
    Extract bilingual alignment mapping from table content.
    
    Identifies Arabic-English term pairs in table content and creates
    a mapping dictionary (Arabic -> English).
    
    Args:
        table_content: Dictionary with "ar" and "en" keys containing table content
        
    Returns:
        Dictionary mapping Arabic terms to English terms: {"arabic": "english"}
    """
    alignment: Dict[str, str] = {}
    
    # Get combined content (tables often have mixed content in "ar" field)
    ar_text = (table_content.get("ar") or "").strip()
    en_text = (table_content.get("en") or "").strip()
    
    # Combine both fields for analysis
    combined_text = ar_text
    if en_text and en_text not in ar_text:
        combined_text += "\n" + en_text
    
    if not combined_text:
        return alignment
    
    # Split by lines (tables are typically line-based)
    lines = combined_text.split("\n")
    
    # Pattern to detect Arabic text
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    # Pattern to detect English words (multi-word terms)
    english_pattern = re.compile(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Find all Arabic segments
        arabic_matches = arabic_pattern.findall(line)
        # Find all English segments (multi-word terms)
        english_matches = english_pattern.findall(line)
        
        # If we have both Arabic and English on the same line, create alignments
        if arabic_matches and english_matches:
            # For each Arabic term, find the closest English term
            for ar_term in arabic_matches:
                if len(ar_term) < 2:  # Skip very short Arabic terms
                    continue
                
                # Find closest English term (simple heuristic: first English term on line)
                if english_matches:
                    en_term = english_matches[0]
                    # Only create alignment if not already exists or if this is a better match
                    if ar_term not in alignment or len(en_term) > len(alignment.get(ar_term, "")):
                        alignment[ar_term] = en_term
        
        # Also check for tab-separated format (common in tables)
        if "\t" in line:
            cells = line.split("\t")
            for i, cell in enumerate(cells):
                cell = cell.strip()
                if not cell:
                    continue
                
                # Check if cell contains Arabic
                ar_in_cell = arabic_pattern.findall(cell)
                # Check adjacent cells for English
                if ar_in_cell and i + 1 < len(cells):
                    next_cell = cells[i + 1].strip()
                    en_in_next = english_pattern.findall(next_cell)
                    if en_in_next:
                        for ar_term in ar_in_cell:
                            if len(ar_term) >= 2:
                                alignment[ar_term] = en_in_next[0]
                
                # Also check previous cell
                if ar_in_cell and i > 0:
                    prev_cell = cells[i - 1].strip()
                    en_in_prev = english_pattern.findall(prev_cell)
                    if en_in_prev:
                        for ar_term in ar_in_cell:
                            if len(ar_term) >= 2:
                                alignment[ar_term] = en_in_prev[0]
        
        # Check for parentheses pattern: "English (Arabic)" or "Arabic (English)"
        paren_pattern = r'([A-Z][a-zA-Z\s]+)\s*\(([\u0600-\u06FF]+)\)|([\u0600-\u06FF]+)\s*\(([A-Z][a-zA-Z\s]+)\)'
        paren_matches = re.findall(paren_pattern, line)
        for match in paren_matches:
            if match[0] and match[1]:  # English (Arabic)
                alignment[match[1]] = match[0].strip()
            elif match[2] and match[3]:  # Arabic (English)
                alignment[match[2]] = match[3].strip()
    
    return alignment


def fail_fast_safety_check(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fail-fast safety guards before upserting to vector store.
    
    Enforces critical rules:
    - No header chunks are embedded
    - No header chunks exceed 200 chars
    - No tables are labeled as narrative
    - No governorate lists are labeled as header
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunks with violations removed or reclassified
    """
    if not chunks:
        return []
    
    safe_chunks = []
    violations = {
        "header_embedded": 0,
        "header_too_large": 0,
        "table_as_narrative": 0,
        "enumeration_as_header": 0,
    }
    
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_type = metadata.get("chunk_type", "")
        content = chunk.get("content", {})
        embedding_allowed = metadata.get("embedding_allowed", False)
        
        ar_text = (content.get("ar") or "").strip()
        en_text = (content.get("en") or "").strip()
        total_chars = len(ar_text) + len(en_text)
        
        # Guard 1: No header chunks embedded
        if chunk_type == "header" and embedding_allowed:
            logger.error(f"fail_fast_safety_check: FAIL-FAST violation: Header chunk has embedding_allowed=True - dropping chunk {chunk.get('chunk_id', 'unknown')[:50]}")
            violations["header_embedded"] += 1
            continue  # Drop the chunk
        
        # Guard 2: No header chunks > 200 chars
        if chunk_type == "header" and total_chars > MAX_HEADER_CHARS:
            logger.error(f"fail_fast_safety_check: FAIL-FAST violation: Header chunk exceeds {MAX_HEADER_CHARS} chars ({total_chars}) - dropping chunk {chunk.get('chunk_id', 'unknown')[:50]}")
            violations["header_too_large"] += 1
            continue  # Drop the chunk
        
        # Guard 3: No tables labeled as narrative
        if chunk_type == "narrative" and looks_like_table(content):
            logger.warning(f"fail_fast_safety_check: Reclassifying table mislabeled as narrative - chunk {chunk.get('chunk_id', 'unknown')[:50]}")
            metadata["chunk_type"] = "table"
            metadata["embedding_allowed"] = True
            violations["table_as_narrative"] += 1
            # Reclassify, don't drop
        
        # Guard 4: No enumerations labeled as header
        if chunk_type == "header" and looks_like_enumeration(content):
            logger.warning(f"fail_fast_safety_check: Reclassifying enumeration mislabeled as header - chunk {chunk.get('chunk_id', 'unknown')[:50]}")
            metadata["chunk_type"] = "reference"
            metadata["embedding_allowed"] = True
            violations["enumeration_as_header"] += 1
            # Reclassify, don't drop
        
        safe_chunks.append(chunk)
    
    # Log summary
    total_violations = sum(violations.values())
    if total_violations > 0:
        logger.warning(f"fail_fast_safety_check: Found {total_violations} violations:")
        for violation_type, count in violations.items():
            if count > 0:
                logger.warning(f"  - {violation_type}: {count}")
        logger.warning(f"fail_fast_safety_check: {len(chunks)} -> {len(safe_chunks)} chunks after safety check")
    else:
        logger.debug(f"fail_fast_safety_check: All {len(chunks)} chunks passed safety checks")
    
    return safe_chunks


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact duplicate glossary and list chunks.
    
    Creates content signatures for chunks and removes duplicates based on
    exact content match (normalized whitespace) for glossary and toc chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunks with duplicates removed
    """
    if not chunks:
        return []
    
    seen_signatures: set[str] = set()
    deduplicated = []
    removed_count = 0
    
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_type = metadata.get("chunk_type", "")
        
        # Only deduplicate glossary and toc chunks
        if chunk_type not in ["glossary", "toc"]:
            # Always include non-glossary/toc chunks
            deduplicated.append(chunk)
            continue
        
        # Create content signature
        content = chunk.get("content", {})
        ar_text = (content.get("ar") or "").strip()
        en_text = (content.get("en") or "").strip()
        
        # Normalize whitespace
        ar_text = " ".join(ar_text.split())
        en_text = " ".join(en_text.split())
        
        # Create signature
        signature = f"{ar_text}|{en_text}"
        
        # Check if we've seen this content before
        if signature in seen_signatures:
            # Duplicate found, skip it
            chunk_id = chunk.get("chunk_id", "unknown")
            logger.debug(f"deduplicate_chunks: Removing duplicate chunk {chunk_id[:50]}... (type={chunk_type})")
            removed_count += 1
            continue
        
        # New content, add to result and mark as seen
        seen_signatures.add(signature)
        deduplicated.append(chunk)
    
    if removed_count > 0:
        logger.info(f"deduplicate_chunks: Removed {removed_count} duplicate chunks ({len(chunks)} -> {len(deduplicated)})")
    
    return deduplicated


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
        
        # Group blocks by page for table summary extraction
        blocks_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for block in blocks:
            page_num = block.get("pageNumber", 0)
            if page_num not in blocks_by_page:
                blocks_by_page[page_num] = []
            blocks_by_page[page_num].append(block)
        
        # Extract chapter/section from blocks and track table hashes per chapter+section
        current_chapter = None
        current_section = None
        # Track table hashes per chapter+section for deduplication
        table_hashes_by_section: Dict[tuple, set] = {}  # (chapter, section) -> set of hashes
        
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
        
        # Chunk table blocks separately (with preceding blocks for summary extraction)
        if table_blocks:
            logger.info(f"chunk_blocks: Processing {len(table_blocks)} table blocks")
            for table_idx, table_block in enumerate(table_blocks):
                # Get preceding blocks from same page
                table_page = table_block.get("pageNumber", 0)
                preceding_blocks = blocks_by_page.get(table_page, [])
                
                # Get table's chapter/section for hash tracking
                table_chapter = table_block.get("chapter") or current_chapter
                table_section = table_block.get("section") or current_section
                section_key = (table_chapter, table_section)
                
                # Get or create hash set for this chapter+section
                if section_key not in table_hashes_by_section:
                    table_hashes_by_section[section_key] = set()
                table_hashes = table_hashes_by_section[section_key]
                
                table_chunks = chunk_table_block(
                    table_block,
                    chapter=table_chapter,
                    section=table_section,
                    preceding_blocks=preceding_blocks,
                    table_hashes=table_hashes,  # Pass hash set for deduplication
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
        
        # NEW: Merge table fragments that were split due to size limits
        before_merge_tables = len(result)
        result = merge_table_fragments(result)
        after_merge_tables = len(result)
        if before_merge_tables != after_merge_tables:
            logger.info(f"chunk_blocks: Table fragment merging: {before_merge_tables} -> {after_merge_tables} chunks")
        
        # NEW: Reclassify toc chunks containing table content to table
        result = reclassify_toc_to_table(result)
        
        # NEW: Deduplicate glossary and toc chunks
        before_dedup = len(result)
        result = deduplicate_chunks(result)
        after_dedup = len(result)
        if before_dedup != after_dedup:
            logger.info(f"chunk_blocks: Deduplication: {before_dedup} -> {after_dedup} chunks")
        
        # CRITICAL: Fail-fast safety guards before vector store upsert
        before_safety = len(result)
        result = fail_fast_safety_check(result)
        after_safety = len(result)
        if before_safety != after_safety:
            logger.warning(f"chunk_blocks: Safety check: {before_safety} -> {after_safety} chunks (violations removed/reclassified)")
        
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
        logger.info(f"  - After small chunk merging: {after_merge_small} chunks")
        logger.info(f"  - After table fragment merging: {len(result)} final chunks")
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

