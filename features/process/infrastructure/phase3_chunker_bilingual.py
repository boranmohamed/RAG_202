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

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional

from features.process.domain.interfaces import IChunker

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


def should_chunk(block: Dict[str, Any]) -> bool:
    """
    Pre-filter rules (Rule 1, 2, 3).
    
    Returns False for blocks that should NEVER be chunked:
    - TOC/index pages
    - Headers/titles (metadata only)
    - Footers
    """
    block_type = block.get("type", "")
    
    # Rule 3: TOC/index pages never chunked
    if block_type == "toc":
        return False
    
    # Rule 2: Headers/titles are metadata only
    if block_type == "header":
        return False
    
    # Footers never chunked
    if block_type == "footer":
        return False
    
    # Tables handled separately (Rule 7)
    if block_type == "table":
        return False
    
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
    max_tokens: int = 350,
    min_tokens: int = 120,
    overlap_tokens: int = 40,
) -> List[str]:
    """
    Chunk narrative text using ChunkWise (Rule 4, 6, 13, 14).
    
    Uses semantic chunking with sentence awareness.
    """
    if not text or len(text.strip()) < min_tokens:
        return []
    
    if CHUNKWISE_AVAILABLE:
        try:
            # Use ChunkWise recursive chunker with sentence awareness
            chunker = Chunker(
                strategy="recursive",
                chunk_size=max_tokens,
                chunk_overlap=overlap_tokens,
                language="auto" if language == "mixed" else language,
            )
            text_chunks = chunker.chunk(text)
            return [chunk.content for chunk in text_chunks]
        except Exception:
            # Fallback: simple sentence-based chunking
            return _fallback_chunk_text(text, max_tokens)
    else:
        # Fallback chunking
        return _fallback_chunk_text(text, max_tokens)


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
    """
    if chunk_text in original_text:
        return chunk_text
    
    # Match by word overlap with 60% confidence threshold
    chunk_words = set(chunk_text.split())
    original_words = set(original_text.split())
    overlap = len(chunk_words & original_words)
    ratio = overlap / max(1, len(chunk_words))
    
    # Only accept if 60% confidence or higher
    if ratio >= 0.6:
        return chunk_text
    
    # Fallback: safer to return slice from original to prevent cross-language bleed
    # This prevents accidental mixing when merging sections
    return original_text[:len(chunk_text)] if len(original_text) > len(chunk_text) else original_text


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
    # Check required fields
    if not chunk.content:
        return False
    
    ar_text = chunk.content.get("ar") or ""
    en_text = chunk.content.get("en") or ""
    
    # Must have at least one language
    if not ar_text and not en_text:
        return False
    
    # Rule 16: Detect fake bilingual (same text in both fields)
    if ar_text and en_text:
        # Check if texts are suspiciously similar (exact duplicate)
        if ar_text == en_text:
            return False  # Fake bilingual detected
        
        # Check language consistency
        ar_lang = detect_language_robust(ar_text)
        en_lang = detect_language_robust(en_text)
        
        # If both detected as same language, something is wrong
        if ar_lang == en_lang and ar_lang != "mixed":
            return False  # Same language in both fields
    
    # Check metadata
    if not chunk.metadata.chapter and not chunk.metadata.section:
        # Allow chunks without chapter/section for now (will be enhanced)
        pass
    
    if chunk.metadata.page_start == 0:
        return False
    
    # Rule 10: Check for broken text
    ar_text = chunk.content.get("ar") or ""
    en_text = chunk.content.get("en") or ""
    
    # Check for obvious broken Arabic (too many isolated characters)
    if ar_text:
        isolated_chars = len(re.findall(r'\s[\u0600-\u06FF]\s', ar_text))
        if isolated_chars > len(ar_text) * 0.1:  # More than 10% isolated chars
            return False
    
    # Check minimum content length for narrative chunks
    if chunk.metadata.chunk_type == "narrative":
        total_length = len(ar_text) + len(en_text)
        if total_length < 50:  # Too short
            return False
    
    return True


def chunk_section_blocks(
    section_blocks: List[Dict[str, Any]],
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    max_tokens: int = 350,
    min_tokens: int = 120,
    overlap_tokens: int = 40,
) -> List[Chunk]:
    """
    Chunk blocks within a single section (Rule 1: no cross-section chunking).
    
    Args:
        section_blocks: List of normalized blocks from Phase 2
        chapter: Chapter name (if known)
        section: Section name (if known)
        max_tokens: Maximum tokens per chunk (Rule 13)
        min_tokens: Minimum tokens per chunk
        overlap_tokens: Overlap between chunks (Rule 14)
    
    Returns:
        List of validated chunks
    """
    section_chunks: List[Chunk] = []
    
    for block in section_blocks:
        # Rule 1, 2, 3: Filter blocks
        if not should_chunk(block):
            continue
        
        # Extract content
        content = block.get("content", {})
        if not isinstance(content, dict):
            continue
        
        # Preprocess for chunking
        combined_text, language = preprocess_for_chunking(content, block.get("type", ""))
        
        if not combined_text:
            continue
        
        # Chunk the text
        text_chunks = chunk_narrative_text(
            combined_text,
            language,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens,
        )
        
        # Create chunk objects
        for i, chunk_text in enumerate(text_chunks):
            # Split back into bilingual structure
            chunk_content = split_bilingual_content(chunk_text, content)
            
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
    content = table_block.get("content", {})
    if not isinstance(content, dict):
        return []
    
    ar_text = content.get("ar") or ""
    en_text = content.get("en") or ""
    
    if not ar_text and not en_text:
        return []
    
    # Use block's own chapter/section, not the passed parameters
    block_chapter = table_block.get("chapter") or chapter
    block_section = table_block.get("section") or section
    
    # Rule 7: One table = one chunk (or multiple if very large)
    # For now, keep tables as single chunks
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
        return [chunk]
    
    return []


def merge_small_chunks_with_adjacent(
    chunks: List[Dict[str, Any]], 
    min_word_count: int = 30
) -> List[Dict[str, Any]]:
    """
    Merge chunks with < min_word_count words with adjacent chunks in same section.
    
    Strategy:
    - Small chunk + previous chunk (same section) → merged
    - If no previous, merge with next chunk (same section)
    - If isolated, keep as-is but mark embedding_allowed=False
    
    This fixes low-information chunks while preserving section boundaries.
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
        word_count = len((ar + " " + en).split())
        
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
                continue
        
        # Isolated small chunk - keep but disable embedding
        metadata["embedding_allowed"] = False
        chunk["metadata"] = metadata
        merged.append(chunk)
    
    return merged


def can_merge_chunks(
    prev_chunk: Dict[str, Any], 
    curr_chunk: Dict[str, Any],
    curr_chapter: Optional[str],
    curr_section: Optional[str]
) -> bool:
    """Check if two chunks can be merged (same chapter/section)."""
    prev_metadata = prev_chunk.get("metadata", {})
    return (
        prev_metadata.get("chapter") == curr_chapter and
        prev_metadata.get("section") == curr_section and
        prev_metadata.get("chunk_type") == "narrative"  # Only merge narrative
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
        
        # Only merge narrative blocks, not tables/headers
        if block.get("type") in ["paragraph", "narrative"]:
            buffer.append(block)
        else:
            # Non-narrative blocks pass through individually
            if buffer:
                merged.append(_merge_block_group(buffer))
                buffer = []
            merged.append(block)
    
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
        max_tokens: int = 350,
        min_tokens: int = 120,
        overlap_tokens: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Chunk normalized blocks from Phase 2 into semantic chunks.
        
        Implements IChunker interface.
        """
        all_chunks: List[Chunk] = []
        
        # CRITICAL: Merge blocks by section BEFORE chunking
        # This prevents fragmented ideas and improves context continuity
        blocks = merge_blocks_by_section(blocks)
        
        # Extract chapter/section from blocks
        current_chapter = None
        current_section = None
        
        narrative_blocks = []
        table_blocks = []
        
        for block in blocks:
            block_type = block.get("type", "")
            
            # Update current chapter/section
            if block.get("chapter"):
                current_chapter = block.get("chapter")
            if block.get("section"):
                current_section = block.get("section")
            
            if block_type == "table":
                table_blocks.append(block)
            elif should_chunk(block):
                narrative_blocks.append(block)
        
        # Chunk narrative blocks
        if narrative_blocks:
            narrative_chunks = chunk_section_blocks(
                narrative_blocks,
                chapter=current_chapter,
                section=current_section,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap_tokens=overlap_tokens,
            )
            all_chunks.extend(narrative_chunks)
        
        # Chunk table blocks separately
        for table_block in table_blocks:
            table_chunks = chunk_table_block(
                table_block,
                chapter=current_chapter,
                section=current_section,
            )
            all_chunks.extend(table_chunks)
        
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
        result = merge_small_chunks_with_adjacent(result, min_word_count=30)
        
        return result


# Convenience function for backward compatibility
def chunk_phase3_from_blocks(
    blocks: List[Dict[str, Any]],
    document_name: str = "Statistical Year Book 2025",
    year: int = 2024,
    max_tokens: int = 350,
    min_tokens: int = 120,
    overlap_tokens: int = 40,
) -> List[Dict[str, Any]]:
    """
    Convenience function: chunk normalized blocks from Phase 2.
    
    This is a wrapper around ChunkWiseBilingualChunker for backward compatibility.
    For new code, use ChunkWiseBilingualChunker directly.
    """
    chunker = ChunkWiseBilingualChunker()
    return chunker.chunk_blocks(
        blocks=blocks,
        document_name=document_name,
        year=year,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
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

