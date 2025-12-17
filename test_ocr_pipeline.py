"""
Test script for OCR Pipeline (pdfplumber + Tesseract).

This script tests the complete OCR pipeline:
1. Phase 1 OCR: Extract with pdfplumber + Tesseract
2. Phase 2 OCR: Normalize with OCR-specific fixes
3. Phase 3: Chunk with ChunkWise rules

Tests with the provided PDF: publicationpdfar1765273617.pdf
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.process.infrastructure.phase1_ocr_extractor_pdfplumber import (
    extract_structured_pdf_ocr,
    extract_blocks_ocr,
)
from features.process.infrastructure.phase2_ocr_preprocessor import (
    normalize_ocr_blocks,
)
from features.process.infrastructure.phase3_chunker_bilingual import (
    ChunkWiseBilingualChunker,
)


def test_phase1_ocr_extraction():
    """Test Phase 1 OCR extraction."""
    print("=" * 80)
    print("TEST 1: Phase 1 OCR Extraction")
    print("=" * 80)
    
    pdf_path = "publicationpdfar1765273617.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return None
    
    print(f"📄 Processing PDF: {pdf_path}")
    print(f"⚙️  Settings: DPI=300, Max Pages=5 (for testing)")
    print()
    
    try:
        # Extract first 5 pages for testing
        pages = extract_structured_pdf_ocr(
            pdf_path,
            max_pages=5,
            ocr_dpi=300
        )
        
        print(f"✅ Extracted {len(pages)} pages")
        print()
        
        # Analyze extraction
        total_blocks = sum(len(page["blocks"]) for page in pages)
        table_blocks = sum(
            1 for page in pages
            for block in page["blocks"]
            if block["type"] == "table"
        )
        text_blocks = total_blocks - table_blocks
        
        print(f"📊 Statistics:")
        print(f"   - Total blocks: {total_blocks}")
        print(f"   - Text blocks: {text_blocks}")
        print(f"   - Table blocks: {table_blocks}")
        print()
        
        # Sample first page
        if pages:
            page1 = pages[0]
            print(f"📄 Sample - Page {page1['pageNumber']}:")
            print(f"   Blocks: {len(page1['blocks'])}")
            
            for i, block in enumerate(page1["blocks"][:3]):  # First 3 blocks
                content = block.get("content", {})
                ar = content.get("ar", "")
                en = content.get("en", "")
                
                print(f"\n   Block {i+1} [{block['type']}]:")
                if ar:
                    ar_preview = ar[:100] + "..." if len(ar) > 100 else ar
                    print(f"      AR: {ar_preview}")
                if en:
                    en_preview = en[:100] + "..." if len(en) > 100 else en
                    print(f"      EN: {en_preview}")
        
        # Save output
        output_file = "test_ocr_phase1_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved output to: {output_file}")
        
        return pages
        
    except Exception as e:
        print(f"❌ Phase 1 extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_phase2_ocr_preprocessing(pages):
    """Test Phase 2 OCR preprocessing."""
    print("\n")
    print("=" * 80)
    print("TEST 2: Phase 2 OCR Preprocessing")
    print("=" * 80)
    
    if not pages:
        print("⏭️  Skipping (no pages from Phase 1)")
        return None
    
    print(f"📝 Normalizing {len(pages)} pages...")
    print()
    
    try:
        normalized = normalize_ocr_blocks(pages)
        
        print(f"✅ Normalized {len(normalized)} blocks")
        print()
        
        # Analyze normalization
        by_type = {}
        by_lang = {"ar": 0, "en": 0, "both": 0}
        
        for block in normalized:
            block_type = block.get("type", "unknown")
            by_type[block_type] = by_type.get(block_type, 0) + 1
            
            content = block.get("content", {})
            has_ar = bool(content.get("ar"))
            has_en = bool(content.get("en"))
            
            if has_ar and has_en:
                by_lang["both"] += 1
            elif has_ar:
                by_lang["ar"] += 1
            elif has_en:
                by_lang["en"] += 1
        
        print(f"📊 Statistics:")
        print(f"   By Type:")
        for btype, count in sorted(by_type.items()):
            print(f"      - {btype}: {count}")
        print(f"   By Language:")
        print(f"      - Arabic only: {by_lang['ar']}")
        print(f"      - English only: {by_lang['en']}")
        print(f"      - Both: {by_lang['both']}")
        print()
        
        # Sample normalized blocks
        print("📄 Sample normalized blocks:")
        for i, block in enumerate(normalized[:3]):
            content = block.get("content", {})
            ar = content.get("ar", "")
            en = content.get("en", "")
            
            print(f"\n   Block {i+1} [Page {block.get('pageNumber')}, {block.get('type')}]:")
            if ar:
                ar_preview = ar[:100] + "..." if len(ar) > 100 else ar
                print(f"      AR: {ar_preview}")
            if en:
                en_preview = en[:100] + "..." if len(en) > 100 else en
                print(f"      EN: {en_preview}")
        
        # Save output
        output_file = "test_ocr_phase2_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved output to: {output_file}")
        
        return normalized
        
    except Exception as e:
        print(f"❌ Phase 2 preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_phase3_chunking(normalized_blocks):
    """Test Phase 3 chunking."""
    print("\n")
    print("=" * 80)
    print("TEST 3: Phase 3 Chunking")
    print("=" * 80)
    
    if not normalized_blocks:
        print("⏭️  Skipping (no blocks from Phase 2)")
        return None
    
    print(f"✂️  Chunking {len(normalized_blocks)} blocks...")
    print(f"⚙️  Settings: max_tokens=350, min_tokens=120, overlap=40")
    print()
    
    try:
        chunker = ChunkWiseBilingualChunker()
        
        chunks = chunker.chunk_blocks(
            blocks=normalized_blocks,
            document_name="Statistical Year Book 2025 (OCR Test)",
            year=2024,
            max_tokens=350,
            min_tokens=120,
            overlap_tokens=40,
        )
        
        print(f"✅ Created {len(chunks)} chunks")
        print()
        
        # Analyze chunks
        by_type = {}
        by_lang = []
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            chunk_type = metadata.get("chunk_type", "unknown")
            by_type[chunk_type] = by_type.get(chunk_type, 0) + 1
            
            languages = metadata.get("language", [])
            by_lang.append(tuple(sorted(languages)))
        
        from collections import Counter
        lang_counts = Counter(by_lang)
        
        print(f"📊 Statistics:")
        print(f"   By Type:")
        for ctype, count in sorted(by_type.items()):
            print(f"      - {ctype}: {count}")
        print(f"   By Language:")
        for langs, count in sorted(lang_counts.items()):
            lang_str = "+".join(langs) if langs else "none"
            print(f"      - {lang_str}: {count}")
        print()
        
        # Sample chunks
        print("📄 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            content = chunk.get("content", {})
            metadata = chunk.get("metadata", {})
            
            ar = content.get("ar", "")
            en = content.get("en", "")
            
            print(f"\n   Chunk {i+1} [{metadata.get('chunk_type')}]:")
            print(f"      ID: {chunk.get('chunk_id')}")
            print(f"      Pages: {metadata.get('page_start')}-{metadata.get('page_end')}")
            print(f"      Chapter: {metadata.get('chapter', 'N/A')}")
            print(f"      Section: {metadata.get('section', 'N/A')}")
            print(f"      Languages: {', '.join(metadata.get('language', []))}")
            
            if ar:
                ar_preview = ar[:100] + "..." if len(ar) > 100 else ar
                print(f"      AR: {ar_preview}")
            if en:
                en_preview = en[:100] + "..." if len(en) > 100 else en
                print(f"      EN: {en_preview}")
        
        # Save output
        output_file = "test_ocr_phase3_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved output to: {output_file}")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Phase 3 chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_complete_pipeline():
    """Test complete OCR pipeline end-to-end."""
    print("\n")
    print("=" * 80)
    print("COMPLETE PIPELINE TEST")
    print("=" * 80)
    print()
    
    # Phase 1
    pages = test_phase1_ocr_extraction()
    if not pages:
        print("\n❌ Pipeline stopped at Phase 1")
        return
    
    # Phase 2
    normalized = test_phase2_ocr_preprocessing(pages)
    if not normalized:
        print("\n❌ Pipeline stopped at Phase 2")
        return
    
    # Phase 3
    chunks = test_phase3_chunking(normalized)
    if not chunks:
        print("\n❌ Pipeline stopped at Phase 3")
        return
    
    print("\n")
    print("=" * 80)
    print("✅ COMPLETE PIPELINE SUCCESS")
    print("=" * 80)
    print()
    print(f"📊 Final Results:")
    print(f"   - Pages processed: {len(pages)}")
    print(f"   - Blocks normalized: {len(normalized)}")
    print(f"   - Chunks created: {len(chunks)}")
    print()
    print("📁 Output files:")
    print("   - test_ocr_phase1_output.json")
    print("   - test_ocr_phase2_output.json")
    print("   - test_ocr_phase3_output.json")
    print()


def main():
    """Main test runner."""
    print("\n")
    print("🔬 OCR Pipeline Test Suite")
    print("=" * 80)
    print()
    print("Testing: pdfplumber + Tesseract OCR pipeline")
    print("PDF: publicationpdfar1765273617.pdf")
    print()
    
    # Check if Tesseract is available
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        langs = pytesseract.get_languages()
        print(f"✅ Available languages: {', '.join(langs)}")
        
        if 'ara' not in langs:
            print("⚠️  WARNING: Arabic language pack not found!")
        if 'eng' not in langs:
            print("⚠️  WARNING: English language pack not found!")
    except Exception as e:
        print(f"❌ Tesseract not available: {e}")
        print("Please install Tesseract and language packs (see docs/OCR_SETUP.md)")
        return
    
    print()
    
    # Run complete pipeline test
    test_complete_pipeline()


if __name__ == "__main__":
    main()

