"""
Evaluate extracted text quality by comparing with original PDF
Extracts sample pages from original and compares with extracted text
"""

import os
import sys
import re

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def extract_original_page(pdf_path, page_num):
    """Extract text from original PDF page for comparison"""
    import fitz  # PyMuPDF
    
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return None
    
    page = doc[page_num]
    # Get raw text without RTL fix for comparison
    text = page.get_text("text")
    doc.close()
    
    return text


def extract_original_page_blocks(pdf_path, page_num):
    """Extract using blocks method (same as our extraction)"""
    import fitz  # PyMuPDF
    
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return None
    
    page = doc[page_num]
    blocks = page.get_text("blocks", sort=True)
    
    text_blocks = []
    for block in blocks:
        if block[6] == 0:  # Text block
            block_text = block[4].strip()
            if block_text and len(block_text) > 2:
                text_blocks.append(block_text)
    
    doc.close()
    return '\n'.join(text_blocks) if text_blocks else None


def load_extracted_text(extracted_file):
    """Load extracted text and parse by pages"""
    if not os.path.exists(extracted_file):
        return None
    
    with open(extracted_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse pages from extracted text
    pages = {}
    page_pattern = r'--- Page (\d+) ---\n\n(.*?)(?=\n\n--- Page \d+ ---|\Z)'
    matches = re.finditer(page_pattern, content, re.DOTALL)
    
    for match in matches:
        page_num = int(match.group(1))
        page_text = match.group(2).strip()
        pages[page_num] = page_text
    
    return pages


def compare_texts(original, extracted):
    """Compare original and extracted texts"""
    if not original or not extracted:
        return {
            'similarity': 0,
            'original_length': len(original) if original else 0,
            'extracted_length': len(extracted) if extracted else 0,
            'length_diff': 0,
            'issues': []
        }
    
    # Basic metrics
    orig_len = len(original)
    extr_len = len(extracted)
    length_diff = abs(orig_len - extr_len)
    length_diff_pct = (length_diff / orig_len * 100) if orig_len > 0 else 0
    
    # Character analysis
    orig_arabic = len(re.findall(r'[\u0600-\u06FF]', original))
    extr_arabic = len(re.findall(r'[\u0600-\u06FF]', extracted))
    
    orig_english = len(re.findall(r'\b[a-zA-Z]{3,}\b', original))
    extr_english = len(re.findall(r'\b[a-zA-Z]{3,}\b', extracted))
    
    # Check for common issues
    issues = []
    
    if length_diff_pct > 50:
        issues.append(f"Significant length difference: {length_diff_pct:.1f}%")
    
    if orig_arabic > 0 and extr_arabic == 0:
        issues.append("Arabic text missing in extraction")
    elif orig_arabic > 0 and extr_arabic < orig_arabic * 0.5:
        issues.append(f"Arabic text significantly reduced: {orig_arabic} -> {extr_arabic}")
    
    if orig_english > 0 and extr_english == 0:
        issues.append("English text missing in extraction")
    elif orig_english > 0 and extr_english < orig_english * 0.5:
        issues.append(f"English text significantly reduced: {orig_english} -> {extr_english}")
    
    # Simple similarity (word overlap)
    orig_words = set(re.findall(r'\b\w+\b', original.lower()))
    extr_words = set(re.findall(r'\b\w+\b', extracted.lower()))
    
    if orig_words:
        overlap = len(orig_words & extr_words)
        similarity = (overlap / len(orig_words)) * 100
    else:
        similarity = 0
    
    return {
        'similarity': similarity,
        'original_length': orig_len,
        'extracted_length': extr_len,
        'length_diff': length_diff,
        'length_diff_pct': length_diff_pct,
        'original_arabic': orig_arabic,
        'extracted_arabic': extr_arabic,
        'original_english': orig_english,
        'extracted_english': extr_english,
        'issues': issues
    }


def evaluate_extraction(pdf_path, extracted_file, sample_pages=[1, 3, 5, 10, 20]):
    """Evaluate extraction quality"""
    
    print("=" * 80)
    print("EXTRACTION QUALITY EVALUATION")
    print("=" * 80)
    
    # Load extracted text
    print("\n[1] Loading extracted text...")
    extracted_pages = load_extracted_text(extracted_file)
    
    if not extracted_pages:
        print(f"   [X] Could not load extracted text from: {extracted_file}")
        return
    
    print(f"   [OK] Loaded {len(extracted_pages)} pages from extracted file")
    
    # Evaluate sample pages
    print("\n[2] Evaluating sample pages...")
    print("   Comparing original PDF with extracted text\n")
    
    results = []
    
    for page_num in sample_pages:
        # Extract from original (0-indexed)
        original_raw = extract_original_page(pdf_path, page_num - 1)
        original_blocks = extract_original_page_blocks(pdf_path, page_num - 1)
        extracted = extracted_pages.get(page_num)
        
        if not extracted:
            print(f"   Page {page_num}: [SKIP] Not found in extracted text")
            continue
        
        # Compare with blocks method (what we used)
        comparison = compare_texts(original_blocks, extracted)
        
        results.append({
            'page': page_num,
            'comparison': comparison
        })
        
        # Print results
        print(f"   Page {page_num}:")
        print(f"      Original length: {comparison['original_length']:,} chars")
        print(f"      Extracted length: {comparison['extracted_length']:,} chars")
        print(f"      Length difference: {comparison['length_diff_pct']:.1f}%")
        print(f"      Similarity: {comparison['similarity']:.1f}%")
        print(f"      Arabic chars - Original: {comparison['original_arabic']}, Extracted: {comparison['extracted_arabic']}")
        print(f"      English words - Original: {comparison['original_english']}, Extracted: {comparison['extracted_english']}")
        
        if comparison['issues']:
            print(f"      [!] Issues: {', '.join(comparison['issues'])}")
        
        # Show sample comparison
        if page_num <= 5:  # Show detailed comparison for first few pages
            print(f"\n      --- Sample Comparison (Page {page_num}) ---")
            print("      Original (first 200 chars):")
            orig_preview = (original_blocks or original_raw or "")[:200]
            print(f"      {orig_preview}")
            print("\n      Extracted (first 200 chars):")
            extr_preview = extracted[:200]
            print(f"      {extr_preview}")
        
        print()
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    
    if results:
        avg_similarity = sum(r['comparison']['similarity'] for r in results) / len(results)
        avg_length_diff = sum(r['comparison']['length_diff_pct'] for r in results) / len(results)
        
        total_orig_arabic = sum(r['comparison']['original_arabic'] for r in results)
        total_extr_arabic = sum(r['comparison']['extracted_arabic'] for r in results)
        total_orig_english = sum(r['comparison']['original_english'] for r in results)
        total_extr_english = sum(r['comparison']['extracted_english'] for r in results)
        
        print(f"\nAverage similarity: {avg_similarity:.1f}%")
        print(f"Average length difference: {avg_length_diff:.1f}%")
        print(f"\nArabic characters:")
        print(f"   Original total: {total_orig_arabic:,}")
        print(f"   Extracted total: {total_extr_arabic:,}")
        print(f"   Retention: {(total_extr_arabic/total_orig_arabic*100) if total_orig_arabic > 0 else 0:.1f}%")
        print(f"\nEnglish words:")
        print(f"   Original total: {total_orig_english:,}")
        print(f"   Extracted total: {total_extr_english:,}")
        print(f"   Retention: {(total_extr_english/total_orig_english*100) if total_orig_english > 0 else 0:.1f}%")
        
        # Quality assessment
        print("\n" + "=" * 80)
        print("QUALITY ASSESSMENT:")
        print("=" * 80)
        
        if avg_similarity >= 80:
            quality = "EXCELLENT"
        elif avg_similarity >= 60:
            quality = "GOOD"
        elif avg_similarity >= 40:
            quality = "FAIR"
        else:
            quality = "POOR"
        
        print(f"\nOverall Quality: {quality}")
        
        if avg_similarity >= 70:
            print("✓ Extraction quality is suitable for ChunkWise processing")
        elif avg_similarity >= 50:
            print("⚠ Extraction quality is acceptable but may need review")
        else:
            print("✗ Extraction quality is poor - consider manual review or alternative methods")
        
        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        
        if total_extr_arabic < total_orig_arabic * 0.7:
            print("⚠ Arabic text retention is low - check RTL processing")
        
        if total_extr_english < total_orig_english * 0.7:
            print("⚠ English text retention is low - check extraction method")
        
        if avg_length_diff > 30:
            print("⚠ Significant length differences - may indicate missing content")
        
        if avg_similarity >= 70:
            print("✓ Text is ready for ChunkWise chunking")
            print("✓ Consider using language='auto' in ChunkWise")
            print("✓ Test with different chunk sizes (512, 1024)")
    
    print("\n" + "=" * 80)


def main():
    """Main evaluation function"""
    pdf_path = "publicationpdfar1765273617.pdf"
    extracted_file = "extracted_text_final_best.txt"
    
    if not os.path.exists(pdf_path):
        print(f"[X] PDF file not found: {pdf_path}")
        return
    
    if not os.path.exists(extracted_file):
        print(f"[X] Extracted text file not found: {extracted_file}")
        print("   Please run final_best_extraction.py first")
        return
    
    # Evaluate with sample pages
    sample_pages = [1, 3, 5, 10, 20, 50, 100]  # Sample different pages
    
    evaluate_extraction(pdf_path, extracted_file, sample_pages)


if __name__ == "__main__":
    main()

