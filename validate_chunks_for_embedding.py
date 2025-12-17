"""
Comprehensive validation script for Phase 3 chunks before Phase 4 embeddings.

This script validates chunks against all quality criteria to ensure:
- No fake bilingual chunks
- Complete metadata
- No broken Arabic text
- Unique chunk IDs
- No TOC leakage
- Embedding control flags present
- No low-information chunks (REFINEMENT 3)

Usage:
    python validate_chunks_for_embedding.py phase3_output.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any


def detect_language_robust(text: str) -> str:
    """Detect language using character ratio."""
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


def validate_pipeline_output(chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Comprehensive validation report.
    
    Includes REFINEMENT 3: low-information chunk detection.
    
    Returns dict of issue_type -> list of chunk_ids
    """
    issues = {
        "fake_bilingual": [],
        "missing_metadata": [],
        "broken_arabic": [],
        "duplicate_ids": [],
        "toc_leakage": [],
        "missing_embedding_flag": [],
        "low_information": [],  # NEW: REFINEMENT 3
    }
    
    seen_ids = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        content = chunk.get("content", {})
        metadata = chunk.get("metadata", {})
        
        ar = content.get("ar") or ""
        en = content.get("en") or ""
        
        # Check 1: Fake bilingual (exact duplicate)
        if ar and en and ar == en:
            issues["fake_bilingual"].append(chunk_id)
        
        # Check 2: Fake bilingual (same language in both fields)
        if ar and en:
            ar_lang = detect_language_robust(ar)
            en_lang = detect_language_robust(en)
            if ar_lang == en_lang and ar_lang != "mixed":
                issues["fake_bilingual"].append(chunk_id)
        
        # Check 3: Missing metadata
        if not metadata.get("chapter") and not metadata.get("section"):
            issues["missing_metadata"].append(chunk_id)
        
        # Check 4: Duplicate IDs
        if chunk_id in seen_ids:
            issues["duplicate_ids"].append(chunk_id)
        seen_ids.add(chunk_id)
        
        # Check 5: Broken Arabic (ligature artifacts)
        if 'اجل' in ar or 'احل' in ar or 'اال' in ar:
            issues["broken_arabic"].append(chunk_id)
        
        # Check 6: TOC leakage
        if metadata.get("chunk_type") == "toc":
            issues["toc_leakage"].append(chunk_id)
        
        # Check 7: Missing embedding flag
        if "embedding_allowed" not in metadata:
            issues["missing_embedding_flag"].append(chunk_id)
        
        # Check 8: REFINEMENT 3 - Low information chunks (orphan fragments, stubs, numeric-only)
        # Only check chunks marked for embedding
        if metadata.get("embedding_allowed", False):
            combined_text = ar + " " + en
            word_count = len(combined_text.split())
            
            # Reject chunks with < 30 words (headings, stubs, orphans)
            if word_count < 30:
                issues["low_information"].append(chunk_id)
    
    return issues


def print_validation_report(issues: Dict[str, List[str]]) -> bool:
    """
    Print validation report and return True if all checks passed.
    """
    print("\n" + "="*80)
    print("PHASE 3 CHUNK VALIDATION REPORT")
    print("="*80 + "\n")
    
    all_passed = True
    
    for issue_type, chunk_ids in issues.items():
        if chunk_ids:
            all_passed = False
            print(f"❌ {issue_type.upper()}: {len(chunk_ids)} chunks affected")
            # Show first 5 examples
            for chunk_id in chunk_ids[:5]:
                print(f"   - {chunk_id}")
            if len(chunk_ids) > 5:
                print(f"   ... and {len(chunk_ids) - 5} more")
            print()
        else:
            print(f"✅ {issue_type.upper()}: 0 issues")
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED - READY FOR PHASE 4")
    else:
        print("❌ VALIDATION FAILED - FIX ISSUES BEFORE PHASE 4")
    print("="*80 + "\n")
    
    return all_passed


def validate_file(json_path: str) -> bool:
    """Validate chunks from JSON file."""
    if not Path(json_path).exists():
        print(f"❌ Error: File not found: {json_path}")
        return False
    
    print(f"Loading chunks from: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        chunks = data.get("chunks", [])
        print(f"Found {len(chunks)} chunks in file")
    elif isinstance(data, list):
        chunks = data
        print(f"Found {len(chunks)} chunks in file")
    else:
        print(f"❌ Error: Unexpected JSON structure")
        return False
    
    if not chunks:
        print("❌ Error: No chunks found in file")
        return False
    
    # Run validation
    issues = validate_pipeline_output(chunks)
    
    # Print report
    return print_validation_report(issues)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_chunks_for_embedding.py <phase3_output.json>")
        print("\nExample:")
        print("  python validate_chunks_for_embedding.py phase3_output.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    passed = validate_file(json_path)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

