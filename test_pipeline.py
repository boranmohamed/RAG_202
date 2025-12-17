"""
Test script for Phase 1 → Phase 2 → Phase 3 pipeline.

This script tests the complete flow:
1. Phase 1+2: Extract and normalize PDF
2. Phase 3: Chunk normalized blocks

Usage:
    python test_pipeline.py
"""

import json
import sys
from pathlib import Path

import requests

# Configuration
API_BASE_URL = "http://localhost:8009"
PDF_PATH = "publicationpdfar1765273617.pdf"

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_step(step_num: int, description: str):
    """Print a formatted step header."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}STEP {step_num}: {description}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}[OK] {message}{RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{RED}[ERROR] {message}{RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{YELLOW}[INFO] {message}{RESET}")


def check_server():
    """Check if API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("API server is running")
            return True
        else:
            print_error(f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API server. Is it running?")
        print_info("Start server with: uvicorn main:app --reload")
        return False
    except Exception as e:
        print_error(f"Error checking server: {e}")
        return False


def test_phase12_extraction():
    """Test Phase 1+2: PDF extraction and normalization."""
    print_step(1, "Phase 1+2: Extract and Normalize PDF")
    
    # Check if PDF exists
    if not Path(PDF_PATH).exists():
        print_error(f"PDF file not found: {PDF_PATH}")
        return None
    
    print_info(f"Processing PDF: {PDF_PATH}")
    
    # Call API
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/pdfs/extract",
            json={"pdf_path": PDF_PATH},
            timeout=300,  # 5 minutes timeout for large PDFs
        )
        
        if response.status_code != 200:
            print_error(f"API returned status {response.status_code}")
            print_error(f"Error: {response.text}")
            return None
        
        data = response.json()
        
        # Verify response structure
        assert "pdf_path" in data, "Missing pdf_path in response"
        assert "pages" in data, "Missing pages in response"
        assert "blocks" in data, "Missing blocks in response"
        
        print_success(f"Extracted {data['pages']} pages")
        print_success(f"Created {len(data['blocks'])} normalized blocks")
        
        # Show sample blocks
        if data['blocks']:
            print_info("\nSample blocks:")
            for i, block in enumerate(data['blocks'][:3], 1):
                print(f"  Block {i}:")
                print(f"    Page: {block['pageNumber']}")
                print(f"    Type: {block['type']}")
                ar_len = len(block['content'].get('ar', '') or '')
                en_len = len(block['content'].get('en', '') or '')
                print(f"    Arabic: {ar_len} chars, English: {en_len} chars")
        
        # Save response for Phase 3
        output_file = "phase12_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print_success(f"Saved Phase 1+2 output to: {output_file}")
        
        return data
        
    except requests.exceptions.Timeout:
        print_error("Request timed out. PDF might be too large.")
        return None
    except Exception as e:
        print_error(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_phase3_chunking(phase12_data):
    """Test Phase 3: Chunk normalized blocks."""
    print_step(2, "Phase 3: Chunk Normalized Blocks")
    
    if not phase12_data:
        print_error("No Phase 1+2 data available. Skipping Phase 3.")
        return None
    
    # Prepare blocks for chunking
    blocks = phase12_data['blocks']
    print_info(f"Chunking {len(blocks)} normalized blocks")
    
    # Call API
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/pdfs/chunk",
            json={
                "blocks": [
                    {
                        "pageNumber": b["pageNumber"],
                        "type": b["type"],
                        "content": {
                            "ar": b["content"].get("ar"),
                            "en": b["content"].get("en"),
                        },
                    }
                    for b in blocks
                ],
                "document_name": "Statistical Year Book 2025",
                "year": 2024,
                "max_tokens": 350,
                "min_tokens": 120,
                "overlap_tokens": 40,
            },
            timeout=300,  # 5 minutes timeout
        )
        
        if response.status_code != 200:
            print_error(f"API returned status {response.status_code}")
            print_error(f"Error: {response.text}")
            return None
        
        data = response.json()
        
        # Verify response structure
        assert "chunks" in data, "Missing chunks in response"
        assert "total_chunks" in data, "Missing total_chunks in response"
        assert "narrative_chunks" in data, "Missing narrative_chunks in response"
        assert "table_chunks" in data, "Missing table_chunks in response"
        
        print_success(f"Created {data['total_chunks']} chunks")
        print_success(f"  - Narrative chunks: {data['narrative_chunks']}")
        print_success(f"  - Table chunks: {data['table_chunks']}")
        
        # Show sample chunks
        if data['chunks']:
            print_info("\nSample chunks:")
            for i, chunk in enumerate(data['chunks'][:3], 1):
                print(f"  Chunk {i}:")
                print(f"    ID: {chunk['chunk_id']}")
                print(f"    Type: {chunk['metadata']['chunk_type']}")
                print(f"    Page: {chunk['metadata']['page_start']}")
                print(f"    Language: {chunk['metadata']['language']}")
                ar_len = len(chunk['content'].get('ar', '') or '')
                en_len = len(chunk['content'].get('en', '') or '')
                print(f"    Arabic: {ar_len} chars, English: {en_len} chars")
        
        # Save response
        output_file = "phase3_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print_success(f"Saved Phase 3 output to: {output_file}")
        
        return data
        
    except requests.exceptions.Timeout:
        print_error("Request timed out. Too many blocks to chunk.")
        return None
    except Exception as e:
        print_error(f"Error during chunking: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_chunks(chunks_data):
    """Verify chunk quality and rules compliance."""
    print_step(3, "Verify Chunk Quality")
    
    if not chunks_data or not chunks_data.get('chunks'):
        print_error("No chunks to verify")
        return False
    
    chunks = chunks_data['chunks']
    
    # Import validation function
    try:
        from validate_chunks_for_embedding import validate_pipeline_output, detect_language_robust
    except ImportError:
        print_error("Could not import validation script")
        return False
    
    # Run comprehensive validation
    print_info("Running comprehensive validation...")
    issues = validate_pipeline_output(chunks)
    
    # Report each issue type
    all_passed = True
    
    for issue_type, chunk_ids in issues.items():
        if chunk_ids:
            all_passed = False
            print_error(f"{issue_type}: {len(chunk_ids)} chunks affected")
            # Show first 3 examples
            for chunk_id in chunk_ids[:3]:
                print(f"    - {chunk_id}")
            if len(chunk_ids) > 3:
                print(f"    ... and {len(chunk_ids) - 3} more")
        else:
            print_success(f"{issue_type}: 0 issues")
    
    # Additional stats
    print_info("\nChunk Statistics:")
    
    # Bilingual content
    bilingual_count = sum(1 for c in chunks if c.get('content', {}).get('ar') and c.get('content', {}).get('en'))
    print(f"  Bilingual chunks: {bilingual_count}/{len(chunks)}")
    
    # Chunk sizes
    sizes = []
    for chunk in chunks:
        content = chunk.get('content', {})
        total_len = len(content.get('ar', '') or '') + len(content.get('en', '') or '')
        sizes.append(total_len)
    
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        print(f"  Chunk sizes: avg={avg_size:.0f}, min={min_size}, max={max_size}")
    
    # Embedding eligibility
    embedding_allowed = sum(1 for c in chunks if c.get('metadata', {}).get('embedding_allowed', False))
    print(f"  Embedding allowed: {embedding_allowed}/{len(chunks)}")
    
    # Summary
    print(f"\n{BLUE}{'='*80}{RESET}")
    if all_passed:
        print_success("All quality checks passed!")
        return True
    else:
        print_error(f"Validation failed - fix issues before Phase 4")
        return False


def main():
    """Run complete pipeline test."""
    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}PDF Processing Pipeline Test{RESET}")
    print(f"{GREEN}Phase 1 -> Phase 2 -> Phase 3{RESET}")
    print(f"{GREEN}{'='*80}{RESET}\n")
    
    # Step 0: Check server
    if not check_server():
        sys.exit(1)
    
    # Step 1: Phase 1+2
    phase12_data = test_phase12_extraction()
    if not phase12_data:
        print_error("Phase 1+2 failed. Cannot proceed to Phase 3.")
        sys.exit(1)
    
    # Step 2: Phase 3
    phase3_data = test_phase3_chunking(phase12_data)
    if not phase3_data:
        print_error("Phase 3 failed.")
        sys.exit(1)
    
    # Step 3: Verify
    verify_chunks(phase3_data)
    
    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}Pipeline test completed successfully!{RESET}")
    print(f"{GREEN}{'='*80}{RESET}\n")


if __name__ == "__main__":
    main()

