"""Debug script to check what's happening with header extraction."""
import sys
sys.path.insert(0, '.')

from features.process.infrastructure.pdf_structured_extractor_pymupdf import extract_structured_pdf

# Extract first 10 pages
pages = extract_structured_pdf("publicationpdfar1765273617.pdf")

# Check pages 1-10 for headers
output = []
output.append("Checking first 10 pages for headers and chapter/section metadata:\n")
for page in pages[:10]:
    page_num = page.get('pageNumber', 0)
    blocks = page.get('blocks', [])
    
    headers = [b for b in blocks if b.get('type') == 'header']
    if headers:
        output.append(f"Page {page_num}: {len(headers)} headers")
        for h in headers:
            content = h.get('content', {})
            output.append(f"  - chapter={h.get('chapter')}, section={h.get('section')}")
            output.append(f"    AR: {(content.get('ar') or '')}")
            output.append(f"    EN: {(content.get('en') or '')}")
    
    # Check if any blocks have chapter/section
    blocks_with_meta = [b for b in blocks if b.get('chapter') or b.get('section')]
    if blocks_with_meta:
        output.append(f"  {len(blocks_with_meta)} blocks have metadata")

output.append("\nDone")

with open('extraction_debug_result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("Results written to: extraction_debug_result.txt")

