"""Quick script to check if chapter/section metadata was extracted."""
import json

with open('phase12_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

blocks = data['blocks']
blocks_with_chapter = [b for b in blocks if b.get('chapter')]
blocks_with_section = [b for b in blocks if b.get('section')]

output = []
output.append(f"Total blocks: {len(blocks)}")
output.append(f"Blocks with chapter: {len(blocks_with_chapter)}")
output.append(f"Blocks with section: {len(blocks_with_section)}")

if blocks_with_chapter:
    output.append("\nSample blocks with chapter:")
    for b in blocks_with_chapter[:5]:
        output.append(f"  Page {b.get('pageNumber')}: chapter='{b.get('chapter')}', section='{b.get('section')}', type={b.get('type')}")
else:
    output.append("\n[ERROR] NO chapter metadata found!")
    output.append("\nLet's check for potential chapter headers:")
    headers = [b for b in blocks[:500] if b.get('type') == 'header']
    output.append(f"Found {len(headers)} headers in first 500 blocks")
    for h in headers[:10]:
        content = h.get('content', {}) or {}
        ar_text = (content.get('ar') or '')
        en_text = (content.get('en') or '')
        output.append(f"\n  Page {h.get('pageNumber')} (type={h.get('type')}):")
        output.append(f"    AR: {ar_text}")
        output.append(f"    EN: {en_text}")

# Write to file
with open('metadata_check_result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("Results written to: metadata_check_result.txt")

