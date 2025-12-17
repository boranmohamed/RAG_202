"""Test the updated regex patterns."""
import re

def _parse_chapter_from_header(text: str):
    if not text:
        return None
    
    # Pattern 1: "Chapter N : Name [digits] Arabic : N الفصل"
    match = re.search(r'Chapter\s+\d+\s*:\s*([^0-9]+?)\s+\d+', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Arabic format "الفصل N : Name"
    match = re.search(r'الفصل\s+\d+\s*:\s*(.+?)(?:\s+\d+|$)', text)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Fallback patterns
    patterns = [
        r'(?:Chapter|الفصل)\s*\d+\s*:\s*(.+)',
        r'(.+?)\s*:\s*(?:Chapter|الفصل)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def _parse_section_from_header(text: str):
    if not text:
        return None
    
    # Pattern 1: "Section N : Name [digits] Arabic : N القسم"
    match = re.search(r'Section\s+\d+\s*:\s*([^0-9]+?)\s+\d+', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Arabic format "القسم N : Name"
    match = re.search(r'القسم\s+\d+\s*:\s*(.+?)(?:\s+\d+|$)', text)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Fallback patterns
    patterns = [
        r'(?:Section|القسم)\s*\d+\s*:\s*(.+)',
        r'(.+?)\s*:\s*(?:Section|القسم)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

# Test cases from the actual PDF
test_cases = [
    ("Chapter 1 : General Information 27 ةـماـعلا تاموـلعلام : 1 لـصفلا", "chapter", "General Information"),
    ("Chapter 2 : Climate 39 خاـنــلام : 2 لــصفلا", "chapter", "Climate"),
    ("Section 1 : Climate 41 خاــنلام : 1 مسقلا", "section", "Climate"),
    ("Section 2 : Population 53 ناـكسلا : 2 مسقلا", "section", "Population"),
]

results = []
results.append("Testing regex patterns:\n")
for text, typ, expected in test_cases:
    if typ == "chapter":
        result = _parse_chapter_from_header(text)
    else:
        result = _parse_section_from_header(text)
    
    status = "[OK]" if result == expected else "[FAIL]"
    results.append(f"{status} {typ}: '{text[:50]}...'")
    results.append(f"     Expected: '{expected}'")
    results.append(f"     Got:      '{result}'")
    results.append("")

# Write to file
with open('regex_test_result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print("Results written to: regex_test_result.txt")

