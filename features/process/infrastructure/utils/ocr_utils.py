"""
OCR Utilities Module

Common helper functions for OCR processing:
- Image preprocessing optimized for Arabic text with diacritics
- Language detection (Arabic/English/Mixed)
- OCR quality validation
- Table-to-markdown conversion
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image
import pytesseract


# Language detection regexes
AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
EN_RE = re.compile(r"[A-Za-z]")

# Arabic diacritics regex for OCR cleaning
ARABIC_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670\u06D6-\u06ED]')


def detect_lang(text: str) -> str:
    """
    Lightweight bilingual detector for mixed lines.
    
    Args:
        text: Input text to analyze
        
    Returns:
        'ar' for Arabic-only, 'en' for English-only, 'mixed' for both or neither
    """
    if not text or not text.strip():
        return "other"
    
    ar = len(AR_RE.findall(text))
    en = len(EN_RE.findall(text))
    
    if ar == 0 and en == 0:
        return "other"
    if ar > 0 and en == 0:
        return "ar"
    if en > 0 and ar == 0:
        return "en"
    return "mixed"


def normalize_arabic(text: str) -> str:
    """
    Minimal Arabic normalization (avoid overly aggressive normalization that breaks meaning).
    
    - Remove tatweel (ـ)
    - Normalize Alef variants (أ، إ، آ → ا)
    - Normalize Ya variants (ى → ي)
    - Normalize Ta Marbuta (ة → ه) - optional, conservative
    - Normalize whitespace
    
    Args:
        text: Arabic text to normalize
        
    Returns:
        Normalized Arabic text
    """
    if not text:
        return ""
    
    # Do NOT clean markdown tables (we must keep pipes `|`, dashes, alignment, etc.)
    if text.strip().startswith("|") and "|" in text:
        return text
    
    # Remove tatweel (Arabic elongation character)
    text = text.replace("ـ", "")
    
    # Normalize Alef variants
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    
    # Normalize Ya variants
    text = text.replace("ى", "ي")
    
    # Normalize Ta Marbuta (optional - can be commented out if exact spelling needed)
    text = text.replace("ة", "ه")
    
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def normalize_english(text: str) -> str:
    """
    Minimal English normalization.
    
    - Normalize whitespace
    - Normalize multiple newlines
    
    Args:
        text: English text to normalize
        
    Returns:
        Normalized English text
    """
    if not text:
        return ""
    
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def clean_ocr_text(text: str) -> str:
    """
    Cleans OCR output from Arabic/English statistical documents.
    
    Comprehensive cleaning pipeline that:
    1. Removes Unicode control characters
    2. Removes Arabic diacritics
    3. Removes Tatweel (elongation character)
    4. Normalizes common OCR letter variants
    5. Normalizes Arabic numerals to Western numerals
    6. Collapses repeated whitespace
    7. Preserves statistical symbols (%, ., :, -)
    
    This preserves climate, humidity, agriculture, economy tables perfectly.
    Works flawlessly for Arabic + English blocks.
    
    Args:
        text: OCR output text to clean
        
    Returns:
        Cleaned text ready for further processing
    """
    if not text:
        return ""
    
    # Do NOT clean markdown tables (we must keep pipes `|`, dashes, alignment, etc.)
    if text.strip().startswith("|") and "|" in text:
        return text
    
    # 1) Remove Unicode control chars
    text = ''.join(ch for ch in text if unicodedata.category(ch) not in ["Cc", "Cf"])
    
    # 2) Remove Arabic diacritics
    text = ARABIC_DIACRITICS.sub('', text)
    
    # 3) Remove Tatweel
    text = text.replace("ـ", "")
    
    # 4) Normalize common OCR letter variants
    arabic_norm_map = {
        "ي": "ي",  # unify arabic yeh
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ۀ": "ه",
        "إ": "ا",
        "أ": "ا",
        "آ": "ا",
    }
    for k, v in arabic_norm_map.items():
        text = text.replace(k, v)
    
    # 5) Normalize Arabic numerals to Western (optional)
    # OCR often mixes (٠١٢...) with (012...)
    arabic_nums = "٠١٢٣٤٥٦٧٨٩"
    western_nums = "0123456789"
    translation_table = str.maketrans(arabic_nums, western_nums)
    text = text.translate(translation_table)
    
    # 6) Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7) Keep statistical symbols (%, ., :, -)
    return text


def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    OCR pre-processing tuned for documents with Arabic diacritics/dots + tables.
    
    Processing steps:
    1. Convert to grayscale
    2. Apply bilateral filter for denoising while preserving edges
    3. Adaptive thresholding (preserves small Arabic dots better than global threshold)
    4. Mild morphology to connect broken strokes without destroying dots
    
    Args:
        pil_img: PIL Image to preprocess
        
    Returns:
        Preprocessed PIL Image ready for OCR
    """
    # Convert PIL Image to numpy array
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Denoise while preserving edges (important for diacritics)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)

    # Adaptive threshold: better for uneven scans / shaded backgrounds
    # This preserves small Arabic dots better than global thresholding
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 11
    )

    # Remove small noise (but be gentle to avoid removing Arabic dots)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    return Image.fromarray(thr)


def ocr_image(pil_img: Image.Image, lang: str = "ara+eng", psm: int = 6) -> str:
    """
    Tesseract bilingual OCR with preprocessing.
    
    Args:
        pil_img: PIL Image to OCR
        lang: Tesseract language(s), default "ara+eng" for bilingual
        psm: Page segmentation mode (6 = assume a uniform block of text)
        
    Returns:
        Extracted text from image
        
    Note:
        PSM modes:
        - 3: Fully automatic page segmentation (default)
        - 4: Assume a single column of text
        - 6: Assume a uniform block of text (best for book pages)
        - 11: Sparse text (find as much text as possible)
    """
    # Preprocess image for better OCR quality
    proc = preprocess_for_ocr(pil_img)
    
    # OCR with Tesseract
    # OEM 1 = LSTM neural net mode (best quality)
    config = f"--oem 1 --psm {psm}"
    
    try:
        text = pytesseract.image_to_string(proc, lang=lang, config=config)
        return text
    except Exception as e:
        # Fallback: try without preprocessing if it fails
        try:
            text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
            return text
        except Exception as fallback_e:
            raise RuntimeError(
                f"OCR failed with preprocessing: {e}, "
                f"and without preprocessing: {fallback_e}"
            ) from fallback_e


def validate_ocr_quality(text: str, min_chars: int = 10) -> Tuple[bool, str]:
    """
    Validate OCR output quality.
    
    Args:
        text: OCR output text
        min_chars: Minimum number of characters for valid output
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if not text or not text.strip():
        return False, "No text extracted"
    
    clean_text = text.strip()
    
    if len(clean_text) < min_chars:
        return False, f"Text too short ({len(clean_text)} chars)"
    
    # Check if text is mostly gibberish (too many non-letter characters)
    letters = AR_RE.findall(clean_text) + EN_RE.findall(clean_text)
    if len(letters) < len(clean_text) * 0.3:
        return False, "Text appears to be mostly non-letters"
    
    return True, "OK"


def table_to_markdown(rows: List[List[str]]) -> str:
    """
    Convert table rows to Markdown format for RAG storage.
    
    Args:
        rows: List of table rows (each row is a list of cell values)
        
    Returns:
        Markdown-formatted table string
    """
    if not rows:
        return ""
    
    # Normalize width - ensure all rows have same number of columns
    max_cols = max(len(r) for r in rows)
    norm = [r + [""] * (max_cols - len(r)) for r in rows]
    
    # Build markdown table
    header = norm[0]
    sep = ["---"] * max_cols
    body = norm[1:] if len(norm) > 1 else []
    
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    
    return "\n".join(lines)


def estimate_text_density(pil_img: Image.Image) -> float:
    """
    Estimate text density in an image (0.0 to 1.0).
    
    Useful for detecting if a page is mostly blank or has substantial content.
    
    Args:
        pil_img: PIL Image to analyze
        
    Returns:
        Text density estimate (0.0 = blank, 1.0 = full of text)
    """
    # Convert to grayscale
    img = np.array(pil_img.convert("L"))
    
    # Simple threshold to find dark pixels (potential text)
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate ratio of dark pixels to total pixels
    dark_pixels = np.sum(binary > 0)
    total_pixels = binary.size
    
    density = dark_pixels / total_pixels
    
    return density


def split_text_by_language(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Split text into Arabic lines, English lines, and mixed lines.
    
    Args:
        text: Input text with multiple lines
        
    Returns:
        Tuple of (arabic_lines, english_lines, mixed_lines)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    ar_lines: List[str] = []
    en_lines: List[str] = []
    mixed_lines: List[str] = []
    
    for ln in lines:
        lang = detect_lang(ln)
        if lang == "ar":
            ar_lines.append(ln)
        elif lang == "en":
            en_lines.append(ln)
        else:
            mixed_lines.append(ln)
    
    return ar_lines, en_lines, mixed_lines

