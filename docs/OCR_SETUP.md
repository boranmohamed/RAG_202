# OCR Setup Guide

This guide explains how to set up Tesseract OCR with Arabic and English language support for the OCR pipeline feature.

## Prerequisites

The OCR pipeline uses **Tesseract OCR** engine with **Arabic (ara)** and **English (eng)** language packs for bilingual document processing.

## Installation

### Windows

#### 1. Install Tesseract Binary

Download and install Tesseract from the official repository:

**Option A: Using Installer (Recommended)**
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (e.g., `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)
3. During installation, make sure to select:
   - ✅ Arabic language data
   - ✅ English language data
4. Add Tesseract to your PATH:
   - Default install location: `C:\Program Files\Tesseract-OCR`
   - Add to System PATH: `C:\Program Files\Tesseract-OCR`

**Option B: Using Chocolatey**
```bash
choco install tesseract
```

#### 2. Verify Installation

```bash
tesseract --version
```

Expected output:
```
tesseract 5.x.x
 leptonica-1.x.x
  ...
```

#### 3. Check Language Packs

```bash
tesseract --list-langs
```

Expected output should include:
```
List of available languages (2):
ara
eng
```

If Arabic is missing, download manually:
1. Go to: https://github.com/tesseract-ocr/tessdata
2. Download `ara.traineddata`
3. Place it in: `C:\Program Files\Tesseract-OCR\tessdata\`

### Linux (Ubuntu/Debian)

```bash
# Install Tesseract
sudo apt update
sudo apt install tesseract-ocr

# Install Arabic language pack
sudo apt install tesseract-ocr-ara

# Install English language pack (usually included)
sudo apt install tesseract-ocr-eng
```

### macOS

```bash
# Install Tesseract
brew install tesseract

# Install Arabic language pack
brew install tesseract-lang
```

## Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- `pytesseract` - Python wrapper for Tesseract
- `opencv-python` - Image preprocessing
- `Pillow` - Image handling
- `pdfplumber` - PDF extraction with OCR support

## Testing Your Setup

### Quick Test

Create a test script `test_ocr_setup.py`:

```python
import pytesseract
from PIL import Image

# Test Tesseract installation
try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract version: {version}")
except Exception as e:
    print(f"❌ Tesseract not found: {e}")
    exit(1)

# Test language packs
try:
    langs = pytesseract.get_languages()
    print(f"✅ Available languages: {langs}")
    
    if 'ara' not in langs:
        print("❌ Arabic language pack not found!")
    else:
        print("✅ Arabic language pack installed")
    
    if 'eng' not in langs:
        print("❌ English language pack not found!")
    else:
        print("✅ English language pack installed")
except Exception as e:
    print(f"❌ Error checking languages: {e}")
```

Run the test:
```bash
python test_ocr_setup.py
```

### Full Pipeline Test

Run the OCR pipeline test with the provided PDF:

```bash
python test_ocr_pipeline.py
```

## Troubleshooting

### Issue: "pytesseract.pytesseract.TesseractNotFoundError"

**Solution:**
- Windows: Add Tesseract to PATH or set explicitly in Python:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```
- Linux/Mac: Install tesseract-ocr package

### Issue: "Requested languages were not found"

**Solution:**
- Verify language packs with: `tesseract --list-langs`
- Manually download missing `.traineddata` files from:
  https://github.com/tesseract-ocr/tessdata
- Place in tessdata directory:
  - Windows: `C:\Program Files\Tesseract-OCR\tessdata\`
  - Linux: `/usr/share/tesseract-ocr/4.00/tessdata/`
  - macOS: `/usr/local/share/tessdata/`

### Issue: Poor OCR Quality

**Solutions:**
1. **Increase DPI**: Use higher `ocr_dpi` parameter (default: 300)
   ```python
   # In API request
   "ocr_dpi": 400  # or 600 for very small text
   ```

2. **Check Image Quality**: Ensure source PDF is not too low resolution

3. **Preprocessing**: The pipeline uses adaptive thresholding tuned for Arabic diacritics and tables. If quality is still poor, you can adjust preprocessing parameters in `phase1_ocr_extractor_pdfplumber.py`

### Issue: Slow OCR Processing

**Solutions:**
1. **Use lower DPI** for faster processing (trade-off with quality):
   ```python
   "ocr_dpi": 200  # faster but less accurate
   ```

2. **Enable parallel processing**: OCR processes pages sequentially by default. For large documents, consider implementing page-level parallelization.

3. **Use embedded text when available**: The pipeline automatically detects and uses embedded text (no OCR) when available, only using OCR for scanned pages.

## Performance Tips

### DPI Selection Guide

| DPI | Use Case | Processing Time | Quality |
|-----|----------|----------------|---------|
| 150 | Draft/testing | Fast | Poor |
| 200 | Quick processing | Medium | Fair |
| 300 | Default (recommended) | Medium | Good |
| 400 | High quality needed | Slow | Very Good |
| 600 | Small text/diacritics | Very Slow | Excellent |

### OCR Best Practices

1. **Hybrid Approach**: The pipeline automatically uses embedded text when available and only runs OCR on scanned pages
2. **Batch Processing**: Process multiple PDFs in parallel at the file level
3. **Caching**: Cache OCR results to avoid reprocessing the same pages
4. **Quality Check**: Validate OCR output by checking text density and language detection confidence

## Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root:

```env
# Tesseract binary path (Windows only, if not in PATH)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Default OCR DPI
OCR_DPI=300

# OCR language preference
OCR_LANGS=ara+eng
```

## Support

For issues or questions:
1. Check Tesseract documentation: https://tesseract-ocr.github.io/
2. Arabic OCR troubleshooting: https://github.com/tesseract-ocr/tesseract/wiki/FAQ
3. Review the OCR pipeline code: `features/process/infrastructure/phase1_ocr_extractor_pdfplumber.py`

