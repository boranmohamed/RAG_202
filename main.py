"""
Entry point for the FastAPI application.

Run with (from project root):

    uvicorn main:app --reload

Currently exposes the \"process PDF\" feature via:

    POST /api/v1/pdfs/process
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI

from features.process.presentation.api import router as process_pdf_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_pipeline.log', encoding='utf-8')
    ]
)

# Set specific log levels for modules
logging.getLogger("features.process.infrastructure.pdf_structured_extractor_pymupdf").setLevel(logging.DEBUG)
logging.getLogger("features.process.infrastructure.phase1_ocr_extractor_pdfplumber").setLevel(logging.DEBUG)
logging.getLogger("features.process.infrastructure.phase2_preprocessor_bilingual").setLevel(logging.DEBUG)
logging.getLogger("features.process.infrastructure.phase3_chunker_bilingual").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("Starting RAG PDF Processing API")

app = FastAPI(title="RAG PDF Processing API", version="0.1.0")

app.include_router(process_pdf_router)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}



