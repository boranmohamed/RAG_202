"""
Entry point for the FastAPI application.

Run with (from project root):

    uvicorn main:app --reload

Currently exposes the \"process PDF\" feature via:

    POST /api/v1/pdfs/process
"""

from fastapi import FastAPI

from features.process.presentation.api import router as process_pdf_router


app = FastAPI(title="RAG PDF Processing API", version="0.1.0")

app.include_router(process_pdf_router)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}



