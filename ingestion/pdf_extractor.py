"""
PDF Text Extractor Module
=========================
Handles structured text and metadata extraction from local government agenda and
proposal PDFs.
"""

from pathlib import Path
from typing import Dict, Any, List
from pypdf import PdfReader


class PDFExtractor:
    """
    Extracts structured layout content and metadata from agenda/proposal PDFs.
    """

    def extract_structured_text(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Reads a PDF file, extracts raw text page-by-page, and collects metadata.

        Args:
            pdf_path: Path to the target PDF file.

        Returns:
            A dictionary containing structured page-by-page text, full text, and metadata.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        reader = PdfReader(pdf_path)
        pages_data: List[Dict[str, Any]] = []
        full_text_list = []

        # Parse text and character count page-by-page
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()
            
            pages_data.append({
                "page_number": page_num,
                "text": cleaned_text,
                "character_count": len(cleaned_text)
            })
            if cleaned_text:
                full_text_list.append(cleaned_text)

        # Collect standard PDF metadata
        raw_metadata = reader.metadata or {}
        cleaned_metadata = {
            "title": str(raw_metadata.get("/Title", "")),
            "author": str(raw_metadata.get("/Author", "")),
            "creator": str(raw_metadata.get("/Creator", "")),
            "producer": str(raw_metadata.get("/Producer", "")),
            "page_count": len(reader.pages),
            "file_size_bytes": pdf_path.stat().st_size
        }

        return {
            "file_path": str(pdf_path),
            "full_text": "\n\n--- Page Break ---\n\n".join(full_text_list),
            "pages": pages_data,
            "metadata": cleaned_metadata
        }
