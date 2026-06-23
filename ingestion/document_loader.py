"""
Document Ingestion Module
=========================
Handles the loading, parsing, and preprocessing of legislative documents,
including PDFs and plain text files.
"""

from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field
from pypdf import PdfReader


class IngestedDocument(BaseModel):
    """
    Data model representing a document that has been successfully loaded and parsed.
    """
    file_path: str = Field(..., description="The absolute or relative path of the source file.")
    content: str = Field(..., description="The full plain text content extracted from the document.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs containing parsed metadata (e.g., page count, title)."
    )


class DocumentLoader:
    """
    Loader for converting physical files (PDFs, plain text) into structured IngestedDocument models.
    """

    def load_pdf(self, path: Path) -> IngestedDocument:
        """
        Parses a PDF document, extracts its text, and collects basic metadata.

        Args:
            path: Path to the target PDF file.

        Returns:
            An IngestedDocument object containing the parsed text and page counts.
        """
        # Ensure the file exists before processing
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        reader = PdfReader(path)
        extracted_text = []

        # Extract text page by page
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)

        full_content = "\n".join(extracted_text)
        metadata = {
            "page_count": len(reader.pages),
            "file_size_bytes": path.stat().st_size,
        }

        return IngestedDocument(
            file_path=str(path),
            content=full_content,
            metadata=metadata
        )

    def load_text(self, path: Path) -> IngestedDocument:
        """
        Reads a raw text file and returns a structured document.

        Args:
            path: Path to the target text file.

        Returns:
            An IngestedDocument object with text content.
        """
        if not path.exists():
            raise FileNotFoundError(f"Text file not found at: {path}")

        # Read content using UTF-8 encoding
        content = path.read_text(encoding="utf-8")
        metadata = {
            "file_size_bytes": path.stat().st_size,
        }

        return IngestedDocument(
            file_path=str(path),
            content=content,
            metadata=metadata
        )
