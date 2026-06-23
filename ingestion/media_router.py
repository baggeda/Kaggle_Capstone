"""
Media Ingestion Router
======================
Routes incoming URLs or local files to the correct downloader/extractor
based on file extension or URL patterns, and outputs structured Pydantic metadata.
"""

import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ingestion.video_downloader import VideoDownloader
from ingestion.audio_downloader import AudioDownloader
from ingestion.pdf_extractor import PDFExtractor
from ingestion.summary_extractor import SummaryExtractor


class IngestedMediaResult(BaseModel):
    """
    Standard data model representing the output of any media ingestion process.
    """
    media_type: str = Field(..., description="Categorization: 'video', 'audio', 'pdf', or 'html'.")
    source: str = Field(..., description="The original source URL or file path.")
    local_path: Optional[str] = Field(None, description="The local path of the saved file (if downloaded).")
    extracted_text: Optional[str] = Field(None, description="The plain text extracted from the document (if applicable).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata captured from the media parser.")


class IngestionRouter:
    """
    Inspects resource parameters and routes them to appropriate download/extraction modules.
    """

    def __init__(self, download_dir: Path):
        """
        Initializes the router with a download folder and helper objects.
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_downloader = VideoDownloader()
        self.audio_downloader = AudioDownloader()
        self.pdf_extractor = PDFExtractor()
        self.summary_extractor = SummaryExtractor()

    def _is_url(self, target: str) -> bool:
        """
        Helper checking if input is a web URL.
        """
        parsed = urllib.parse.urlparse(target)
        return bool(parsed.scheme and parsed.netloc)

    def ingest(self, target: str) -> IngestedMediaResult:
        """
        Determines the format, runs the required download/extraction step,
        and returns an IngestedMediaResult.

        Args:
            target: A local file path string or remote URL link.

        Returns:
            An IngestedMediaResult model containing the parsed metadata.
        """
        is_url = self._is_url(target)
        parsed = urllib.parse.urlparse(target)
        filename = Path(parsed.path).name if is_url else Path(target).name
        suffix = Path(filename).suffix.lower()

        # Handle remote URLs first (downloading)
        local_path = None
        if is_url:
            local_path = self.download_dir / filename
            if not local_path.suffix:
                # Add fallback extensions if missing in URL
                if "m3u8" in target:
                    local_path = local_path.with_suffix(".ts")
                else:
                    local_path = local_path.with_suffix(".html")
            
            # Route download
            if suffix in {".mp4", ".avi", ".mkv"} or "m3u8" in target:
                local_path = self.video_downloader.download_video(target, local_path)
            elif suffix in {".mp3", ".wav", ".aac"}:
                local_path = self.audio_downloader.download_audio(target, local_path)
            elif suffix == ".pdf":
                # Download pdf using standard urllib
                req = urllib.request.Request(target, headers=self.video_downloader.headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    local_path.write_bytes(resp.read())
            else:
                # Treat as HTML summary portal page
                req = urllib.request.Request(target, headers=self.video_downloader.headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read().decode("utf-8", errors="ignore")
                    local_path.write_text(content, encoding="utf-8")
        else:
            local_path = Path(target)

        # Post-process or extract text based on local file extension
        media_type = "html"
        extracted_text = None
        metadata = {}

        # Safely resolve file size if it exists
        file_size = 0
        if local_path.exists():
            try:
                file_size = local_path.stat().st_size
            except Exception:
                pass

        file_suffix = local_path.suffix.lower()
        if file_suffix in {".mp4", ".avi", ".mkv", ".ts"}:
            media_type = "video"
            metadata = {"file_size_bytes": file_size}
        elif file_suffix in {".mp3", ".wav", ".aac"}:
            media_type = "audio"
            metadata = {"file_size_bytes": file_size}
        elif file_suffix == ".pdf":
            media_type = "pdf"
            extracted_data = self.pdf_extractor.extract_structured_text(local_path)
            extracted_text = extracted_data["full_text"]
            metadata = extracted_data["metadata"]
        else:
            # HTML or text summaries
            media_type = "html"
            # Extract summary from file only if it exists
            if local_path.exists():
                extracted_text = self.summary_extractor.extract_summary_from_file(local_path)
            metadata = {"file_size_bytes": file_size}

        return IngestedMediaResult(
            media_type=media_type,
            source=target,
            local_path=str(local_path),
            extracted_text=extracted_text,
            metadata=metadata
        )
