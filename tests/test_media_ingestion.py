"""
Unit Tests for Media Ingestion Modules
======================================
Verifies video downloads, HLS stream merging, audio downloads, FFmpeg wrappers,
PDF text parsing, HTML summary stripping, and unified media routing.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from ingestion.video_downloader import VideoDownloader
from ingestion.audio_downloader import AudioDownloader
from ingestion.pdf_extractor import PDFExtractor
from ingestion.summary_extractor import SummaryExtractor
from ingestion.media_router import IngestionRouter


def test_video_download_direct(tmp_path):
    """
    Asserts direct video downloading writes file chunks correctly.
    """
    downloader = VideoDownloader()
    output_file = tmp_path / "test.mp4"

    mock_resp = MagicMock()
    # Return 2 chunks then empty
    mock_resp.read.side_effect = [b"chunk1", b"chunk2", b""]
    mock_resp.__enter__.return_value = mock_resp

    with patch('urllib.request.urlopen', return_value=mock_resp):
        res = downloader.download_video("https://example.com/test.mp4", output_file)
        assert res == output_file
        assert output_file.read_bytes() == b"chunk1chunk2"


def test_video_download_hls(tmp_path):
    """
    Asserts HLS stream downloader parses playlist and merges .ts chunks.
    """
    downloader = VideoDownloader()
    output_file = tmp_path / "stream.mp4"

    # Mock responses: 1st for playlist, 2nd and 3rd for TS chunks
    mock_playlist = """#EXTM3U
#EXT-X-VERSION:3
#EXTINF:10.0,
segment1.ts
#EXTINF:10.0,
https://example.com/segment2.ts
"""
    mock_m3u8_resp = MagicMock()
    mock_m3u8_resp.read.return_value = mock_playlist.encode("utf-8")
    mock_m3u8_resp.__enter__.return_value = mock_m3u8_resp

    mock_ts_resp = MagicMock()
    mock_ts_resp.read.return_value = b"TS_DATA_CHUNK"
    mock_ts_resp.__enter__.return_value = mock_ts_resp

    # URL open mock side effects
    def urlopen_side_effect(req, *args, **kwargs):
        # Determine if it's the playlist request or TS requests
        url = req.full_url if hasattr(req, 'full_url') else req
        if "stream.m3u8" in url:
            return mock_m3u8_resp
        return mock_ts_resp

    with patch('urllib.request.urlopen', side_effect=urlopen_side_effect):
        res = downloader.download_video("https://example.com/stream.m3u8", output_file)
        # Verify the TS output path or final renamed path
        assert res == output_file
        assert output_file.exists()
        # It merged 2 segments
        assert output_file.read_bytes() == b"TS_DATA_CHUNKTS_DATA_CHUNK"


def test_audio_download_and_extraction(tmp_path):
    """
    Asserts direct audio download and FFmpeg audio-from-video extraction.
    """
    downloader = AudioDownloader()
    output_audio = tmp_path / "test.mp3"

    # Test download
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"audio_bytes", b""]
    mock_resp.__enter__.return_value = mock_resp

    with patch('urllib.request.urlopen', return_value=mock_resp):
        res = downloader.download_audio("https://example.com/test.mp3", output_audio)
        assert res == output_audio
        assert output_audio.read_bytes() == b"audio_bytes"

    # Test extraction
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"dummy_video")
    extracted_audio = tmp_path / "extracted.wav"

    with patch('shutil.which', return_value="/usr/bin/ffmpeg"), \
         patch('subprocess.run') as mock_run:
        
        mock_run.return_value = MagicMock(returncode=0)
        
        res_extract = downloader.extract_audio_from_video(video_file, extracted_audio)
        
        assert res_extract == extracted_audio
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-vn" in args
        assert "-ar" in args


def test_pdf_extraction(tmp_path):
    """
    Asserts PDF Extractor parses reader text and metadata correctly.
    """
    pdf_path = tmp_path / "agenda.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy")

    extractor = PDFExtractor()

    # Mock pypdf reader
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "City Council Agenda Items"
    mock_reader.pages = [mock_page]
    mock_reader.metadata = {"/Title": "June Council Agenda", "/Author": "City Clerk"}

    with patch('ingestion.pdf_extractor.PdfReader', return_value=mock_reader):
        res = extractor.extract_structured_text(pdf_path)
        
        assert res["file_path"] == str(pdf_path)
        assert "City Council Agenda Items" in res["full_text"]
        assert len(res["pages"]) == 1
        assert res["metadata"]["title"] == "June Council Agenda"
        assert res["metadata"]["author"] == "City Clerk"


def test_summary_html_extraction():
    """
    Asserts HTML parser strips styling, headers, script blocks, and collects text.
    """
    extractor = SummaryExtractor()
    sample_html = """
    <html>
        <head>
            <style>body { color: red; }</style>
            <script>console.log("hello");</script>
        </head>
        <body>
            <h1>Meeting Brief</h1>
            <p>The council voted to adopt the <strong>2026 budget</strong> amendment.</p>
            <div>Second block.</div>
        </body>
    </html>
    """

    clean_text = extractor.extract_summary_from_html(sample_html)
    
    assert "Meeting Brief" in clean_text
    assert "The council voted to adopt the 2026 budget amendment." in clean_text
    assert "Second block." in clean_text
    assert "console.log" not in clean_text
    assert "color: red" not in clean_text


def test_media_router(tmp_path):
    """
    Asserts routing of files/URLs to correct processing structures.
    """
    router = IngestionRouter(download_dir=tmp_path / "router_downloads")

    # Mock out downloaders & extractors to isolate router logic
    with patch.object(router.video_downloader, 'download_video') as mock_v_dl, \
         patch.object(router.audio_downloader, 'download_audio') as mock_a_dl, \
         patch.object(router.pdf_extractor, 'extract_structured_text') as mock_pdf, \
         patch.object(router.summary_extractor, 'extract_summary_from_file') as mock_html_ext:

        # Configure mocks
        mock_v_dl.return_value = tmp_path / "router_downloads" / "video.mp4"
        mock_a_dl.return_value = tmp_path / "router_downloads" / "audio.mp3"
        mock_pdf.return_value = {
            "full_text": "Extracted PDF content",
            "metadata": {"title": "PDF Title"}
        }
        mock_html_ext.return_value = "Extracted HTML content"

        # 1. Route Video URL
        res_v = router.ingest("https://example.gov/stream.m3u8")
        assert res_v.media_type == "video"
        mock_v_dl.assert_called_once()

        # 2. Route PDF URL
        # We write dummy bytes to intercept the urlopen file write
        mock_pdf_resp = MagicMock()
        mock_pdf_resp.read.return_value = b"pdf_data"
        mock_pdf_resp.__enter__.return_value = mock_pdf_resp
        
        with patch('urllib.request.urlopen', return_value=mock_pdf_resp):
            res_pdf = router.ingest("https://example.gov/doc.pdf")
            assert res_pdf.media_type == "pdf"
            assert res_pdf.extracted_text == "Extracted PDF content"
            mock_pdf.assert_called_once()
