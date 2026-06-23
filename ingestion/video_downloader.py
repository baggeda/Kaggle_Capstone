"""
Video Downloader Module
=======================
Handles downloading municipal meeting video streams.
Supports direct file downloading (e.g., MP4) and native HLS (.m3u8) stream
parsing and chunk merging using standard Python libraries.
"""

import os
import re
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


class VideoDownloader:
    """
    Downloads video files and streams from municipal portals.
    """

    def __init__(self, timeout: int = 15):
        """
        Initializes the downloader with standard timeout and user-agent headers.
        """
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def download_video(self, url: str, output_path: Path) -> Path:
        """
        Downloads a video from a URL. Detects if the source is HLS (.m3u8) or a direct video file.

        Args:
            url: The source URL of the video or playlist.
            output_path: Path specifying where to save the video file.

        Returns:
            The Path of the downloaded file.
        """
        # Ensure parent folder exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parsed_url = urllib.parse.urlparse(url)
        path_lower = parsed_url.path.lower()

        if ".m3u8" in path_lower or "m3u8" in parsed_url.query.lower():
            print(f"Detected HLS stream: {url}. Beginning segment merging...")
            return self._download_hls(url, output_path)
        else:
            print(f"Detected direct video file: {url}. Downloading...")
            return self._download_direct(url, output_path)

    def _download_direct(self, url: str, output_path: Path) -> Path:
        """
        Downloads a direct video file in binary chunks.
        """
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                with open(output_path, "wb") as out_file:
                    # Write in 1MB chunks to prevent memory bloat
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
            return output_path
        except Exception as e:
            raise IOError(f"Failed to download direct video from {url}: {e}")

    def _download_hls(self, playlist_url: str, output_path: Path) -> Path:
        """
        Downloads an HLS (.m3u8) playlist stream by parsing it, downloading the
        individual .ts chunks, and merging them into a single file.
        """
        req = urllib.request.Request(playlist_url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                playlist_content = response.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise IOError(f"Failed to fetch HLS playlist from {playlist_url}: {e}")

        # Parse segment links from the playlist (lines not starting with #)
        segments = []
        for line in playlist_content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # Resolve relative segment URLs relative to the playlist URL
                segment_url = urllib.parse.urljoin(playlist_url, line)
                segments.append(segment_url)

        if not segments:
            raise ValueError(f"No TS segments found in the playlist at {playlist_url}")

        print(f"Found {len(segments)} segments to download.")

        # Ensure we write as a merged file (typically .ts format)
        # If output path is configured as .mp4, we will write as .ts first,
        # since standard concatenation creates a valid TS stream.
        ts_output_path = output_path
        if output_path.suffix.lower() == ".mp4":
            # Native concatenation results in TS format. We name it .ts during processing,
            # or keep it as .ts since it contains transport stream packets.
            ts_output_path = output_path.with_suffix(".ts")

        try:
            with open(ts_output_path, "wb") as out_file:
                for idx, seg_url in enumerate(segments):
                    print(f"Downloading segment {idx + 1}/{len(segments)}...")
                    seg_req = urllib.request.Request(seg_url, headers=self.headers)
                    with urllib.request.urlopen(seg_req, timeout=self.timeout) as seg_resp:
                        out_file.write(seg_resp.read())

            # Rename back to final path if they were different
            if ts_output_path != output_path:
                if output_path.exists():
                    os.remove(output_path)
                os.rename(ts_output_path, output_path)

            return output_path
        except Exception as e:
            raise IOError(f"Failed during HLS stream downloading or merging: {e}")
