"""
Audio Downloader and Extractor Module
=====================================
Handles downloading municipal audio records and extracting audio tracks
from downloaded meeting videos using native system FFmpeg commands.
"""

import shutil
import subprocess
from pathlib import Path
import urllib.request


class AudioDownloader:
    """
    Downloads audio files and extracts audio tracks from video files.
    """

    def __init__(self, timeout: int = 15):
        """
        Initializes the downloader with standard timeout and headers.
        """
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def download_audio(self, url: str, output_path: Path) -> Path:
        """
        Downloads a direct audio file (e.g., MP3, WAV) from a URL.

        Args:
            url: The audio file link.
            output_path: Path specifying where to save the audio file.

        Returns:
            The Path of the downloaded file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers=self.headers)

        try:
            print(f"Downloading audio: {url}...")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                with open(output_path, "wb") as out_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
            return output_path
        except Exception as e:
            raise IOError(f"Failed to download audio file from {url}: {e}")

    def extract_audio_from_video(self, video_path: Path, output_path: Path) -> Path:
        """
        Extracts the audio track from a video file using the system's FFmpeg utility.
        Formats the output audio to 16kHz mono (ideal for LLM transcription pipelines).

        Args:
            video_path: Path to the downloaded video file.
            output_path: Target path for the extracted audio file.

        Returns:
            The Path of the generated audio file.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Source video file not found: {video_path}")

        # Check if ffmpeg command exists on the PATH
        if not shutil.which("ffmpeg"):
            raise EnvironmentError(
                "FFmpeg executable not found on system PATH. "
                "Please install FFmpeg to enable audio extraction from video files."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command:
        # -y (overwrite output files)
        # -i video_path (input file)
        # -vn (disable video recording)
        # -ar 16000 (set audio sampling rate to 16000 Hz)
        # -ac 1 (set audio channels to 1 - mono)
        # -ab 64k (set audio bitrate)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-ab", "64k",
            str(output_path)
        ]

        try:
            print(f"Extracting audio from {video_path} to {output_path}...")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise IOError(
                f"FFmpeg extraction failed with exit code {e.returncode}.\n"
                f"Stderr: {e.stderr}"
            )
