"""
Speaker Diarization Module
==========================
Interfaces with pyannote.audio to partition audio files by speaker identity.
Provides a defensive fallback generator if pyannote or PyTorch are unavailable.
"""

import os
from pathlib import Path
from typing import List, Optional
from transcription.audio_processor import DiarizationSegment


class SpeakerDiarizer:
    """
    Coordinates speaker diarization using pyannote.audio or mock fallback.
    """

    def diarize(self, audio_path: Path, hf_token: Optional[str] = None) -> List[DiarizationSegment]:
        """
        Runs the speaker diarization pipeline on the specified audio file.

        Args:
            audio_path: Path to the target audio file.
            hf_token: Hugging Face API token (required for pyannote.audio pretrained model).

        Returns:
            A list of DiarizationSegment objects.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        token = hf_token or os.environ.get("HF_TOKEN")

        try:
            # Attempt to import pyannote.audio dynamically
            from pyannote.audio import Pipeline
            import torch

            if not token:
                print("Warning: Hugging Face token (HF_TOKEN) not found. Falling back to mock diarization.")
                return self._diarize_fallback(audio_path)

            print(f"Loading pyannote.audio pipeline utilizing HF token...")
            # Initialize pipeline (using CPU by default unless CUDA is available)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            print(f"Running speaker diarization on: {audio_path.name}")
            diarization = pipeline(str(audio_path))

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    DiarizationSegment(
                        start=float(turn.start),
                        end=float(turn.end),
                        speaker=str(speaker)
                    )
                )
            return segments

        except ImportError:
            print("Warning: pyannote.audio or torch is not installed. Falling back to mock diarization.")
            return self._diarize_fallback(audio_path)
        except Exception as e:
            print(f"Error during pyannote.audio diarization ({e}). Falling back to mock diarization.")
            return self._diarize_fallback(audio_path)

    def _diarize_fallback(self, audio_path: Path) -> List[DiarizationSegment]:
        """
        Generates simulated speaker turns for offline verification and testing.
        """
        # Alternate speakers every 15 seconds as a reasonable mock for city council proceedings
        duration = 120.0  # Default mock duration of 2 minutes
        segment_length = 15.0
        segments = []
        
        current_time = 0.0
        speaker_idx = 1
        
        while current_time < duration:
            end_time = min(current_time + segment_length, duration)
            segments.append(
                DiarizationSegment(
                    start=current_time,
                    end=end_time,
                    speaker=f"SPEAKER_0{speaker_idx}"
                )
            )
            current_time = end_time
            # Alternate between SPEAKER_01 and SPEAKER_02
            speaker_idx = 2 if speaker_idx == 1 else 1

        return segments
