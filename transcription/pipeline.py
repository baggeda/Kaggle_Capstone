"""
Diarized Transcription Pipeline
===============================
Runs OpenAI Whisper transcription and speaker diarization, merging
them via a maximum timestamp overlap algorithm to generate a formatted transcript:
[HH:MM:SS] Speaker: Dialogue.
"""

import os
from pathlib import Path
from typing import List, Optional
from openai import OpenAI

from transcription.audio_processor import (
    AudioMetadata,
    AudioProcessor,
    DiarizationSegment,
    TranscriptionSegment,
    DialogueLine,
    DiarizedTranscript
)
from transcription.diarizer import SpeakerDiarizer


class DiarizedTranscriptionPipeline:
    """
    Coordinates Whisper transcription and speaker diarization into a unified dialogue.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self.processor = AudioProcessor()
        self.diarizer = SpeakerDiarizer()

    def _format_timestamp(self, seconds: float) -> str:
        """
        Formats floating-point seconds into [HH:MM:SS] format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

    def _merge_transcription_and_diarization(
        self,
        whisper_segments: List[TranscriptionSegment],
        diarizer_segments: List[DiarizationSegment]
    ) -> List[DialogueLine]:
        """
        Assigns each Whisper segment to the speaker with the maximum overlap.
        """
        dialogue_lines = []

        for w_seg in whisper_segments:
            best_speaker = "UNKNOWN_SPEAKER"
            max_overlap = -1.0

            for d_seg in diarizer_segments:
                # Calculate the overlap boundary
                overlap_start = max(w_seg.start, d_seg.start)
                overlap_end = min(w_seg.end, d_seg.end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > max_overlap and overlap > 0.0:
                    max_overlap = overlap
                    best_speaker = d_seg.speaker

            # Format time
            timestamp_str = self._format_timestamp(w_seg.start)
            dialogue_lines.append(
                DialogueLine(
                    timestamp=timestamp_str,
                    speaker=best_speaker,
                    text=w_seg.text
                )
            )

        return dialogue_lines

    def run(
        self,
        audio_path: Path,
        openai_api_key: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> DiarizedTranscript:
        """
        Executes the transcription and speaker diarization pipeline on an audio/video file.

        Args:
            audio_path: Path to the target audio/video file.
            openai_api_key: OpenAI API key.
            hf_token: Hugging Face access token for pyannote.audio.

        Returns:
            A DiarizedTranscript model containing lines and formatted full text.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Source file not found: {audio_path}")

        # Extract metadata
        metadata = self.processor.extract_metadata(audio_path)

        # 1. Fetch Diarization Segments
        diarizer_segments = self.diarizer.diarize(audio_path, hf_token=hf_token)

        # 2. Fetch Whisper Transcription Segments
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        whisper_segments = []

        if not api_key:
            print("Warning: OpenAI API Key not found. Running with mock transcription.")
            whisper_segments = self._get_mock_whisper_segments()
        else:
            try:
                print("Calling OpenAI Whisper API for transcript segments...")
                client = OpenAI(api_key=api_key)
                
                # Perform transcription with segment granularity
                with open(audio_path, "rb") as audio_file:
                    resp = client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )
                
                # Parse segments
                if hasattr(resp, "segments"):
                    for seg in resp.segments:
                        whisper_segments.append(
                            TranscriptionSegment(
                                start=float(seg.start),
                                end=float(seg.end),
                                text=str(seg.text).strip()
                            )
                        )
                else:
                    print("OpenAI response did not contain segment timestamps. Using full text.")
                    # Fallback to single chunk
                    whisper_segments = [
                        TranscriptionSegment(
                            start=0.0,
                            end=metadata.duration_seconds or 120.0,
                            text=resp.text
                        )
                    ]
            except Exception as e:
                print(f"Error calling OpenAI Whisper API ({e}). Falling back to mock transcription.")
                whisper_segments = self._get_mock_whisper_segments()

        # 3. Merge raw transcription segments and speaker turns
        dialogue_lines = self._merge_transcription_and_diarization(whisper_segments, diarizer_segments)

        # 4. Generate final dialogue transcript text
        formatted_dialogue = []
        for line in dialogue_lines:
            formatted_dialogue.append(f"{line.timestamp} {line.speaker}: {line.text}")

        full_transcript_text = "\n".join(formatted_dialogue)

        return DiarizedTranscript(
            metadata=metadata,
            lines=dialogue_lines,
            full_text=full_transcript_text
        )

    def _get_mock_whisper_segments(self) -> List[TranscriptionSegment]:
        """
        Provides mock text segments for local verification.
        """
        return [
            TranscriptionSegment(start=1.0, end=8.0, text="Good evening everyone. The City Council meeting is now in session."),
            TranscriptionSegment(start=9.5, end=14.0, text="Thank you, Mayor. Roll call shows all council members are present."),
            TranscriptionSegment(start=16.0, end=28.0, text="Excellent. First item is the public hearing for the Greer park expansion program."),
            TranscriptionSegment(start=30.0, end=43.0, text="I would like to voice a comment. Local citizens are concerned about parking spaces around the park."),
            TranscriptionSegment(start=45.0, end=58.0, text="Thank you for that comment. We will direct the planning department to review parking allocations."),
            TranscriptionSegment(start=60.0, end=75.0, text="Let's proceed to vote on the budget reconciliation bill. All in favor say aye."),
            TranscriptionSegment(start=77.0, end=80.0, text="Aye."),
            TranscriptionSegment(start=82.0, end=95.0, text="The ayes have it. The budget bill passes. Let's adjourn the meeting."),
        ]
