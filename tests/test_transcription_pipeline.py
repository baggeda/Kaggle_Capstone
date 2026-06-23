"""
Unit Tests for the Diarized Transcription Pipeline
===================================================
Verifies timestamp conversion, speaker alignment overlap merging,
and mock end-to-end execution of the transcription pipeline.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from transcription.audio_processor import DiarizationSegment, TranscriptionSegment
from transcription.pipeline import DiarizedTranscriptionPipeline


def test_timestamp_formatting():
    """
    Asserts floating seconds are formatted precisely to [HH:MM:SS].
    """
    pipeline = DiarizedTranscriptionPipeline()
    
    assert pipeline._format_timestamp(0.0) == "[00:00:00]"
    assert pipeline._format_timestamp(45.2) == "[00:00:45]"
    assert pipeline._format_timestamp(3665.9) == "[01:01:05]"
    assert pipeline._format_timestamp(86399.0) == "[23:59:59]"


def test_merge_transcription_and_diarization():
    """
    Verifies that Whisper segments are assigned to the speaker with the maximum overlap.
    """
    pipeline = DiarizedTranscriptionPipeline()

    # Whisper segments (what was said)
    whisper_segs = [
        # Segment 1: from 10.0 to 20.0 seconds
        TranscriptionSegment(start=10.0, end=20.0, text="Hello world"),
        # Segment 2: from 25.0 to 30.0 seconds
        TranscriptionSegment(start=25.0, end=30.0, text="Adjourned")
    ]

    # Speaker turns (who spoke when)
    diarizer_segs = [
        # Speaker A overlaps Segment 1 by 2 seconds (10.0 to 12.0)
        DiarizationSegment(start=0.0, end=12.0, speaker="SpeakerA"),
        # Speaker B overlaps Segment 1 by 8 seconds (12.0 to 20.0)
        DiarizationSegment(start=12.0, end=24.0, speaker="SpeakerB"),
        # Speaker C covers segment 2 entirely (25.0 to 30.0)
        DiarizationSegment(start=24.5, end=35.0, speaker="SpeakerC")
    ]

    dialogue = pipeline._merge_transcription_and_diarization(whisper_segs, diarizer_segs)

    assert len(dialogue) == 2
    # Segment 1 must map to SpeakerB (8s overlap > 2s overlap)
    assert dialogue[0].speaker == "SpeakerB"
    assert dialogue[0].timestamp == "[00:00:10]"
    assert dialogue[0].text == "Hello world"

    # Segment 2 must map to SpeakerC (5s overlap)
    assert dialogue[1].speaker == "SpeakerC"
    assert dialogue[1].timestamp == "[00:00:25]"
    assert dialogue[1].text == "Adjourned"


def test_pipeline_end_to_end(tmp_path):
    """
    Asserts end-to-end execution of the pipeline (mocking external API calls).
    """
    audio_file = tmp_path / "meeting.wav"
    audio_file.write_bytes(b"dummy audio wav data")

    pipeline = DiarizedTranscriptionPipeline()

    # Mock Whisper response
    mock_whisper_segment_1 = MagicMock(start=2.0, end=6.0, text="Proposal adopted.")
    mock_whisper_segment_2 = MagicMock(start=8.0, end=15.0, text="Thank you, Clerk.")
    
    mock_response = MagicMock()
    mock_response.segments = [mock_whisper_segment_1, mock_whisper_segment_2]

    # Mock diarizer response
    mock_diarizer_segs = [
        DiarizationSegment(start=0.0, end=7.0, speaker="SPEAKER_A"),
        DiarizationSegment(start=7.0, end=20.0, speaker="SPEAKER_B")
    ]

    with patch('transcription.pipeline.OpenAI') as mock_openai_class, \
         patch.object(pipeline.diarizer, 'diarize', return_value=mock_diarizer_segs):
        
        mock_client = mock_openai_class.return_value
        mock_client.audio.transcriptions.create.return_value = mock_response

        # Execute
        transcript = pipeline.run(audio_file, openai_api_key="sk-testkey")

        # Assertions
        assert transcript is not None
        assert len(transcript.lines) == 2
        
        # Segment 1 (2.0 to 6.0) maps to SPEAKER_A (overlap: 4.0s)
        assert transcript.lines[0].speaker == "SPEAKER_A"
        assert transcript.lines[0].timestamp == "[00:00:02]"
        assert transcript.lines[0].text == "Proposal adopted."

        # Segment 2 (8.0 to 15.0) maps to SPEAKER_B (overlap: 7.0s)
        assert transcript.lines[1].speaker == "SPEAKER_B"
        assert transcript.lines[1].timestamp == "[00:00:08]"
        assert transcript.lines[1].text == "Thank you, Clerk."

        # Verify full formatted dialogue text structure
        expected_full_text = (
            "[00:00:02] SPEAKER_A: Proposal adopted.\n"
            "[00:00:08] SPEAKER_B: Thank you, Clerk."
        )
        assert transcript.full_text == expected_full_text
